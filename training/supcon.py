import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from typing import Literal

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils_training
from cls_head_val import Classification_Head
from models import eegnet, eeginception, conformer, eegwrn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # print(f'Batch Size: {batch_size}')
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # print(f'Mask: {mask}')
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(f'Tile Mask: {mask}')
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # print(f'Logits mask: {mask}')

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print(f'exp_logits: {exp_logits}')
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(f'log_prob: {log_prob}')
        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        # print(f'mask_pos_pairs: {mask_pos_pairs}')

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        # print(f'mean_log_prob_pos: {mean_log_prob_pos}')
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print(f'Loss1: {loss}')
        loss = loss.view(anchor_count, batch_size).mean()
        # print(f'Loss2: {loss}')

        return loss
    
    
class SupCon(pl.LightningModule):
    def __init__(self,
                 model:Literal['EEGNet', 'EEGInception', 'Conformer', 'EEGWRN'],
                 ckpt_dir:str,
                 gpu=int,
                 lr:float=0.001,
                 hidden_dim:int=128,
                 temperature:float=0.1, 
                 weight_decay:float=5e-5, 
                 max_epochs:int=100,
                 val:Literal['KNN', 'Kmeans', 'classifier'] = 'classifier'):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0
        
        
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.auc = torchmetrics.AUROC(task='binary')
        self.confmat = torchmetrics.ConfusionMatrix(task='binary')
        
        self.contrastive_loss = SupConLoss(temperature=self.hparams.temperature)
        # Encoder
        if self.hparams.model == 'EEGNet':
            
            self.encoder = eegnet.EEGNet(n_chans=8)
            self.in_features = self.encoder.dense.in_features
            print(f'In Features dim: {self.in_features}')
            self.encoder.dense = nn.Identity() 
            
        elif self.hparams.model == 'EEGInception':
            
            self.encoder = eeginception.EEGInception()
            self.in_features = self.encoder.dense.in_features
            print(f'In Features dim: {self.in_features}')
            self.encoder.dense = nn.Identity() 
            
        elif self.hparams.model == 'Conformer':
            
            self.encoder = conformer.Conformer()
            self.in_features = self.encoder[2].dense[0].in_features
            print(f'In Features dim: {self.in_features}')
            self.encoder[2].dense = nn.Identity()
            
        elif self.hparams.model == 'EEGWRN':
            
            self.encoder = eegwrn.EEGWRN()
            self.in_features = self.encoder.dense[0].in_features
            print(f'In Features dim: {self.in_features}')
            self.encoder.dense = nn.Identity() 
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        self.projections = []
        self.embeddings = []
        self.labels = []
     
        
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.5, 0.99) if self.hparams.model == 'Conformer' else (0.9, 0.999)
        )
        
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50
        )
        
        return [optimizer], [lr_scheduler]     
        
    def forward(self, x):
        embeddings, _ = self.encoder(x)
        return embeddings, self.projection_head(embeddings)
        
    def supcon_loss(self, batch):

        views, labels = batch
    
        _, features_1 = self.forward(views[0])
        _, features_2 = self.forward(views[1]) 
        _, features_3 = self.forward(views[2]) 
        _, features_4 = self.forward(views[3]) 
        
        features_1 = F.normalize(features_1, p=2, dim=-1)
        features_2 = F.normalize(features_2, p=2, dim=-1)
        features_3 = F.normalize(features_3, p=2, dim=-1)
        features_4 = F.normalize(features_4, p=2, dim=-1)
        
        features = torch.cat([features_1.unsqueeze(1), features_2.unsqueeze(1), 
                              features_3.unsqueeze(1), features_4.unsqueeze(1)], dim=1)
        
        # print(labels.squeeze(1))
        loss = self.contrastive_loss(features, labels=labels.squeeze(1))
        
        return loss
    
    def training_step(self, batch, batch_idx):

        loss = self.supcon_loss(batch)
        self.log('train_loss', loss)
        
        if batch_idx % 100 == 0:
            x = []
            for dim in range(len(batch[0])):
                samples, _ = batch
                # print(samples[dim].shape)
                # print(samples[dim][0].shape)
                x.append(samples[dim][0].cpu())
            
            fig = utils_training.plot_views(x)
            self.logger.experiment.add_figure("train_sample_views", fig, self.current_epoch)
            x.clear()       
        return loss
    
    def k_means_val(self, normalized_embeddings, labels):
        
        embeddings_cpu = normalized_embeddings.cpu().numpy()
        labels_cpu = labels.squeeze().cpu().numpy()
        
        kmeans = MiniBatchKMeans(n_clusters=2, batch_size=1024 ,random_state=42)
        clusters = kmeans.fit_predict(embeddings_cpu)
        
        # Classification Performance metrics
        ari = adjusted_rand_score(labels_cpu, clusters)
        silhouette = silhouette_score(embeddings_cpu, clusters)
        
        # Analyse Cluster Morphology
        compactness = kmeans.inertia_
        cluster_centers = kmeans.cluster_centers_
        inter_cluster_distances = np.linalg.norm(cluster_centers[0] - cluster_centers[1])
        
        # T-SNE
        fig = utils_training.plot_tsne(embeddings_cpu, labels_cpu, clusters)
        
        metrics = {"val_ari": ari,
                "val_silhouette": silhouette,
                "val_compactness": compactness,
                "val_inter_cluster_distance": inter_cluster_distances}
        # Log metrics
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        self.logger.experiment.add_figure("T-SNEs", fig, self.current_epoch)
        return metrics
        
    def knn_val(self, embeddings, projections, labels):
        
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        embeddings_cpu = normalized_embeddings.cpu().numpy()
        labels_cpu = labels.squeeze().cpu().numpy()
        
        train_val_embeddings, test_val_embeddings, train_val_labels, test_val_labels = train_test_split(
            embeddings_cpu, labels_cpu, test_size=0.2, random_state=42)
        
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn.fit(train_val_embeddings, train_val_labels)        
        predictions = knn.predict(test_val_embeddings)
        
        random_indices = np.random.randint(0, len(embeddings), size=1200)
        tsnes = utils_training.plot_tsne(embeddings[random_indices], labels[random_indices], projections[random_indices])
        self.logger.experiment.add_figure("T-SNEs", tsnes, self.current_epoch)
        
        predictions = torch.tensor(predictions, device=labels.device)
        labels = torch.tensor(test_val_labels, device=labels.device)
        
        acc = self.accuracy(predictions, labels)
        f1_score = self.f1_score(predictions, labels)
        recall = self.recall(predictions, labels)
        precision = self.precision(predictions, labels)
        auc = self.auc(predictions, labels)  
        confmat = self.confmat(predictions, labels)
                
        metrics = {'val_accuracy':acc,
                'val_f1score':f1_score,
                'val_recall':recall,
                'val_precision': precision,
                'val_auc':auc}
        
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
        confusion_matrix = utils_training.plot_confusion_matrix(confmat, metrics)
        self.logger.experiment.add_figure('Confusion Matrix', confusion_matrix, self.current_epoch)
        return metrics
       
    def classifier_eval(self, embeddings, labels):
        
        cls_head = Classification_Head(self.in_features)
        
        # split the data 
        embeddings = embeddings.detach()
        labels = labels.float()
        dataset = TensorDataset(embeddings, labels)
        train_size = int(0.6 * len(dataset))
        val_size = int(0.1 * train_size)
        test_size = len(dataset) - train_size -val_size
        train_dataset, test_dataset = random_split(dataset, 
                                                   [train_size + val_size, test_size], 
                                                   generator=torch.Generator().manual_seed(42))
        
        train_dataset, val_dataset = random_split(train_dataset,
                                                  [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        for batch in test_loader:
            test_data, test_labels = batch
            tsnes = utils_training.plot_tsne(test_data, test_labels)
            self.logger.experiment.add_figure("T-SNEs", tsnes, self.current_epoch)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_val_loss',
            dirpath=os.path.join(self.hparams.ckpt_dir, 'cls_validation'),
            filename=f'cls_head_{self.hparams.model}_{self.hparams.temperature}',
            save_last=True,
            mode='min'
        )
        
        early_stop = EarlyStopping(monitor='val_val_loss', patience=30)
        
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[self.hparams.gpu],
            max_epochs=100,
            callbacks=[checkpoint_callback, early_stop]
        )
        
        trainer.fit(cls_head, train_loader, val_loader)
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)

        self.log_dict(test_results[0], on_epoch=True, on_step=False, prog_bar=True)
        
        return test_results
    
    def validation_step(self, batch, batch_idx):
        signals, labels = batch
        
        embeddings, projections = self.forward(signals)
        
        self.projections.append(projections)
        self.embeddings.append(embeddings)
        self.labels.append(labels)
        
        return {'projections': projections, 'embeddings':embeddings, 'labels': labels}
    
    def on_validation_epoch_end(self):
        
        projections = torch.cat(self.projections, dim=0).squeeze()
        labels = torch.cat(self.labels, dim=0).squeeze()
        embeddings = torch.cat(self.embeddings, dim=0).squeeze()
        
        self.projections.clear()
        self.labels.clear()
        self.embeddings.clear()
        
        if self.hparams.val == 'KNN':
            metrics = self.knn_val(embeddings, projections, labels) 
        elif self.hparams.val == 'KMeans':
            metrics = self.k_means_val(embeddings, labels)
        elif self.hparams.val == 'classifier':
            embeddings.requires_grad_()
            metrics = self.classifier_eval(embeddings, labels)
        return metrics
        
        