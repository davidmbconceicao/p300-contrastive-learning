import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import adjusted_rand_score, silhouette_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from typing import Literal
from itertools import permutations

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils_training
from cls_head_val import Classification_Head
from models import eegnet, eeginception, conformer, eegwrn


class SimCLR(pl.LightningModule):
    def __init__(self, 
                 model:Literal['EEGNet', 'EEGInception', 'Conformer', 'EEGWRN'],
                 gpu:int, 
                 ckpt_dir:str,
                 lr:float,
                 hidden_dim:int=128, 
                 temperature:float=0.1, 
                 weight_decay:float=5e-5, 
                 max_epochs:int=200,
                 val:Literal['KNN', 'Kmeans', 'classifier']='classifier'):
        
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0
        
        self.projections = []
        self.labels = []
        self.embeddings = []
        
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.auc = torchmetrics.AUROC(task='binary')
        self.confmat = torchmetrics.ConfusionMatrix(task='binary')
        
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
            
        
    def define_param_groups(self, optimizer_name):
        def exclude_from_wd_and_adaptation(name):
            if 'bn' in name:
                return True
            if optimizer_name == 'lars' and 'bias' in name:
                return True

        param_groups = [
            {
                'params': [p for name, p in self.named_parameters() if not exclude_from_wd_and_adaptation(name)],
                'weight_decay': self.hparams.weight_decay,
                'layer_adaptation': True,
            },
            {
                'params': [p for name, p in self.named_parameters() if exclude_from_wd_and_adaptation(name)],
                'weight_decay': 0.,
                'layer_adaptation': False,
            },
        ]
        return param_groups

        
    def configure_optimizers(self):
        param_groups = self.define_param_groups('adam')
        optimizer = optim.Adam(
            param_groups, 
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
        """Forward pass returning the encoded features only."""
        embeddings, _ = self.encoder(x)
        # print(f'Embeddings Shape:  {embeddings.shape}')
        return embeddings, self.projection_head(embeddings)

       
    def info_nce_loss(self, batch):
            
        samples = torch.cat(batch, dim=0)
        # encode samples
        _, feats = self.forward(samples)
        
        # Compute Cosine Similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
            
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
            
        # Logging loss
        self.log(f"train_loss", nll)

        return nll

    def nt_bxent_loss(self, batch):
        """ For multi-positive views case
        From https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
        """
        
        samples = torch.cat(batch, dim=0).to(self.device)
        # print(f'Samples Shape: {samples.shape}')
        
        _ , x = self.forward(samples)
        # print(f'X Shape: {x.shape}')

        batch_size = batch[0].shape[0]
        # print(f'Batch Size: {batch_size}')
        pos_indices = self.generate_pos_indices(batch_size).to(self.device)
        # print(f'Pos_Indices1: {pos_indices}')
        
        temperature = self.hparams.temperature

        # Add indexes of the principal diagonal elements to pos_indices
        pos_indices = torch.cat([
            pos_indices,
            torch.arange(x.size(0), device=self.device).reshape(x.size(0), 1).expand(-1, 2),
        ], dim=0)
        # print(f'Pos_Indices2: {pos_indices}')
        
        # Ground truth labels
        target = torch.zeros(x.size(0), x.size(0), device=self.device)
        target[pos_indices[:,0], pos_indices[:,1]] = 1.0
        # print(f'Target Matrix: {target}')

        # Cosine similarity
        xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
        # Set logit of diagonal element to "inf" signifying complete
        # correlation. sigmoid(inf) = 1.0 so this will work out nicely
        # when computing the Binary Cross Entropy Loss.
        xcs[torch.eye(x.size(0), device=self.device).bool()] = float("inf")
        # print(f'xcs: {xcs}')
        

        # Standard binary cross entropy loss. We use binary_cross_entropy() here and not
        # binary_cross_entropy_with_logits() because of https://github.com/pytorch/pytorch/issues/102894
        # The method *_with_logits() uses the log-sum-exp-trick, which causes inf and -inf values
        # to result in a NaN result.
        loss = F.binary_cross_entropy((xcs / temperature).sigmoid(), target, reduction="none")
        # print(f'loss: {loss}')
        target_pos = target.bool()
        target_neg = ~target_pos
        
        loss_pos = torch.zeros(x.size(0), x.size(0), device=self.device).masked_scatter(target_pos, loss[target_pos])
        # print(f'loss_pos: {loss_pos}')
        loss_neg = torch.zeros(x.size(0), x.size(0), device=self.device).masked_scatter(target_neg, loss[target_neg])
        # print(f'loss_neg: {loss_neg}')
        loss_pos = loss_pos.sum(dim=1)
        loss_neg = loss_neg.sum(dim=1)
        num_pos = target.sum(dim=1)
        num_neg = x.size(0) - num_pos
        
        loss = ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()
        # print(f'loss: {loss}')
        self.log(f"train_loss", loss)

        return loss
    
    def generate_pos_indices(self, batch_size, n_views=4):
        pos_indices = []
        for i in range(batch_size):
            values = [i, batch_size + i, 2*batch_size + i, 3*batch_size + i]
            all_permutations = list(permutations(values, 2))
            pos_indices.extend(all_permutations)
            
        return torch.tensor(pos_indices, device=self.device)

    def training_step(self, batch, batch_idx):
        loss =  self.nt_bxent_loss(batch)
        # print(f'Batch Len: {len(batch)}')
        # print(f'Batch[0] Shape: {batch[0].shape}')
        
        if batch_idx % 100 == 0:
            x = []
            for dim in range(len(batch)):
                x.append(batch[dim][0].cpu())
            
            fig = utils_training.plot_views(x)
            self.logger.experiment.add_figure("train_sample_views", fig, self.current_epoch)
            
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
        # print(test_results)
        self.log_dict(test_results[0], on_epoch=True, on_step=False, prog_bar=True)

        # confmat = self.confmat(test_results[1]['logits'], test_results[1]['y'])
        # confusion_matrix = utils_training.plot_confusion_matrix(confmat, test_results[0])
        # self.logger.experiment.add_figure('Confusion Matrix',confusion_matrix, self.current_epoch)
        
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