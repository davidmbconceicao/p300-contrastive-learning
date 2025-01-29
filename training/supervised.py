import os
import sys
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from typing import Literal

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils_training
from evaluation import utils_evaluation
from models import eegnet, eeginception, conformer, eegwrn


class Supervised(pl.LightningModule):
    def __init__(self,
                 model: Literal['EEGNet', 'EEGInception', 'Conformer', 'EEGWRN'],
                 results_file:str = None,
                 optimizer: Literal['Adam', 'AdamW', 'SGD'] = 'Adam',
                 lr:float = 1e-3,
                 weight_decay:float = 1e-5,
                 n_chans:int=8, 
                 eval_dataset: Literal['GIB_UVA', 'ALS', 'BCI']='GIB_UVA',
    ):
        super(Supervised, self).__init__()
        
        self.save_hyperparameters()
        
        if self.hparams.model == 'EEGNet':
            self.model = eegnet.EEGNet(n_chans=n_chans)
        elif self.hparams.model == 'EEGInception':
            self.model = eeginception.EEGInception(n_chans=n_chans)
        elif self.hparams.model == 'Conformer':
            self.model = conformer.Conformer()
        elif self.hparams.model == 'EEGWRN':
            self.model = eegwrn.EEGWRN()
            
        if eval_dataset != 'GIB_UVA':
            pos_weight = torch.tensor([5 / 1], dtype=torch.float32)
            self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_func = nn.BCEWithLogitsLoss()
        
        # self.loss_func = nn.BCEWithLogitsLoss()
        
        self.logits = []
        self.embeddings = []
        self.labels = []
        self.codes = []
        
        # Metrics
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.auc = torchmetrics.AUROC(task='binary')
        self.confmat = torchmetrics.ConfusionMatrix(task='binary')
        
    def forward(self, x):
        embeddings, logits = self.model(x)
        return embeddings, logits
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.hparams.lr,  
                                   betas=(0.5, 0.99) if self.hparams.model == 'Conformer' else (0.9, 0.999),
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(),
                                   lr=self.hparams.lr,  
                                   betas=self.hparams.betas,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(),
                                   lr=self.hparams.lr,  
                                   momentum=self.hparams.momentum,
                                   weight_decay=self.hparams.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=0.005)
            return [optimizer], [scheduler]
        return optimizer

    def common_step(self, batch, batch_idx):
        x, y = batch
        embeddings, logits = self.forward(x)
        loss = self.loss_func(logits, y)
        return loss, logits, y
    
    def training_step(self, batch, batch_idx):
        loss, logits, y = self.common_step(batch, batch_idx)
        
        self.log_dict(
            {'train_loss': loss}, 
            on_epoch=True,
            on_step=False,
            prog_bar=True
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        signals, labels = batch
        
        embeddings, logits = self.forward(signals)
        
        self.logits.append(logits)
        self.embeddings.append(embeddings)
        self.labels.append(labels)
        
        return {'logits': logits, 'embeddings':embeddings, 'labels': labels}
    
    def on_validation_epoch_end(self):
        logits = torch.cat(self.logits, dim=0).squeeze()
        labels = torch.cat(self.labels, dim=0).squeeze()
        embeddings = torch.cat(self.embeddings, dim=0).squeeze()
        
        self.logits.clear()
        self.labels.clear()
        self.embeddings.clear()
        
        # plot TSNEs
        random_indices = np.random.randint(0, len(embeddings), size=1200)
        tsnes = utils_training.plot_tsne(embeddings[random_indices], labels[random_indices])
        self.logger.experiment.add_figure("T-SNEs", tsnes, self.current_epoch)
        
        loss = self.loss_func(logits, labels)
        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)
        recall = self.recall(logits, labels)
        precision = self.precision(logits, labels)
        auc = self.auc(logits, labels)
        confmat = self.confmat(logits, labels)
                
        metrics = {'val_loss': loss,
                'val_accuracy':acc,
                'val_f1score':f1,
                'val_recall':recall,
                'val_precision': precision,
                'val_auc':auc}
        
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
        confusion_matrix = utils_training.plot_confusion_matrix(confmat, metrics)
        self.logger.experiment.add_figure('Confusion Matrix', confusion_matrix, self.current_epoch)
        return loss

    
    # def validation_step(self, batch, batch_idx):
    #     loss, logits, y = self.common_step(batch, batch_idx)
        
    #     acc = self.accuracy(logits, y)
    #     f1 = self.f1_score(logits, y)
    #     recall = self.recall(logits, y)
    #     precision = self.precision(logits, y)
    #     auc = self.auc(logits, y)
    #     confmat = self.confmat(logits, y)
                
    #     metrics = {'val_loss': loss,
    #             'val_accuracy':acc,
    #             'val_f1score':f1,
    #             'val_recall':recall,
    #             'val_precision': precision,
    #             'val_auc':auc}
        
    #     self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
    #     confusion_matrix = utils_training.plot_confusion_matrix(confmat, metrics)
    #     self.logger.experiment.add_figure('Confusion Matrix', confusion_matrix, self.current_epoch)
    #     return loss
    
    def character_recognition_per_trials(self, logits, codes, n_trials,
                                         n_chars, target_string):
                
        char_recog_acc = []
        for trials in n_trials:
            print()
            print(f"Number of Trials: {trials}")
                    
            pred = utils_evaluation.predict_string(logits, codes, trials, n_chars)
            acc = utils_evaluation.calculate_string_accuracy(pred, target_string)
                    
            print(f"{target_string} -> Target String")
            print(f"{pred} -> Predicted String")
            print(f"Accuracy -> {acc}")
                    
            char_recog_acc.append(acc)
            
        print(f'Character Recognition Accuracy Over Trials -> {char_recog_acc}')
        # fig = utils_training.plot_results_across_trials(char_recog_acc, n_trials)
        # self.logger.experiment.add_figure('Results across trials', fig)
        return char_recog_acc

    def binary_classification(self, logits, labels):
        acc = self.accuracy(logits, labels)
        print(f'Accuracy: {acc}')
        f1_score = self.f1_score(logits, labels)
        print(f'F1_Score: {f1_score}')
        recall = self.recall(logits, labels)
        print(f'Recall: {recall}')
        precision = self.precision(logits, labels)
        print(f'Precision: {precision}')
        auc = self.auc(logits, labels)  
        print(f'AUC: {auc}')
        confmat = self.confmat(logits, labels)
                
        metrics = {'Accuracy':acc,
                'F1_Score':f1_score,
                'Recall':recall,
                'Precision': precision,
                'AUC':auc}
        
        fig = utils_training.plot_confusion_matrix(confmat, metrics)
        self.logger.experiment.add_figure('Confusion Matrix', fig) 
        return metrics     
       
    
    def test_step(self, batch, batch_idx):
        
        if self.hparams.eval_dataset == 'ALS':
            test_data, labels, codes = batch
            _, logits = self.forward(test_data)
            
            self.logits.append(logits)
            self.labels.append(labels)
            self.codes.append(codes)
            
            return {'logits': logits, 'codes':codes, 'labels': labels}
        
        if self.hparams.eval_dataset == 'BCI':
            test_data, codes = batch
            _, logits = self.forward(test_data)
            
            self.logits.append(logits)
            self.codes.append(codes)
            
            return {'logits': logits, 'codes': codes}
        
    def on_test_epoch_end(self):
            
        logits = torch.cat(self.logits, dim=0).squeeze()
        labels = torch.cat(self.labels, dim=0).squeeze() if len(self.labels) > 0 else None
        codes = torch.cat(self.codes, dim=0)
        
        self.logits.clear()
        self.labels.clear()
        self.codes.clear()
        
        n_chars, n_trials = (35, range(1,11)) if self.hparams.eval_dataset == 'ALS' else  (100, range(1,16))
        target_string = self.trainer.datamodule.target_string
        
        print('Starting Character Recognition Evaluation')
        cra_results = self.character_recognition_per_trials(logits, codes, n_trials, 
                                                n_chars, target_string)
        
        with open(self.hparams.results_file, "a") as file:
            file.write(f'\n{cra_results}')
        
        print('Starting Binary Evaluation')   
        self.binary_classification(logits, labels) if self.hparams.eval_dataset == 'ALS' else None
        
        return cra_results
            
        
        
        
            
        