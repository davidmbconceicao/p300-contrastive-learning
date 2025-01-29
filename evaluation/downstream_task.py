import torch
import numpy as np
import pandas as pd
import os
import sys

import pytorch_lightning as pl
from torch import nn, optim
import torchmetrics
from typing import Literal

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils_evaluation
from training import utils_training
from training import simclr, supcon, supervised

class ClassificationHead(nn.Module):
    def __init__(self, 
                #  n_layers, 
                in_features,
                linear:bool = False
                ):
        super(ClassificationHead, self).__init__()
        
        # layers = []
        # if n_layers == 1:
        #     # Single linear layer case
        #     layers.append(nn.Linear(in_features, 1))
        # else:
        #     for i in range(n_layers):
        #         out_features = in_features // (i + 2)
        #         layers.append(nn.Linear(in_features, out_features))
                
        #         if i < n_layers - 1:
        #             layers.append(nn.BatchNorm1d(out_features))
        #             layers.append(nn.ReLU(inplace=True))
        #             layers.append(nn.Dropout(0.3))
                
        #         in_features = out_features
            
        #     layers.append(nn.Linear(out_features, 1))
        
        # self.classification_head = nn.Sequential(*layers)
        self.classification_head = nn.Sequential(nn.Linear(in_features, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        
    
    def forward(self, x):
        return self.classification_head(x)
    
    
    
class DownstremTask(pl.LightningModule):
    """ Performs fine-tune or linear evaluation of a pre-trained model
    on a downstream task
    """
    def __init__(self,
                 pretraining_method: Literal['Supervised', 'SimCLR', 'SupCon'],
                 model_path:str,
                 results_file:str,
                 evaluation: Literal['Fine-Tune', 'Linear'],
                 eval_dataset: Literal['BCI', 'ALS', 'IntraOverall', 'IntraMD'],
                 optimizer: Literal['Adam', 'AdamW', 'SGD'],
                 learning_rate:float = 0.001,
                 weight_decay:float = 4e-5,
                 betas:tuple = (0.9, 0.999),
                 momentum:float = 0.9,
                 cls_head_layers:int=2,
                 model=None):
        super(DownstremTask, self).__init__()
        
        self.save_hyperparameters()
        
        if pretraining_method == 'SimCLR':
            model = simclr.SimCLR.load_from_checkpoint(model_path)
            self.encoder = model.encoder
            in_features = model.projection_head[0].in_features
            
        elif pretraining_method == 'SupCon':
            model = supcon.SupCon.load_from_checkpoint(model_path)  
            self.encoder = model.encoder
            in_features = model.projection_head[0].in_features
        
        elif pretraining_method == 'Supervised':
            model = supervised.Supervised.load_from_checkpoint(model_path)
            
            try:
                in_features = model.model.dense.in_features
                model.model.dense = nn.Identity()
            except AttributeError:
                in_features = model.model[2].dense[0].in_features
                model.model[2].dense = nn.Identity()
                
            self.encoder = model
            
            
        # self.classification_head = ClassificationHead(in_features)
        self.classification_head = nn.Linear(in_features, 1)
        # self.classification_head = nn.Sequential(nn.Linear(in_features, 256),
        #                          nn.ReLU(),
        #                          nn.Linear(256, 1))
           
        # Freeze encoder parameters in Linear Evaluation
        if evaluation == 'Linear':
            for param in self.encoder.parameters():
                param.requires_grad = False
            
        # Adjust loss for class inbalance  
        pos_weight = torch.tensor([5 / 1], dtype=torch.float32)
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Metrics
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.auc = torchmetrics.AUROC(task='binary')
        self.confmat = torchmetrics.ConfusionMatrix(task='binary')
        
        self.logits = []
        self.test_annotations = []
        self.speller = []
        self.labels = []
        self.codes = []
        
    def forward(self, x):
        embedings, _ = self.encoder(x)
        out = self.classification_head(embedings)    
        return embedings, out    
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.hparams.learning_rate,  
                                   betas=self.hparams.betas,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(),
                                   lr=self.hparams.learning_rate,  
                                   betas=self.hparams.betas,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(),
                                   lr=self.hparams.learning_rate,  
                                   momentum=self.hparams.momentum,
                                   weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def common_step(self, batch, batch_idx):
        x, y = batch
        embeddings, scores = self.forward(x)
        loss = self.loss_func(scores, y)
        return loss, scores, y
    
    def training_step(self, batch, batch_idx):
        loss, scores, y = self.common_step(batch, batch_idx)
        
        self.log_dict(
            {'train_loss': loss}, 
            on_epoch=True,
            on_step=False,
            prog_bar=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self.common_step(batch, batch_idx)
        
        # probabilities = torch.sigmoid(scores)
        # predictions = (probabilities > 0.5).int()
        
        acc = self.accuracy(scores, y)
        f1 = self.f1_score(scores, y)
        recall = self.recall(scores, y)
        precision = self.precision(scores, y)
        auc = self.auc(scores, y)
        
        self.log_dict(
            {
                'val_loss': loss,
                'val_accuracy': acc,
                'val_f1_score': f1,
                'val_recall': recall,
                'val_precision': precision,
                'val_auc': auc,
            }, 
            on_epoch=True,
            on_step=False,
            prog_bar=True
        )
        return loss

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
        fig = utils_training.plot_results_across_trials(char_recog_acc, n_trials)
        self.logger.experiment.add_figure('Results across trials', fig)
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
        
        if  self.hparams.eval_dataset == 'IntraOverall' or self.hparams.eval_dataset == 'IntraMD':
            signals, test_annotations, speller = batch
            embeddings, logits = self.forward(signals)
            
            self.logits.append(logits)
            self.test_annotations.append(test_annotations)
            self.speller.append(torch.stack([speller[0][0], speller[0][1]], dim=1))  
            
    def on_test_epoch_end(self):
        
        if self.hparams.eval_dataset == 'IntraOverall' or self.hparams.eval_dataset == 'IntraMD':
            logits = torch.cat(self.logits, dim=0).squeeze()
            test_annotations = torch.cat(self.test_annotations, dim=0)
            speller = torch.cat(self.speller, dim=0)
            print('Starting Character Recognition Evaluation')
            results = utils_evaluation.character_recognition_Gib_UVA(logits, test_annotations, speller)
            with open(self.hparams.results_file, "a") as file:
                file.write(f'{results}')
                file.write('\n')
           
        else:
            logits = torch.cat(self.logits, dim=0).squeeze()
            labels = torch.cat(self.labels, dim=0).squeeze() if len(self.labels) > 0 else None
            codes = torch.cat(self.codes, dim=0)
            
            n_chars, n_trials = (35, range(1,11)) if self.hparams.eval_dataset == 'ALS' else  (100, range(1,16))
            target_string = self.trainer.datamodule.target_string
            
            print('Starting Character Recognition Evaluation')
            cra_results = self.character_recognition_per_trials(logits, codes, n_trials, 
                                                    n_chars, target_string)
            print('Starting Binary Evaluation')   
            binary_results = self.binary_classification(logits, labels) if self.hparams.eval_dataset == 'ALS' else None
            
            with open(self.hparams.results_file, "a") as file:
                file.write(f'{cra_results}')
                file.write('\n')
            if binary_results:
                binary_results_file = f'/workspace/project/GitHub/newScripts/CrossEvaluation/ALS Results/{self.hparams.pretraining_method}_results/{self.hparams.model}_1CLS_50Folfd_binary_evaluation_results.txt'
                with open(binary_results_file, "a") as file:
                    metrics_extracted = {key: value.item() for key, value in binary_results.items()}
                    file.write(f'\n{metrics_extracted}')