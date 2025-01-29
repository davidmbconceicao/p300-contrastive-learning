import torch
from torch import nn, optim
import torchmetrics
import pytorch_lightning as pl

class Classification_Head(pl.LightningModule):
    def __init__(self, in_features):
        super(Classification_Head, self).__init__()
        self.save_hyperparameters()
        
        self.cls_head = nn.Sequential(nn.Linear(self.hparams.in_features, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        
        pos_weight = torch.tensor([5 / 1], dtype=torch.float32)
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Metrics
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.auc = torchmetrics.AUROC(task='binary')
        self.confmat = torchmetrics.ConfusionMatrix(task='binary')
        
    def forward(self, x):
        return self.cls_head(x)
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
        
    def common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        return loss, logits, y
        
    def training_step(self, batch, batch_idx):
        loss, logits, y = self.common_step(batch, batch_idx)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.common_step(batch, batch_idx)
        metrics = {'val_val_loss': loss}
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, logits, y = self.common_step(batch, batch_idx)
        
        acc = self.accuracy(logits, y)
        f1 = self.f1_score(logits, y)
        recall = self.recall(logits, y)
        precision = self.precision(logits, y)
        auc = self.auc(logits, y)
        confmat = self.confmat(logits, y)
                    
        metrics = {'val_accuracy':acc,
                'val_f1score':f1,
                'val_recall':recall,
                'val_precision': precision,
                'val_auc':auc}
        
        # confmat = {'logits': logits,
        #            'y': y}
            
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        # self.log_dict(confmat, on_epoch=True, on_step=False, prog_bar=True)
            
        # confusion_matrix = utils.plot_confusion_matrix(confmat, metrics)
        # self.logger.experiment.add_figure('Confusion Matrix', confusion_matrix, self.current_epoch)
        return metrics