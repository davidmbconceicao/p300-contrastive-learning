import os
import sys
import mne
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import utils_datasets

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
als_data_folder = os.path.join(parent_dir, 'Data', 'ALS')

class ALSDataset(Dataset):
    def __init__(self,
                 test_subject:int,
                 train:bool=True) -> None:
        super().__init__()
        
        self.test_subject = test_subject
        self.train = train
        
        self.signal_transform = transforms.Compose([
                utils_datasets.Normalize(),
                utils_datasets.ToTensor()
        ])
        self.label_transform = utils_datasets.LabelTransform()
        
        als_data_file = os.path.join(als_data_folder, 'ALS_signal_epochs_epo.fif')
        als_annotations_file = os.path.join(als_data_folder, 'ALS_annotations.csv')
        als_epochs = mne.read_epochs(als_data_file, verbose='CRITICAL').get_data(copy=True)
        als_annotations = pd.read_csv(als_annotations_file)
        
        test_indices = als_annotations.index[als_annotations['subject']==test_subject].to_list()
        self.test_data = als_epochs[test_indices]
        self.test_annotations = als_annotations.iloc[test_indices]
        
        train_indices = als_annotations.index[als_annotations['subject'] != test_subject].to_list()
        self.train_data = als_epochs[train_indices]  # Index the MNE epochs object
        self.train_annotations = als_annotations.iloc[train_indices] 
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.test_data)
    
    def __getitem__(self, index):
        
        if self.train:
            epoch = np.expand_dims(self.train_data[index], axis=0)
            epoch = self.signal_transform(epoch)
            label = self.train_annotations.iloc[index]['label']
            label= self.label_transform(label)  
            return epoch, label
            
        else:
            epoch = np.expand_dims(self.test_data[index], axis=0)
            epoch = self.signal_transform(epoch)
            label = self.test_annotations.iloc[index]['label']
            label= self.label_transform(label)  
            codes = self.test_annotations.iloc[index]['row_col']
            return epoch, label, codes
    
class AlSDataModule(pl.LightningDataModule):
    def __init__(self,
                 test_subject:int,
                 batch_size:int,
                 num_workers:int,):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.test_subject in [x for x in range(8)], 'ValueError: test_subject should be a value between 0 and 7 (inclusive)'
        
    def setup(self, stage):
        # pl.seed_everything(42, workers=True)
        
        if stage == "fit" or stage is None:
            train_dataset = ALSDataset(test_subject=self.hparams.test_subject,
                                    train=True)
            
            train_size = int(len(train_dataset) * 0.80)
            val_size = int(len(train_dataset) * 0.20)

            self.train_set, self.val_set = random_split(
                train_dataset, [train_size, val_size]
            )
            
        if stage == 'test':
            self.test_set = ALSDataset(test_subject=self.hparams.test_subject,
                                    train=False)
            self.target_string = 'WGAMNF7ODYKPUUABAJJOCYIJYVNO6A7YGPD'
            
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        ) 
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )    
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=len(self.test_set),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )