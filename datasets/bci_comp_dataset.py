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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils_datasets
from evaluation import utils_evaluation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
bci_data_folder = os.path.join(parent_dir, 'Data', 'BCI Comp')

class BCICompDataset(Dataset):
    def __init__(self,
                 subject:str,
                 n_chans:int=8,
                 train:bool=True,
                 n_chars:float=85) -> None:
        super().__init__()
        
        self.signal_transform = transforms.Compose([
                utils_datasets.Normalize(),
                utils_datasets.ToTensor()
        ])
        
        self.label_transform = utils_datasets.LabelTransform()
        
        self.subject = subject
        self.train = train
        self.n_chans = n_chans
        self.n_chars = n_chars
        
        if n_chans == 8:
            chans = ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz']
            
        elif n_chans == 16:
            chans = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 
                    'PO8', 'F3', 'F4', 'FCz', 'C3', 'C4', 'CP3', 
                    'CPz', 'CP4']     
        else:
            chans = ['eeg']
            
        if train:
            bci_data_path = os.path.join(bci_data_folder, f'Subject_{subject}_Train_epo.fif')
            bci_annotations_path = os.path.join(bci_data_folder, f'Subject_{subject}_Train_annotations.csv')
            
            # define train annotations
            bci_annotations = pd.read_csv(bci_annotations_path) 
            train_chars = np.random.choice([i for i in range(85)], self.n_chars, replace=False)
            train_indices = bci_annotations[bci_annotations['character'].isin(train_chars)].index.values
            self.bci_annotations = bci_annotations[bci_annotations['character'].isin(train_chars)]
            
            # get train data
            bci_data = mne.read_epochs(bci_data_path, verbose='CRITICAL')
            self.bci_data = bci_data[train_indices].get_data(picks=chans)
          
        else:
            bci_data_path = os.path.join(bci_data_folder, f'Subject_{subject}_Test_epo.fif')
            bci_annotations_path = os.path.join(bci_data_folder, f'Subject_{subject}_Test_annotations.csv') 
            
            self.bci_data = mne.read_epochs(bci_data_path, verbose='CRITICAL').get_data(picks=chans)
            self.bci_annotations = pd.read_csv(bci_annotations_path) 

        
    def __len__(self):
        return len(self.bci_data)
    
    def __getitem__(self, index):
        epoch = np.expand_dims(self.bci_data[index], axis=0)
        epoch = self.signal_transform(epoch)
        codes = self.bci_annotations.iloc[index]['row_col'] 
        
        if self.train:
            label = self.bci_annotations.iloc[index]['label']
            label= self.label_transform(label) 
            return epoch, label
        
        return epoch, codes
    
class BCICompDataModule(pl.LightningDataModule):
    def __init__(self,
                 subject:str,
                 batch_size:int,
                 num_workers:int,
                 n_chans:int=8,
                 n_chars=85) -> None:
        super().__init__()
        
        self.save_hyperparameters()
        
    def setup(self, stage):
        
        if stage == "fit" or stage is None:
            train_dataset = BCICompDataset(
                subject=self.hparams.subject,
                n_chans=self.hparams.n_chans,
                train=True,
                n_chars=self.hparams.n_chars
            )

            train_size = int(len(train_dataset) * 0.9)
            val_size = int(len(train_dataset) - train_size)

            self.train_set, self.val_set = random_split(
                train_dataset, [train_size, val_size]
            )
            print(len(self.train_set))
            print(len(self.val_set))

        if stage == 'test':
            self.test_set = BCICompDataset(
                subject=self.hparams.subject,
                n_chans=self.hparams.n_chans,
                train=False,
            )
                
            self.target_string = utils_evaluation.get_target_string_BCIComp(self.hparams.subject)

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
            batch_size=1024,
            # batch_size=len(self.test_set),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )