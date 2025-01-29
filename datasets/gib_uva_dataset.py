import os
import sys
import mne
import ast
import random
import pandas as pd
import numpy as np
from tqdm import tqdm 
from typing import Literal
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import utils_datasets

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
gib_uva_data_folder = os.path.join(parent_dir, 'Data', 'Gib_UVA')
gib_uva_cross_data_folder = os.path.join(parent_dir, 'Data', 'Gib_UVA', 'Cross')
gib_uva_overall_data_folder = os.path.join(parent_dir, 'Data', 'Gib_UVA', 'Overall')
gib_uva_MD_data_folder = os.path.join(parent_dir, 'Data', 'Gib_UVA', 'MD')


class PreTrainingDataset(Dataset):     
    """ Dataset partitioned into train, val and test sets
    Augmentation applied is Progressive Evokeds
    """ 
    def __init__(self, 
                 dataset:Literal['Cross', 'IntraOverall', 'IntraMD'],
                 training_method: Literal['Supervised', 'SimCLR', 'SupCon'],
                 partition: Literal['train', 'val']):
             
        self.training_method = training_method
        self.partition = partition     
        self.dataset = dataset
         
        self.contrastive_transform = utils_datasets.ContrastiveAugmentations() 
        self.label_transform = utils_datasets.LabelTransform()
        self.signal_transform = transforms.Compose([
            utils_datasets.Normalize(),
            utils_datasets.ToTensor()
        ])

        if self.dataset == 'Cross':
            folder = gib_uva_cross_data_folder
        if self.dataset == 'IntraOverall':
            folder = gib_uva_overall_data_folder
        if self.dataset == 'IntraMD':
            folder = gib_uva_MD_data_folder
   
        data_file = f'{partition}_epochs_epo.fif'
        annotations_file = f'{partition}_annotations.csv'
        
        self.signal_epochs = mne.read_epochs(os.path.join(folder, data_file), verbose='CRITICAL', preload=True).get_data(copy=True)
        self.epochs_annotations = pd.read_csv(os.path.join(folder, annotations_file))
        
        if self.partition == 'train':
            # if os.path.exists(os.path.join(folder, 'train_evokeds_epo.fif')):
            self.evoked_signals = mne.read_epochs(os.path.join(folder, 'train_evokeds_epo.fif'), verbose='CRITICAL', preload=True).get_data(copy=True)
            self.evoked_annotations = pd.read_csv(os.path.join(folder, 'train_evoked_annotations.csv'))
            # else:
            #     self.evoked_signals = create_evokeds(folder)
            #     self.evoked_annotations = pd.read_csv(os.path.join(folder, 'train_evoked_annotations.csv'))
                
            
    def __len__(self):
        return len(self.signal_epochs)
    
    def __getitem__(self, index):
        
        if self.partition == 'train' and self.training_method in ['SimCLR', 'SupCon']:
            # get signal epoch
            signal_epoch = self.signal_epochs[index]
            label = self.epochs_annotations.iloc[index]['labels']
            char = self.epochs_annotations.iloc[index]['char'], 
            code = self.epochs_annotations.iloc[index]['codes']
            
            
            evoked_indices = self.evoked_annotations[(self.evoked_annotations['char']==char[0]) & (self.evoked_annotations['codes']==code)]['evoked_indices'].values[0]
            # print(evoked_inidices)
            evoked_indices_list = ast.literal_eval(evoked_indices)
            evokeds = self.evoked_signals[evoked_indices_list]
            
            # Randomly select 3 distinct evokeds (without replacement)
            selected_evokeds_indices = np.random.choice(evokeds.shape[0], size=3, replace=False)
            # print(f'Epoch Index: {index}, Selected Evokeds: {selected_evokeds_indices}')
            selected_evokeds = evokeds[selected_evokeds_indices]
            
            views = [signal_epoch] + [x for x in selected_evokeds]
                
            if self.training_method == 'SimCLR':
                return [self.signal_transform(x) for x in views]
            #     return [self.contrastive_transform(x) for x in views]
            # return [self.contrastive_transform(x) for x in views], self.label_transform(label)  
            return [self.signal_transform(x) for x in views], self.label_transform(label)  
           
        # in any other case we just want to return the signal and label 
        signal_epoch = self.signal_epochs[index]
        label = self.epochs_annotations.iloc[index]['labels']
        # codes = self.epochs_annotations.iloc[index]['codes']
        return self.signal_transform(signal_epoch), self.label_transform(label)

#! Deixar cutout e randomfilter e dizer que testes preliminares não deram bons resultados.

class PreTrainingDataModule(pl.LightningDataModule):
    def __init__(self, 
                 training_method:Literal['Supervised', 'SimCLR', 'SupCon'],
                 dataset:Literal['Cross', 'IntraOverall', 'IntraMD'],
                 batch_size:int=300, 
                 num_workers:int=4, 
                 balanced_dataset:bool=False):
        super().__init__()
        self.save_hyperparameters()
        self.training_method = training_method
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.balanced_dataset = balanced_dataset
        self.class_weights = None

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.train_dataset = PreTrainingDataset(dataset=self.dataset, training_method=self.training_method, partition='train')
            self.val_dataset = PreTrainingDataset(dataset=self.dataset, training_method=self.training_method, partition='val')
            
            if self.balanced_dataset:
                if self.dataset == 'Cross':
                    folder = gib_uva_cross_data_folder
                if self.dataset == 'IntraOverall':
                    folder = gib_uva_overall_data_folder
                if self.dataset == 'IntraMD':
                    folder = gib_uva_MD_data_folder

                if os.path.exists(os.path.join(folder, 'class_weights.npy')):
                    self.class_weights = np.load(os.path.join(folder, 'class_weights.npy'), allow_pickle=True)
                    print("Loaded class weights from file.")
                else:
                    self.class_weights = self.compute_class_weights(self.train_dataset)
                    np.save(os.path.join(folder, 'class_weights.npy'), [self.class_weights])  # Save to file
                    print("Computed and saved class weights to file.")
        
    def compute_class_weights(self, dataset):
        # Compute the class weights based on the labels in the dataset
        print('Computing Class Weights...')
        labels = [l.item() if isinstance(l, torch.Tensor) else l for _, l in tqdm(dataset, desc="Processing labels", leave=False)]
        total_count = len(labels)
        pos_count = sum(labels)
        neg_count = total_count - pos_count
        
        # Here we set fixed weights for positive and negative classes
        pos_weight = 1.0 / pos_count if pos_count > 0 else 0
        neg_weight = 1.0 / neg_count if neg_count > 0 else 0
        
        return [pos_weight if label == 1 else neg_weight for label in labels]
    
    def on_train_start(self):
        # Log DataModule parameters to TensorBoard
        datamodule_params = self.trainer.datamodule.hparams
        for param_name, param_value in datamodule_params.items():
            self.log(f"datamodule/{param_name}", param_value)
            
    def train_dataloader(self):
        if self.balanced_dataset:
            
            train_sampler = WeightedRandomSampler(weights=self.class_weights.squeeze(),
                                                num_samples=len(self.train_dataset), 
                                                replacement=True)

            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                                sampler=train_sampler, persistent_workers=True, pin_memory=True)
            
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                                shuffle=True, persistent_workers=True, pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          persistent_workers=True, pin_memory=True)
        
        
################################ EVALUATION DATASETS ################################


class SubjectDependentEval(Dataset):
    """ Subject Dependent evaluation is destined to the IntraMDDataset.
    This class separates each subject data into train and test sets, where the 
    train set correspond to the data of n characters, where n = fine_tuning and the 
    test set correspond to the reamining characters. If fine_tuning = 1 or 5 we chose
    1 or 5 characters among the pool of characters with 15 trials. If fine_tuning is 
    greater, then we chose n characters at random from all characters with the
    precaution of having at least 10 characters with 15 trials in the test set
    """
    def __init__(self,
        eval_dataset: Literal['IntraMD', 'IntraOverall'],
        subject:int,
        fine_tuning:Literal[1, 5, 10, 20, 30],
        partition:Literal['train', 'test']
        ) -> None:
        
        self.eval_dataset = eval_dataset
        self.subject = subject
        self.fine_tuning = fine_tuning 
        self.partition = partition
        
        self.label_transform = utils_datasets.LabelTransform()
        self.signal_transform = transforms.Compose([
            utils_datasets.Normalize(),
            utils_datasets.ToTensor()
        ])
        
        folder = gib_uva_MD_data_folder if self.eval_dataset == 'IntraMD' else gib_uva_overall_data_folder
        
        test_annotations = pd.read_csv(os.path.join(folder, f'test_annotations.csv'))
        test_epochs = mne.read_epochs(os.path.join(folder, f'test_epochs_epo.fif')).get_data(copy=True)
        test_sequences = pd.read_csv(os.path.join(gib_uva_data_folder, 'test_subjects_sequences.csv'))
        
        # print(test_epochs.shape)
        
        # get subject specific data
        subject_annotations = test_annotations[test_annotations['subject']==self.subject]
        subject_chars = np.unique(subject_annotations['char'])
        # print(subject_chars)
        self.sequences = test_sequences[test_sequences['subject']==self.subject]['trials_by_sequence']
        # print(self.sequences)
        speller_matrices = test_sequences[test_sequences['subject']==self.subject]['speller_matrix']
        self.speller_matrices = [ast.literal_eval(item) for item in speller_matrices]
        
       # define which chars go for training and get the speller matrices of each char
        if self.fine_tuning in [1,5]:
            # if fine tuning chars = 1 or 5 we choose these chars from the pool of chars that have 15 trials
            train_chars = [trials for sequence in self.sequences for key, trials in ast.literal_eval(sequence).items() if key == 14][0]
            # print(train_chars)
        else:
            # if fine tuning chars is more than 5, garantee that at least 10 chars with 15 trials go to evaluation
            chars_with_15_trials = [trials for sequence in self.sequences for key, trials in ast.literal_eval(sequence).items() if key==14]
            excluded_chars = random.sample(chars_with_15_trials[0], 10)
            train_chars = [char for char in subject_chars if char not in excluded_chars]
            
        # train_chars = [trials for sequence in sequences for key, trials in ast.literal_eval(sequence).items() if key != 14]
        # train_chars = [item for sublist in train_chars for item in sublist] if train_chars else []
        speller_matrices = test_sequences[test_sequences['subject']==self.subject]['speller_matrix']
        self.speller_matrices = [ast.literal_eval(item) for item in speller_matrices]
            
        # randomly choose the fine-tuning (retraining) chars
        ft_chars = np.random.choice(train_chars, self.fine_tuning, replace=False)
            
        self.ft_labels = test_annotations[(test_annotations['subject']==self.subject) & (test_annotations['char'].isin(ft_chars))]['labels'].values
        ft_indices = test_annotations[(test_annotations['subject']==self.subject) & (test_annotations['char'].isin(ft_chars))].index.to_list()
        self.ft_epochs = test_epochs[ft_indices]
        print(f'Retraining Characters: {ft_chars}')
        # print(self.ft_epochs.shape)
            
        # define the test chars
        eval_chars = list(set(subject_chars) - set(ft_chars))
        print(eval_chars)
        self.eval_annotations = test_annotations[(test_annotations['subject']==self.subject) & (test_annotations['char'].isin(eval_chars))]
        eval_indices = test_annotations[(test_annotations['subject']==self.subject) & (test_annotations['char'].isin(eval_chars))].index.to_list()
        # print(eval_indices)
        self.eval_codes = self.eval_annotations['codes'].values
        self.eval_epochs = test_epochs[eval_indices]
        
        print(self.eval_epochs.shape)
        
    def __len__(self):
        if self.partition == 'train':
            return len(self.ft_epochs)
        return len(self.eval_epochs)
    
    def __getitem__(self, index):
        if self.partition == 'train':
            epoch = self.ft_epochs[index]
            label = self.ft_labels[index]
            return self.signal_transform(epoch), self.label_transform(label)
        if self.partition == 'test':
            epoch = self.eval_epochs[index]
            eval_annotation = self.eval_annotations.iloc[index]
            eval_char = eval_annotation['char']
            # seqs = [s for sequence in self.sequences for s, x in sequence.items() if eval_char in x]
            eval_annotation = torch.tensor(self.eval_annotations.iloc[index].values, dtype=torch.float32)
            speller = [k for speller_dict in self.speller_matrices for k, x in speller_dict.items() if eval_char in x]
            return self.signal_transform(epoch), eval_annotation, speller
  
 
 
class SubjectDependentDataloader(pl.LightningDataModule):
    def __init__(self, 
        eval_dataset:Literal['IntraMD', 'IntraOverall'],
        subject:int,
        fine_tuning:Literal[1, 5, 10, 20, 30],
        batch_size:int,
        num_workers:int
        ):
        super().__init__()
        
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            train_dataset = SubjectDependentEval(
                eval_dataset=self.hparams.eval_dataset,
                subject = self.hparams.subject,
                fine_tuning= self.hparams.fine_tuning,
                partition='train'
            )
            
            train_size = int(len(train_dataset)*0.85)
            val_size = len(train_dataset) - train_size
            self.train_set, self.val_set = random_split(train_dataset, [train_size, val_size])
    
        if stage == 'test':
            self.test_dataset = SubjectDependentEval(
                eval_dataset=self.hparams.eval_dataset,
                subject = self.hparams.subject,
                fine_tuning= self.hparams.fine_tuning,
                partition='test'
            )
            
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          shuffle=True,
                          persistent_workers=True,
                          pin_memory=True)
    
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          shuffle=False,
                          persistent_workers=True,
                          pin_memory=True)
    
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False,
                          persistent_workers=True,
                          pin_memory=True)
    
 
 
 
       
class SubjectIndependentEval(Dataset):
    """ Subject independent evaluation is used in the IntraOverall dataset
    and to do validation at the end of each training session. 
    Leave-one-out cross validation is performed, i.e. each subject data is reserved 
    for evaluation while the remaining data is used for retraining the networks"""
    def __init__(self,
        phase:Literal['valCross', 'valOverall', 'valMD', 'testOverall'],
        subject:int,
        partition:Literal['train', 'test']
        ) -> None:
        
        self.phase = phase
        self.subject = subject
        self.partition = partition
        
        self.label_transform = utils_datasets.LabelTransform()
        self.signal_transform = transforms.Compose([
            utils_datasets.Normalize(),
            utils_datasets.ToTensor()
        ])
        
        if phase == 'valCross':
            folder = gib_uva_cross_data_folder
            annotations_file = os.path.join(folder, f'val_annotations.csv')
            epochs_file = os.path.join(folder, f'val_epochs_epo.fif')
        elif phase == 'valOverall':
            folder = gib_uva_overall_data_folder
            annotations_file = os.path.join(folder, f'val_annotations.csv')
            epochs_file = os.path.join(folder, f'val_epochs_epo.fif')
        elif phase == 'valMD':
            folder = gib_uva_MD_data_folder
            annotations_file = os.path.join(folder, f'val_annotations.csv')
            epochs_file = os.path.join(folder, f'val_epochs_epo.fif')
        elif phase == 'testOverall':
            folder = gib_uva_overall_data_folder
            annotations_file = os.path.join(folder, f'test_annotations.csv')
            epochs_file = os.path.join(folder, f'test_epochs_epo.fif')
        
        # load test files
        test_annotations = pd.read_csv(annotations_file)
        test_epochs = mne.read_epochs(epochs_file)
        test_sequences = pd.read_csv(os.path.join(gib_uva_data_folder, 'test_subjects_sequences.csv'))
        
        # get subject specific data
        subject_annotations = test_annotations[test_annotations['subject']==self.subject]
        subject_chars = np.unique(subject_annotations['char'])
        self.sequences = test_sequences[test_sequences['subject']==self.subject]['trials_by_sequence']
        # print(self.sequences)
        speller_matrices = test_sequences[test_sequences['subject']==self.subject]['speller_matrix']
        self.speller_matrices = [ast.literal_eval(item) for item in speller_matrices]
        
        # Define the retraining data
        ft_annotations = test_annotations[test_annotations['subject']!=self.subject]
        ft_indices = test_annotations[test_annotations['subject']!=self.subject].index.to_list()
        self.ft_labels = ft_annotations['labels'].values
        self.ft_epochs = test_epochs[ft_indices].get_data(copy=True)
            
        # Define the evaluation data (all charcters that have 14 sequences)
        eval_chars = subject_chars
            
        self.eval_annotations = test_annotations[(test_annotations['subject']==self.subject) & (test_annotations['char'].isin(eval_chars))]
        eval_indices = test_annotations[(test_annotations['subject']==self.subject) & (test_annotations['char'].isin(eval_chars))].index.to_list()
        self.eval_epochs = test_epochs[eval_indices].get_data(copy=True)
        
    def __len__(self):
        if self.partition == 'train':
            return len(self.ft_epochs)
        return len(self.eval_epochs)
    
    def __getitem__(self, index):
        if self.partition == 'train':
            epoch = self.ft_epochs[index]
            label = self.ft_labels[index]
            return self.signal_transform(epoch), self.label_transform(label)
        if self.partition == 'test':
            epoch = self.eval_epochs[index]
            eval_annotation = self.eval_annotations.iloc[index]
            eval_char = eval_annotation['char']
            # seqs = [s for sequence in self.sequences for s, x in sequence.items() if eval_char in x]
            eval_annotation = torch.tensor(self.eval_annotations.iloc[index].values, dtype=torch.float32)
            speller = [k for speller_dict in self.speller_matrices for k, x in speller_dict.items() if eval_char in x]
            return self.signal_transform(epoch), eval_annotation, speller
        
        
class SubjectIndependentDataloader(pl.LightningDataModule):
    def __init__(self, 
        phase:Literal['valCross', 'valOverall', 'valMD', 'testOverall'],
        subject:int,
        batch_size:int,
        num_workers:int
        ):
        super().__init__()
        
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            train_dataset = SubjectIndependentEval(
                phase=self.hparams.phase,
                subject = self.hparams.subject,
                partition='train'
            )
            
            train_size = int(len(train_dataset)*0.85)
            val_size = len(train_dataset) - train_size
            self.train_set, self.val_set = random_split(train_dataset, [train_size, val_size])
    
        if stage == 'test':
            self.test_dataset = SubjectIndependentEval(
                phase=self.hparams.phase,
                subject = self.hparams.subject,
                partition='test'
            )
            
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          shuffle=True,
                          persistent_workers=True,
                          pin_memory=True)
    
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          shuffle=False,
                          persistent_workers=True,
                          pin_memory=True)
    
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=len(self.test_dataset),
                          num_workers=self.hparams.num_workers,
                          shuffle=False,
                          persistent_workers=True,
                          pin_memory=True)
        

if __name__ == '__main__':
    pass