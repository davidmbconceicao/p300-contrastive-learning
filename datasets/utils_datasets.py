import numpy as np
import mne
from typing import Literal, Optional

import torch
from torchvision import transforms
from mne.decoding import Scaler

######### AUGMENTATIONS CLASSES #########

class RandomFilter:
    def __init__(self) -> None:
        info = {
            'sfreq': 128,
            'ch_names': ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz'], 
            'ch_types' : 8 * ['eeg'],
        }
        self.info = mne.create_info(ch_names=info['ch_names'], sfreq=info['sfreq'], ch_types=info['ch_types'])
        
    
    def __call__(self, sample):
        # print(f'Sample Shape before RandomFilter: {sample.shape}')
        sample = np.expand_dims(sample, axis=0) if len(sample.shape) < 3 else sample
        epoch = mne.EpochsArray(sample, self.info, verbose='CRITICAL')
        l_freq = 0.5
        h_freq = np.random.randint(10, 46)
        epoch.filter(l_freq, h_freq, verbose='CRITICAL')
        return epoch.get_data(copy=True)
    
class Normalize:
    def __init__(self) -> None:
        self.scaler = Scaler(scalings='mean')
    
    def __call__(self, sample):
        # print(f'Sample Shape before Normalize: {sample.shape}')
        sample = np.expand_dims(sample, axis=0) if len(sample.shape) < 3 else sample
        return self.scaler.fit_transform(sample)
    
class CutOut:
    """Applies a cutout of specified size (height=mask_size, width=mask_size*2.4)
    randomly to a section of the signal. a cutout is a zone where all data 
    points are set to 0
    """
    def __init__(self, mask_size=5):
        self.mask_size = mask_size
        
    def __call__(self, sample):
        """
        Args:
            sample (np.array): shape (n_epochs, n_chans, n_time_samples)

        Returns:
            X (np.array): shape (n_epochs, n_chans, n_times)
        """
        sample = sample.copy()
        # print(f'Sample Shape before CutOut: {sample.shape}')
        _, height, width = sample.shape

        for s in sample:
            # Generate random top-left corner for the mask
            top = np.random.randint(0, height - self.mask_size)
            left = np.random.randint(0, width - self.mask_size)
            
            # Apply the mask (set the cutout region to zero)
            s[top:top + self.mask_size, left:left + int(self.mask_size*3.4)] = 0
        
        #! Justify the choice of cutout dims
        return sample
    
class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        """
        Args:
            sample (np.array): shape (n_epochs, n_chans, n_time_samples)

        Returns:
            X (torch.Tensor): shape (n_epochs, 1, n_chans, n_time_samples)
        """
        # print(f'Sample Shape before Totensor: {sample.shape}')
        sample = torch.from_numpy(sample).type(torch.float32)
        return sample
    
class LabelTransform:
    def __init__(self) -> None:
        pass
    
    def __call__(self, label):
        label = np.expand_dims(label, axis=0).astype(np.float32)
        return torch.from_numpy(label)
 
 
    
class ContrastiveAugmentations:
    def __init__(self) -> None:
        self.contrast_transforms = transforms.Compose([
            RandomFilter(),
            Normalize(),
            # CutOut(),
            ToTensor()
        ])
    
    def __call__(self, signal):
        # print(f'Epoch shape: {epoch.shape}')
        # print(f'Evoked shape: {evoked.shape}')
        return self.contrast_transforms(signal)