import torch
import pytorch_lightning as pl

class ClearCacheCallback(pl.Callback):
    def __init__(self, gpu):
        self.gpu = gpu 
        
    def on_batch_end(self, trainer, pl_module):
        # Clear GPU cache at the end of each batch
        torch.cuda.set_device(self.gpu)
        torch.cuda.empty_cache()


def set_deterministic(seed=42):
    # Set seed for Python's built-in random module
    # random.seed(seed)
    # Set seed for numpy
    # np.random.seed(seed)
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # Ensure deterministic behavior in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f'Set determinist with seed {seed}')