import os
import sys
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import Literal

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from evaluation import downstream_task
from datasets import gib_uva_dataset, bci_comp_dataset

def subject_dependent_evaluation(
    pretraining_method:Literal['Supervised', 'SimCLR', 'SupCon'],
    eval_dataset:Literal['BCI', 'IntraMD', 'IntraOverall'],
    gpu:int,
    results_file:str,
    checkpoint_dir:str,
    pretrained_model_path:str
    ):
    
    if eval_dataset == 'BCI':
        test_subjects = ['A', 'B']
        fine_tuning_values = [1, 5, 10, 25, 40, 85]
        k_fold = [50, 50, 50, 50, 50, 50]
        
    elif eval_dataset == 'IntraOverall':
        test_subjects = [ 2,  8,  9, 10, 16, 17, 20, 
                         33, 34, 45, 59, 60, 64, 66]
        fine_tuning_values = [1, 5, 10]
        k_fold = [50, 50, 50]


     
    with open(results_file, "a") as file:
            file.write(f'Evaluation of  -> ')
            file.write(pretrained_model_path)
            file.write('\n')   
            
    for k, ft in zip(k_fold, fine_tuning_values):
        for subject in test_subjects:
            
            print()
            print(pretrained_model_path)
            print()
            model_name = f'{pretraining_method}(test_subject_{subject}_with_{ft}_ft_chars)'
            
            with open(results_file, "a") as file:
                file.write(f'Subject {subject} with {ft} chars retraining')
                file.write('\n')
            
            for _ in range(k):
                # utils.set_deterministic(seed=np.random.randint(0, 1000))
                
                checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=checkpoint_dir,
                    filename=model_name,
                    save_top_k=1,
                    mode='min'
                )
                
                pretrained_model = downstream_task.DownstremTask(
                    pretraining_method=pretraining_method,
                    model_path=pretrained_model_path,
                    results_file=results_file,
                    evaluation='Linear',
                    eval_dataset=eval_dataset,
                    optimizer='Adam',
                    learning_rate=1e-3,
                    weight_decay=6e-5,
                    cls_head_layers=1
                )
                
                if eval_dataset == 'BCI':
                    dm_eval = bci_comp_dataset.BCICompDataModule(
                        subject=subject,
                        n_chars=ft,
                        batch_size=256,
                        num_workers=15
                    )
                
                elif eval_dataset == 'IntraMD' or eval_dataset == 'IntraOverall':
                    dm_eval = gib_uva_dataset.SubjectDependentDataloader(
                        eval_dataset = eval_dataset,
                        subject=subject,
                        fine_tuning=ft,
                        batch_size=256,
                        num_workers = 10
                    )
                
                trainer = pl.Trainer(
                    accelerator='gpu',
                    devices=[gpu],
                    callbacks=checkpoint_callback,
                    log_every_n_steps=5,
                    max_epochs=100
                )
                
                trainer.fit(pretrained_model, dm_eval)
                trainer.validate(ckpt_path='best', datamodule=dm_eval)
                trainer.test(ckpt_path='best', datamodule=dm_eval)
                
                
                
            