import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import Literal

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from evaluation import downstream_task
from datasets import gib_uva_dataset, als_dataset

def subject_independent_evaluation(
    pretraining_method:Literal['Supervised', 'SimCLR', 'SupCon'],
    eval_dataset:Literal['ALS', 'IntraOverall'],
    gpu:int,
    results_file:str,
    checkpoint_dir:str,
    pretrained_model_path:str,
    name_model=None
    ):
    
    if eval_dataset == 'ALS':
        test_subjects = [x for x in range(8)]
        # test_subjects=[0]
    elif eval_dataset == 'IntraOverall':
        test_subjects = [ 2,  8,  9, 10, 16, 17, 20, 
                         33, 34, 45, 59, 60, 64, 66]
     
    with open(results_file, "a") as file:
        file.write(f'Evaluation of  -> ')
        file.write(pretrained_model_path)
        file.write('\n')   
            

    for subject in test_subjects:
        
        model_name = f'{pretraining_method}(test_subject_{subject})'
        with open(results_file, "a") as file:
                file.write(f'Subject {subject}')
                file.write('\n')
                
        for _ in range(50): 
                
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
                cls_head_layers=1,
                model = name_model
            )
                    
            if eval_dataset == 'ALS':
                dm_eval = als_dataset.AlSDataModule(
                    test_subject=subject,
                    batch_size=256,
                    num_workers=20
                )
                    
            elif eval_dataset == 'IntraOverall':
                dm_eval = gib_uva_dataset.SubjectIndependentDataloader(
                    phase='testOverall',
                    subject=subject,
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
                