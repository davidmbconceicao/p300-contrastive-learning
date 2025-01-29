import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import Literal

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from evaluation import downstream_task
from datasets import gib_uva_dataset



def validate_model(
    pretraining_method:Literal['Supervised', 'SimCLR', 'SupCon'],
    dataset:Literal['Cross', 'IntraOverall', 'IntraMD'],
    gpu:int,
    results_file:str, 
    checkpoint_dir:str,
    pretrained_model_path:str = None,
):
    
    if dataset == 'Cross' or dataset == 'IntraOverall':
        phase = 'valCross' if dataset=='Cross' else 'valOverall'
        val_subjects = [ 0, 4, 18, 63]
    elif dataset == 'IntraMD':
        phase = 'valMD'
        val_subjects = [4, 8, 13, 19, 25, 26, 29, 55]
        
    results_file = os.path.join(os.getcwd(), results_file)
    
    with open(results_file, "a") as file:
            file.write(f'Validation of  -> ')
            file.write(pretrained_model_path)
            file.write('\n')
            
    for subject in val_subjects:
        utils.set_deterministic()
        
        with open(results_file, "a") as file:
                file.write(f'Validation of Subject {subject}')
                file.write('\n')
                
        model_name = f'{pretraining_method}_test_subject_{subject}'
          
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=checkpoint_dir,
            filename=model_name,
            save_top_k=1,
            mode='min'
        )
                    
        # Define Model

        pretrained_model = downstream_task.DownstremTask(
                pretraining_method=pretraining_method,
                model_path=pretrained_model_path,
                results_file=results_file,
                evaluation='Linear',
                eval_dataset=dataset,
                optimizer='Adam',
                learning_rate=1e-3,
                weight_decay=6e-5,
                cls_head_layers=2
        )
        
        dm_eval =  gib_uva_dataset.SubjectIndependentDataloader(
            phase=phase,
            subject=subject,
            batch_size=256,
            num_workers=20
        )   
                 
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[gpu],
            callbacks=checkpoint_callback,
            log_every_n_steps=5,
            max_epochs=100,
        )
            
        trainer.fit(pretrained_model, dm_eval)
        trainer.validate(ckpt_path='best', datamodule=dm_eval)
        trainer.test(ckpt_path='best', datamodule=dm_eval)
        