import os
import sys
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from training import supervised
from datasets import bci_comp_dataset, als_dataset
import utils

current_dir = os.path.dirname(os.path.abspath(__file__))

model_name = 'Conformer'
subjects = range(8)
gpu = 2
eval_results_file = os.path.join(current_dir, f'{model_name}_results_ALS', f'{model_name}_50Folfd_evaluation_results_ALS.txt')
eval_checkpoint_dir = os.path.join(current_dir, f'{model_name}_results_ALS', f'{model_name}_50Folfd_evaluation_checkpoints_ALS')

with open(eval_results_file, "a") as file:
        file.write(f'\nEvaluation of  -> ')
        file.write(f'{model_name}')
        

for subject_in_q in subjects:
    
    with open(eval_results_file, "a") as file:
        # file.write(f'\nSubject {subject_in_q} with 85 chars retraining')
        file.write(f'\nSubject {subject_in_q}')
        file.write('\n')
                    
    model_name_second = f'{model_name}-{subject_in_q}'

    for _ in range(50):
                
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=eval_checkpoint_dir,
            filename=model_name,
            save_top_k=1,
            mode='min'
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min'
        )
                
        model = supervised.Supervised(
            results_file=eval_results_file,
            model=model_name,
            eval_dataset='ALS'
        )
                
        # dm = bci_comp_dataset.BCICompDataModule(
        #     subject=subject_in_q,
        #     batch_size=256,
        #     num_workers=10
        # )
        dm = als_dataset.AlSDataModule(
            test_subject=subject_in_q,
            batch_size=256,
            num_workers=20
        )
                
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[gpu],
            callbacks=[checkpoint_callback, early_stop],
            log_every_n_steps=5,
            max_epochs=100
        )
                
        trainer.fit(model, dm)
        trainer.test(ckpt_path='best', datamodule=dm)

        



