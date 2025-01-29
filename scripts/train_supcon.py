import os
import sys
import torch
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import supcon
from datasets import gib_uva_dataset
import utils

def train_supcon(
    model:Literal['EEGNet', 'EEGInception', 'Conformer'],
    pretraining_dataset: Literal['Cross', 'IntraOverall', 'IntraMD'],
    checkpoint_dir:str,
    resume_from_checkpoint:bool,
    gpu:int,
    temperature:float,
    lr:float,
    weight_decay:float = 6e-5,
    batch_size:int=512,
    epochs:int=200,
    num_workers:int=20
    ):
    
    utils.set_deterministic()

    MODEL_NAME = f'{model}_SupCon_temp_{temperature}' 
    CHECKPOINT_DIR = os.path.join(checkpoint_dir, MODEL_NAME)

    logger = TensorBoardLogger(os.path.join(checkpoint_dir, 'tb_logs_SupCon'), name=MODEL_NAME)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1score',
        dirpath=CHECKPOINT_DIR,
        filename=MODEL_NAME,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    # MODEL
    model = supcon.SupCon(
        model=model,
        ckpt_dir=CHECKPOINT_DIR,
        gpu=gpu,
        lr=lr,
        temperature=temperature,
        weight_decay=weight_decay,
        max_epochs=epochs,
        val='KNN'
    )

    # DATAMODULE
    dm = gib_uva_dataset.PreTrainingDataModule(
        training_method='SupCon',
        dataset=pretraining_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        balanced_dataset=True
    )

    # TRAINER
    trainer = pl.Trainer(
            logger=logger,
            accelerator='gpu',
            devices=[gpu],
            max_epochs=epochs,
            callbacks=[checkpoint_callback],
            log_every_n_steps=5,
            check_val_every_n_epoch=10
        )

    if resume_from_checkpoint:
        saved_model_path = os.path.join(CHECKPOINT_DIR, 'last.ckpt')
        trainer.fit(model, dm, ckpt_path=saved_model_path)
    else:
        trainer.fit(model, dm)
    
    
if __name__ == '__main__':
    pass

    # intra_evaluation.test_Gib_UVA(
    #     model='SupCon',
    #     dataset=PRETRAINING_DATASET,
    #     phase='val',
    #     gpu=GPU,
    #     results_file=f'results_{PRETRAINING_DATASET}_{MODEL_NAME}.txt',
    #     checkpoint_dir=os.path.join(current_dir, f'checkpoints_evaluation_{PRETRAINING_DATASET}', MODEL_NAME),
    #     pre_trained_model_path=os.path.join(CHECKPOINT_DIR, 'last.ckpt'),
    #     evaluation='Linear',
    #     optimizer='Adam',
    #     learning_rate=1e-3,
    #     weight_decay=6e-5,
    #     betas=(0.9, 0.999),
    #     batch_size=256,
    #     epochs=100,
    #     patience=50
    # )
        
    # intra_evaluation.test_Gib_UVA(
    #     model='SupCon',
    #     dataset=PRETRAINING_DATASET,
    #     phase='test',
    #     gpu=GPU,
    #     results_file=f'results_{PRETRAINING_DATASET}_{MODEL_NAME}.txt',
    #     checkpoint_dir=os.path.join(current_dir, f'checkpoints_evaluation_{PRETRAINING_DATASET}', MODEL_NAME),
    #     pre_trained_model_path=os.path.join(CHECKPOINT_DIR, 'last.ckpt'),
    #     evaluation='Linear',
    #     optimizer='Adam',
    #     learning_rate=1e-3,
    #     weight_decay=6e-5,
    #     betas=(0.9, 0.999),
    #     batch_size=256,
    #     epochs=100,
    #     patience=50
    # )