import os
import sys
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import supervised
from datasets import gib_uva_dataset
import utils


def train_supervised(
    model:Literal['EEGNet', 'EEGInception', 'Conformer'],
    pretraining_dataset: Literal['Cross', 'IntraOverall', 'IntraMD'],
    checkpoint_dir:str,
    resume_from_checkpoint:bool,
    gpu:int,
    lr:float,
    weight_decay:float = 6e-5,
    batch_size:int=1024,
    epochs:int=200,
    num_workers:int=20
    ):
    
    utils.set_deterministic()

    MODEL_NAME = f'{model}_Supervised' 
    CHECKPOINT_DIR = os.path.join(checkpoint_dir, MODEL_NAME)

    logger = TensorBoardLogger(os.path.join(checkpoint_dir, 'tb_logs_Supervised'), name=MODEL_NAME)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1score',
        dirpath=CHECKPOINT_DIR,
        filename=MODEL_NAME,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    # MODEL
    model = supervised.Supervised(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
    )

    # DATAMODULE
    dm = gib_uva_dataset.PreTrainingDataModule(
        training_method='Supervised',
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
        )

    if resume_from_checkpoint:
        saved_model_path = os.path.join(CHECKPOINT_DIR, 'last.ckpt')
        trainer.fit(model, dm, ckpt_path=saved_model_path)
    else:
        trainer.fit(model, dm)
        
if __name__ == '__main__':
    pass