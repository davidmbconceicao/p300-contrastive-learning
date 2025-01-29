import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_simclr import train_simclr
from train_supcon import train_supcon
from train_supervised import train_supervised
import subject_independent_eval
import subject_dependent_eval
import validation

current_dir = os.path.dirname(os.path.abspath(__file__))

# Parameters
pretraining_method = 'SupCon'
model = 'Conformer'
train_checkpoint_dir = os.path.join(current_dir, f'{pretraining_method}_results')
resume_from_checkpoint = False
gpu=0
temperature=0.1
lr=0.001

if pretraining_method in ['SupCon', 'SimCLR']:
    model_name = f'{model}_{pretraining_method}_temp_{temperature}'
elif pretraining_method == 'Supervised':
    model_name = f'{model}_Supervised' 
    
    
val_results_file = os.path.join(current_dir, f'{pretraining_method}_results', f'{model}_validation_results.txt')
val_checkpoint_dir = os.path.join(current_dir, f'{pretraining_method}_results', f'{model}_validation_checkpoints')

eval_results_file = os.path.join(current_dir, f'{pretraining_method}_results', f'{model}_1CLS_evaluation_results.txt')
eval_checkpoint_dir = os.path.join(current_dir, f'{pretraining_method}_results', f'{model}_1CLS_evaluation_checkpoints')

if __name__ == '__main__':
    print(model_name)

    # Train Model
    if pretraining_method == 'Supervised':
        train_supervised(
            model=model,
            pretraining_dataset='IntraOverall',
            checkpoint_dir=train_checkpoint_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            gpu=gpu,
            lr=lr,
        )
        
    elif pretraining_method == 'SupCon':
        train_supcon(
            model=model,
            pretraining_dataset='IntraOverall',
            checkpoint_dir=train_checkpoint_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            gpu=gpu,
            temperature=temperature,
            lr=lr,
        )
        
    elif pretraining_method == 'SimCLR':
        train_simclr(
            model=model,
            pretraining_dataset='IntraOverall',
            checkpoint_dir=train_checkpoint_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            gpu=gpu,
            temperature=temperature,
            lr=lr
        )

    # # Validate Model


    # validation.validate_model(
    #     pretraining_method=pretraining_method,
    #     dataset='IntraOverall',
    #     checkpoint_dir=val_checkpoint_dir,
    #     results_file=val_results_file,
    #     gpu=gpu,
    #     pretrained_model_path=os.path.join(train_checkpoint_dir, model_name, 'last.ckpt')
    #     )


    # Evaluate Model

    subject_dependent_eval.subject_dependent_evaluation(
        pretraining_method=pretraining_method,
        eval_dataset='IntraOverall',
        gpu=gpu,
        results_file=eval_results_file,
        checkpoint_dir=eval_checkpoint_dir,
        pretrained_model_path=os.path.join(train_checkpoint_dir, model_name, 'last.ckpt')
    )
    