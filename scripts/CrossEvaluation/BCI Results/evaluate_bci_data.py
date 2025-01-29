import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import subject_dependent_eval

current_dir = os.path.dirname(os.path.abspath(__file__))

# Parameters
pretraining_method = 'SupCon'
model = 'Conformer'
pretrained_model_path = '/workspace/project/GitHub/newScripts/IntraEvaluation/SupCon_results/Conformer_SupCon_temp_0.1/last.ckpt'
resume_from_checkpoint = False
gpu=0
temperature=0.1
lr=0.001

if pretraining_method in ['SupCon', 'SimCLR']:
    model_name = f'{model}_{pretraining_method}_temp_{temperature}'
elif pretraining_method == 'Supervised':
    model_name = f'{model}_Supervised' 

eval_results_file = os.path.join(current_dir, f'{pretraining_method}_results', f'{model}_1CLS_INTRA_50Folfd_evaluation_results.txt')
eval_checkpoint_dir = os.path.join(current_dir, f'{pretraining_method}_results', f'{model}_1CLS_INTRA_50Folfd_evaluation_checkpoints')

if __name__ == '__main__':
    print(model_name)
    
    subject_dependent_eval.subject_dependent_evaluation(
        pretraining_method=pretraining_method,
        eval_dataset='BCI',
        gpu=gpu,
        results_file=eval_results_file,
        checkpoint_dir=eval_checkpoint_dir,
        pretrained_model_path=pretrained_model_path
    )