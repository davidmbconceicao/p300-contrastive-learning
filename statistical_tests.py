import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from matplotlib.lines import Line2D 

from scipy.stats import shapiro, levene, wilcoxon, friedmanchisquare, ttest_rel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
import re

def read_and_separate_blocks(filepath, dataset='Intra'):
    """
    Reads a file, separates blocks based on fine_tuning values, and extracts numerical lists.
    This version excludes blocks containing 'Block X Mean' and handles the last block correctly.

    Args:
        filepath (str): Path to the file.

    Returns:
        dict: A dictionary where keys are fine_tuning values, and values are lists of extracted numerical lists.
    """
    blocks = {}
    
    with open(filepath, 'r') as file:
        content = file.read()

    # Adjust the regex to handle the last block correctly
    r_expression = r"Subject\s+\d+\s+with\s+(\d+)\s+chars\s+retraining(.*?)(?=Subject\s+\d+|$)" if dataset == 'Intra' else r"Subject\s+[A-Z]\s+with\s+(\d+)\s+chars\s+retraining(.*?)(?=Subject\s+[A-Z]|$)"
    block_pattern = r_expression
    matches = re.finditer(block_pattern, content, re.DOTALL)

    for match in matches:
        fine_tuning = int(match.group(1))
        block_data = match.group(2).strip()

        # Extract all lists of numbers in the block
        lists = re.findall(r"\[.*?\]", block_data)
        numerical_lists = [eval(lst) for lst in lists]

        if fine_tuning not in blocks:
            blocks[fine_tuning] = []
        blocks[fine_tuning].append(numerical_lists)
    
    for ft_value, block in blocks.items():
        blocks[ft_value] = np.array(block).reshape(-1, 15)
    
    return blocks

def statistical_analysis_Intra(file_paths:list, model, retraining_chars = [1, 5, 10], dataset='Intra'):
    strategies = ['Supervised', 'SimCLR', 'SupCon']
    
    # Create a larger figure for publication-quality plots
    if dataset=='Intra':
        fig, axes = plt.subplots(1, len(retraining_chars), figsize=(18, 6)) 
    else:
        fig, axes = plt.subplots(2, len(retraining_chars) // 2, figsize=(18, 12)) 
    axes = axes.flatten()
    
    retraining_char_dict = {i: [] for i in retraining_chars}
    # print(retraining_char_dict)
    
    for file_path in file_paths:
        blocks = read_and_separate_blocks(file_path, dataset)
        # print(blocks)
        for n_char in retraining_chars:
            retraining_char_dict[n_char].append(blocks[n_char])
        
        
    for retraining_idx, (retraining_char, reatraining_data) in enumerate(retraining_char_dict.items()):
        
        means = [np.mean(d, axis=0) for d in reatraining_data]
        std_deviations = [np.std(d, axis=0) for d in reatraining_data]
        
        print(f'Analysing data from {retraining_char} Retraining Chars')
        trial_results = {trial: [np.array(model)[:, trial] for model in reatraining_data] for trial in range(15)}
        
        friedman_results = []
        significant_trials = []  # To store significant trials for strategy 0 vs 2
        
        for trial, data in trial_results.items():
            print(f'\nN Trials -> {trial+1}')
            trial_data = [d for d in data]
            
            # Check normality and homogeneity of variance.
            normality = all(shapiro(d).pvalue > 0.05 for d in trial_data)
            homogeneity = levene(*trial_data).pvalue > 0.05
            print(f'\nIs data Normally Distributed? - {normality}')
            print(f'Is Variance Homogeneous? - {homogeneity}')
            
            f_stat, p_value = friedmanchisquare(*trial_data)
            print(f'\nFriedman Test results: (F-stat = {f_stat}, p_value = {p_value})')
            friedman_results.append((trial + 1, f_stat, p_value, "Friedman Test"))
            
            # Wilcoxon post-hoc test
            if p_value < 0.05:
                p_values = []
                pairs = list(itertools.combinations(range(len(trial_data)), 2))  
                adjusted_p_values = []
                
                for (i, j) in pairs:
                    stat, p_value_wilcoxon = wilcoxon(trial_data[i], trial_data[j])
                    p_values.append((i, j, p_value_wilcoxon))
                    
                    # Apply Benjamini-Hochberg correction
                    sorted_p_values = sorted(p_values, key=lambda x: x[2])  # Sort by p-value
                    m = len(sorted_p_values)  # Total number of tests
                    print(m)
                    
                    for k, (i, j, p) in enumerate(sorted_p_values):
                        adjusted_p = p * m / (k + 1)
                        adjusted_p = min(adjusted_p, 1)  # Ensure adjusted p-value does not exceed 1
                        adjusted_p_values.append((i, j, adjusted_p))
                
                print(f"Post-hoc Wilcoxon tests for trial {trial + 1}:")
                for (i, j, adj_p_value) in adjusted_p_values:
                    print(f"Wilcoxon test between {strategies[i]} and {strategies[j]}: p-value = {p_value}")
                    if dataset == 'Intra' and i == 0 and j == 2 and adj_p_value < 0.05:
                        significant_trials.append((trial + 1, i, j, adj_p_value))
    
        pallete = sns.color_palette(n_colors=8)
        blue, orange, green, red, gray = pallete[0], pallete[1], pallete[2], pallete[3], pallete[7]
        colors = [orange, green, blue]

        # Plot results
        for strategy_idx, (strategy, color) in enumerate(zip(strategies, colors)):
            axes[retraining_idx].plot(range(1, 16), means[strategy_idx], label=strategy, color=color, linewidth=2)
        
        # Only add vertical lines for significant differences between strategy 0 and 2
        for trial, i, j, p_value in significant_trials:
            if p_value < 0.05:  # For significance at p < 0.05
                # Get the values for strategy 0 and 2 at this trial
                y_val_i = means[0][trial-1]  # Supervised (Strategy 0)
                y_val_j = means[2][trial-1]  # SupCon (Strategy 2)
                
                # Calculate a small buffer to extend the line beyond the two points
                y_buffer = (max(means[0]) - min(means[0])) * 0.05  # Extend 5% of the range
                
                # Set the line color based on the p-value
                line_color = gray if p_value < 0.01 else red
                
                # Plot a vertical line from y_val_0 to y_val_2, with a buffer above and below
                axes[retraining_idx].plot([trial, trial], 
                                          [min(y_val_i, y_val_j) - y_buffer, max(y_val_i, y_val_j) + y_buffer], 
                                          color=line_color, linestyle='-', alpha=0.7, linewidth=2)
        
        axes[retraining_idx].set_title(f"N = {retraining_chars[retraining_idx]}", fontsize=16)
        axes[retraining_idx].set_xlabel("Trial", fontsize=14)
        axes[retraining_idx].set_ylabel("Average Accuracy", fontsize=14)
        axes[retraining_idx].set_ylim([0, 105])
        axes[retraining_idx].set_xticks([2, 4, 6, 8, 10, 12, 14])
        axes[retraining_idx].tick_params(axis='both', labelsize=12)
        axes[retraining_idx].grid(True, linestyle='--', alpha=0.6)
        axes[retraining_idx].legend(fontsize=12)
    
    # Create custom legend handles for the vertical lines
    red_line = Line2D([0], [0], color=red, lw=2, linestyle='-', label='p < 0.05')
    black_line = Line2D([0], [0], color=gray, lw=2, linestyle='-', label='p < 0.01')
    
    # Adjust layout for better spacing
    plt.suptitle(f'{model} \nAverage Accuracy across Trials for each N', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust title space
    
    # Add custom legend to the lower right corner
    fig.legend(handles=[black_line, red_line], loc='upper right', fontsize=12, title="Significance", title_fontsize=14, bbox_to_anchor=(0.99, 0.96))
    
    plt.show()
    
    
if __name__ == '__main__':
    file_paths = [
    '/workspace/project/GitHub/newScripts/IntraEvaluation/Supervised_results/EEGNet_1CLS_evaluation_results.txt',
    '/workspace/project/GitHub/newScripts/IntraEvaluation/SimCLR_results/EEGNet_1CLS_evaluation_results.txt',
    '/workspace/project/GitHub/newScripts/IntraEvaluation/SupCon_results/EEGNet_1CLS_evaluation_results.txt'
    ]
    statistical_analysis_Intra(file_paths, 'EEGNet')