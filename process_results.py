import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from matplotlib.lines import Line2D 
import re
import ast
import os
import math

from scipy.stats import shapiro, levene, wilcoxon, friedmanchisquare, ttest_rel, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM


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
    if dataset == 'Intra':
        r_expression = r"Subject\s+\d+\s+with\s+(\d+)\s+chars\s+retraining(.*?)(?=Subject\s+\d+|$)"
    elif dataset == 'BCI':
        r_expression = r"Subject\s+[A-Z]\s+with\s+(\d+)\s+chars\s+retraining(.*?)(?=Subject\s+[A-Z]|$)"
        
    matches = re.finditer(r_expression, content, re.DOTALL)
    
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


def read_als_file(filepath):
    """
    Reads an ALS file and combines all numerical lists into a single 2D list.

    Args:
        filepath (str): Path to the ALS file.

    Returns:
        list: A 2D list where each inner list is a numerical list from the file.
    """
    combined_lists = []

    with open(filepath, 'r') as file:
        content = file.read()

    # Match numerical lists in the file
    list_pattern = r"\[.*?\]"
    lists = re.findall(list_pattern, content)

    # Parse and combine all lists into a single 2D list
    for lst in lists:
        try:
            numerical_list = ast.literal_eval(lst)  # Safely parse the list
            combined_lists.append(numerical_list)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing list: {lst}. Skipping. Error: {e}")

    return combined_lists

def compute_ITR(accuracy, n_trials, dataset='BCI'):
    N = 36
    T = 2.5 + 2.1*n_trials if dataset == 'BCI' else 2.5 + 3*n_trials
    
    term1 = math.log2(N)
    term2 = accuracy * math.log2(accuracy)
    term3 = (1-accuracy) * math.log2((1-accuracy) / (N-1))
    
    itr = (term1 + term2 + term3) * 60 / T
    return itr

def parse_file(filename):
    results = []
    with open(filename, 'r') as file:
        block = []
        for line in file:
            line = line.strip()
            if line.startswith('Subject'):
                if block:  # If there's an existing block, calculate its mean
                    block_mean = np.round(np.mean(block, axis=0), 2)
                    results.append(block_mean)
                    block = []
            elif line.startswith('['):
                data = eval(line)  # Convert string to list
                block.append(data)
        if block:  # Process the last block
            block_mean = np.round(np.mean(block, axis=0), 2)
            results.append(block_mean)
    return results

def compute_group_statistics(data, indices):
    """Compute mean and std for specific subject groups."""
    group_data = np.array(data)[indices]
    group_mean = np.round(np.mean(group_data, axis=0), 2)
    group_std = np.round(np.std(group_data, axis=0), 2)
    return group_mean, group_std

def format_mean_std(mean, std, indices=range(15)):
    # Format as "mean ± std"
    return ' & '.join([f'{mean[i]} ± {std[i]}' for i in indices])


def get_differences(file_paths:list, model, retraining_chars = [1, 5, 10], dataset='Intra'):
    
    strategies = ['Supervised', 'SimCLR', 'SupCon']
    subjects = ['MD SUBJECTS', 'CONTROL SUBJECTS', 'ALL SUBJECTS']
    trials_analysed = range(15)
    
    md_subjects_data = {i: [] for i in retraining_chars}
    control_subjects_data = {i: [] for i in retraining_chars}
    all_subjects_data = {i: [] for i in retraining_chars}
    
    data = [md_subjects_data, control_subjects_data, all_subjects_data] if dataset == 'Intra' else [all_subjects_data]
    
    for file_path in file_paths:
        blocks = read_and_separate_blocks(file_path, dataset)
        for n_char in retraining_chars:
            control_subjects_data[n_char].append(blocks[n_char][:350]) if dataset == 'Intra' else None
            md_subjects_data[n_char].append(blocks[n_char][350:]) if dataset == 'Intra' else None
            all_subjects_data[n_char].append(blocks[n_char])
    
    for sub, retraining_char_dict in enumerate(data):   
        
        sub = 2 if dataset == 'BCI'else sub
        print(f'\n{subjects[sub]}\n')
        
        for retraining_idx, (retraining_char, reatraining_data) in enumerate(retraining_char_dict.items()):
            
            means = [np.mean(d, axis=0) for d in reatraining_data]
            # print(means)
            std_deviations = [np.std(d, axis=0) for d in reatraining_data]
            
            print(f'Analysing data from {retraining_char} Retraining Chars')
            trial_results = {trial: [np.array(model)[:, trial] for model in reatraining_data] for trial in trials_analysed}
            
            friedman_results = []
            differences = {'SimCLR': [], 'SupCon': []}
    
            
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
                    
                        
                    for k, (i, j, p) in enumerate(sorted_p_values):
                        adjusted_p = p * m / (k + 1)
                        adjusted_p = min(adjusted_p, 1)  # Ensure adjusted p-value does not exceed 1
                        adjusted_p_values.append((i, j, adjusted_p))
                            
                    
                    print(f"Post-hoc Wilcoxon tests for trial {trial + 1}:")
                    
                    for (i, j, adj_p_value) in adjusted_p_values:
                        diff = means[j][trial] - means[i][trial]
                        print(f"Wilcoxon test between {strategies[j]} and {strategies[i]}: p-value = {adj_p_value}: diff -> {means[j][trial]:.2f} - {means[i][trial]:.2f} = {diff:.2f}")
                        if adj_p_value < 0.05 and strategies[i] == 'Supervised':
                            differences[strategies[j]].append(diff)
                            
            print(f'\n {subjects[sub]} Differences {retraining_char}: {differences}')
            print(f'\nAverage Difference: ({np.mean(differences["SimCLR"]):.2f}, {np.mean(differences["SupCon"]):.2f})\n')   
            

def plot_baseline_comparisons(file_paths: list, model_names: list):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    pallete = sns.color_palette(n_colors=8)
    blue, orange, green, red, gray = pallete[0], pallete[1], pallete[2], pallete[3], pallete[7]
    colors = [blue, orange]
    
    titles = ['EEGNet', 'EEG-Inception']

    plot_groups = {
        0: ['EEGNet Modified', 'EEGNet Original'],
        1: ['EEG-Inception Modified', 'EEG-Inception Original'],
    }

    def configure_axis(ax, model):
        ax.set_title(model, fontsize=24)
        ax.set_xlabel("Number of Trials", fontsize=22)
        ax.set_ylabel("Average Accuracy", fontsize=22)
        ax.set_ylim([0, 105])
        ax.set_xticks(range(2, 16, 2))
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Prepare data for statistical analysis
    all_data = {group_idx: {} for group_idx in plot_groups.keys()}
    trial_indices = range(1, 16)

    for file_path, model_name in zip(file_paths, model_names):
        model_blocks = [key for key, models in plot_groups.items() if model_name in models]
        block = read_and_separate_blocks(file_path, dataset='BCI')

        for key in model_blocks:
            all_data[key][model_name] = block[85]

    # Perform statistical analysis and plot
    for group, group_data in all_data.items():
        means = []
        for i, (key, values) in enumerate(group_data.items()):
            mean = np.mean(values, axis=0)
            means.append(mean)
            axes[group].plot(trial_indices, mean, label=key,
                             color=colors[i], linewidth=2)
        configure_axis(axes[group], titles[group])
        models_legend = axes[group].legend(fontsize=20, loc='upper left')
        red_line = Line2D([0], [0], color=red, lw=2, linestyle='-', label='p < 0.05')
        black_line = Line2D([0], [0], color=gray, lw=2, linestyle='-', label='p < 0.01')
        axes[group].legend(handles=[black_line, red_line], loc='lower right', fontsize=20, 
                           title="Significance", title_fontsize=20)
        
        #  bbox_to_anchor=(0.99, 0.99)
        axes[group].add_artist(models_legend)
        
        # print(group_data.values())
        trial_results = {trial: [np.array(data)[:, trial] for data in group_data.values()] for trial in range(15)}

        stat_results = []

        # Loop through each trial to perform statistical tests
        for trial, data in trial_results.items():
            print(f'\nN Trials -> {trial+1}')
            trial_data = [d for d in data]

            # Check normality and homogeneity of variance.
            normality = all(shapiro(d).pvalue > 0.05 for d in trial_data)
            homogeneity = levene(*trial_data).pvalue > 0.05
            print(f'\nIs data Normally Distributed? - {normality}')
            print(f'Is Variance Homogeneous? - {homogeneity}')

            # Perform the Mann-Whitney U Test
            stat, p_value = mannwhitneyu(*trial_data, alternative='two-sided')
            print(f'\nMann Whitney U Test results: (F-stat = {stat}, p_value = {p_value})')
            stat_results.append((trial + 1, stat, p_value, "Mann Whitney U Test"))

            # Plot significance lines if p-value is less than 0.05 or 0.01
            if p_value < 0.05:
                
                y_val_i = means[0][trial] 
                y_val_j = means[1][trial] 

                y_buffer = (max(means[0]) - min(means[0])) * 0.05
                
                # Set the line color based on p-value
                line_color = "gray" if p_value < 0.01 else "red"

                # Plot the significance line between the two models' values at this trial
                axes[group].plot([trial+1, trial+1], 
                                          [min(y_val_i, y_val_j) - y_buffer, max(y_val_i, y_val_j) + y_buffer], 
                                          color=line_color, linestyle='-', alpha=0.7, linewidth=2)
    
    # plt.suptitle('Performance Comparison Between Original and Modified Models', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    plt.show()


def plot_baseline(file_paths: list, model_names: list, dataset='BCI'):
    # Setup the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette(n_colors=8)
    blue, orange, green, red, purple, gray = palette[0], palette[1], palette[2], palette[3], palette[4], palette[7]
    
    trial_indices = range(1,16) if dataset == 'BCI' else range(1,11)
    
    # Model Colors and Plot groups
    # model_colors = {
    #     'EEGNet': blue, 'EEGInception': orange, 'Conformer': green
    # }

    model_colors = [blue, orange, green, red, purple, gray]

    # Prepare data for statistical analysis
    all_data = {model_name: [] for model_name in model_names}

    for file_path, model_name in zip(file_paths, model_names):
        if dataset == 'BCI':
            block = read_and_separate_blocks(file_path, dataset='BCI')
            # print(block)
            all_data[model_name] = block[85]  # Assuming block[85] holds the required accuracy data
        elif dataset == 'ALS':
            block = read_als_file(file_path)
            all_data[model_name] = block

    # Perform Friedman's test on all trials
    trial_results = {trial-1: [np.array(all_data[model_name])[:, trial-1] for model_name in model_names] for trial in trial_indices}
    friedman_results = []
    p_values = []

    # Perform Friedman's test across all trials
    for trial, data in trial_results.items():
        stat, p_value = friedmanchisquare(*data)
        friedman_results.append((trial + 1, stat, p_value))
        p_values.append(p_value)

    # Wilcoxon post-hoc test
        if p_value < 0.05:
            p_values_wilcoxon = []
            pairs = list(itertools.combinations(range(len(data)), 2))  
            adjusted_p_values = []
            
            for (i, j) in pairs:
                stat, p_value_wilcoxon = wilcoxon(data[i], data[j])
                p_values_wilcoxon.append((i, j, p_value_wilcoxon))
                    
            # Apply Benjamini-Hochberg correction
            sorted_p_values = sorted(p_values_wilcoxon, key=lambda x: x[2])  # Sort by p-value
            m = len(sorted_p_values)  # Total number of tests
                    
            for k, (i, j, p) in enumerate(sorted_p_values):
                adjusted_p = p * m / (k + 1)
                adjusted_p = min(adjusted_p, 1)  # Ensure adjusted p-value does not exceed 1
                adjusted_p_values.append((i, j, adjusted_p))
                
            print(f"\nPost-hoc Wilcoxon tests for trial {trial + 1}:")
            for (i, j, adj_p_value) in adjusted_p_values:
                print(f"Wilcoxon test between {model_names[i]} and {model_names[j]}: p-value = {adj_p_value}")

    ax2 = ax.twinx()
    ax2.set_ylabel("ITR (bits/min)", fontsize=22, color='gray')
    ax2.set_ylim([0, 13])
    ax2.set_yticks(range(0, 13, 3))
    ax2.tick_params(axis='y', labelsize=16, colors='gray')
    ax2.grid(False)
    
    # Plotting the mean of each model
    for i, model_name in enumerate(model_names):
        mean = np.mean(all_data[model_name], axis=0)
        print(f'{model_name}: \nMean = {mean}')
        ax.plot(trial_indices, mean, label=model_name, color=model_colors[i], linewidth=2)
        
        itr_values = [compute_ITR(acc/100, trial+1, dataset) for trial, acc in enumerate(mean)]
        print(f'\n{itr_values}')
        ax2.plot(trial_indices, itr_values, label=f'{model_name} ITR', color=model_colors[i], linestyle='dotted')

    # Configure axis and title
    ax.set_xlabel("Number of Trials", fontsize=22)
    ax.set_ylabel("Average Accuracy", fontsize=22)
    ax.set_ylim([0, 105])
    ax.set_xticks(trial_indices)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=16)
    # ax.set_title(f'{dataset} Dataset\nPerformance Comparison Between Models Across Trials', fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    plt.show()
    
          
def plot_intra_and_BCI(file_paths:list, model, retraining_chars = [1, 5, 10], 
                       dataset='Intra', significance_legend_loc = ['upper right', 'lower right', 'lower right']):
    strategies = ['Supervised', 'SimCLR', 'SupCon']
    
    # Create a larger figure for publication-quality plots
    if len(retraining_chars)<=3:
        fig, axes = plt.subplots(1, len(retraining_chars), figsize=(20, 6)) 
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
                
                    
                for k, (i, j, p) in enumerate(sorted_p_values):
                    adjusted_p = p * m / (k + 1)
                    adjusted_p = min(adjusted_p, 1)  # Ensure adjusted p-value does not exceed 1
                    adjusted_p_values.append((i, j, adjusted_p))
                        
                
                print(f"Post-hoc Wilcoxon tests for trial {trial + 1}:")
                for (i, j, adj_p_value) in adjusted_p_values:
                    print(f"Wilcoxon test between {strategies[i]} and {strategies[j]}: p-value = {adj_p_value}")
                    # if dataset == 'Intra' and i == 0 and j == 2 and adj_p_value < 0.05:
                        # Add only significant trials involving the top two strategies
                    top_two_indices = np.argsort([means[s][trial] for s in range(len(strategies))])[-2:]
                    if i in top_two_indices and j in top_two_indices and adj_p_value < 0.05:
                        significant_trials.append((trial + 1, i, j, adj_p_value))
                        # significant_trials.append((trial + 1, i, j, adj_p_value))
    
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
                y_val_i = means[0][trial - 1]  # Supervised (Strategy 0)
                y_val_j = means[2][trial - 1]  # SupCon (Strategy 2)
                
                # Calculate a small buffer to extend the line beyond the two points
                y_buffer = (max(means[0]) - min(means[0])) * 0.05  # Extend 5% of the range
                
                # Set the line color based on the p-value
                line_color = gray if p_value < 0.01 else red
                
                # Plot a vertical line from y_val_0 to y_val_2, with a buffer above and below
                axes[retraining_idx].plot([trial, trial], 
                                          [min(y_val_i, y_val_j) - y_buffer, max(y_val_i, y_val_j) + y_buffer], 
                                          color=line_color, linestyle='-', alpha=0.7, linewidth=2)
        
        axes[retraining_idx].set_title(f"N = {retraining_chars[retraining_idx]}", fontsize=22)
        axes[retraining_idx].set_xlabel("Number of Trials", fontsize=22)
        axes[retraining_idx].set_ylabel("Average Accuracy", fontsize=22)
        axes[retraining_idx].set_ylim([0, 105])
        axes[retraining_idx].set_xticks([2, 4, 6, 8, 10, 12, 14])
        axes[retraining_idx].tick_params(axis='both', labelsize=16)
        axes[retraining_idx].grid(True, linestyle='--', alpha=0.6)
        # Create legends
        main_legend = axes[retraining_idx].legend(fontsize=17, loc='upper left')
        
        red_line = Line2D([0], [0], color=red, lw=2, linestyle='-', label='p < 0.05')
        black_line = Line2D([0], [0], color=gray, lw=2, linestyle='-', label='p < 0.01')
        axes[retraining_idx].legend(handles=[black_line, red_line], loc=significance_legend_loc[retraining_idx], fontsize=16, 
               title="Significance", title_fontsize=17)
        
        axes[retraining_idx].add_artist(main_legend)
        
        print(means)
    
    # Adjust layout for better spacing
    # plt.suptitle(f'{model} Performances on the BCI Comp Dataset', fontsize=20)
    plt.suptitle(model, fontsize=26)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust title space
    
    plt.show()
    
    
def get_differences_als(files, model):

    strategies = ['Supervised', 'SimCLR', 'SupCon']
    
    blocks = []
    means = []
    print(f'\n{model}')
    for idx, file_path in enumerate(files):
        print(f'\n{strategies[idx]}')
        block = read_als_file(file_path)
        blocks.append(block)
        mean = np.mean(block, axis=0)
        means.append(mean)
        std = np.std(block, axis=0)
        
        print( ' & '.join([f'{mean[i]:.2f} ± {std[i]:.2f}' for i in [0, 4, 9]]))
        
    trial_results = {trial: [np.array(strategy)[:, trial] for strategy in blocks] for trial in range(10)}
    # print(trial_results)
    friedman_results = []
    differences = {'SimCLR': [], 'SupCon': []}
    
            
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
                    
                        
            for k, (i, j, p) in enumerate(sorted_p_values):
                adjusted_p = p * m / (k + 1)
                adjusted_p = min(adjusted_p, 1)  # Ensure adjusted p-value does not exceed 1
                adjusted_p_values.append((i, j, adjusted_p))
                            
                    
            print(f"Post-hoc Wilcoxon tests for trial {trial + 1}:")
                    
            for (i, j, adj_p_value) in adjusted_p_values:
                diff = means[j][trial] - means[i][trial]
                print(f"Wilcoxon test between {strategies[j]} and {strategies[i]}: p-value = {adj_p_value}: diff -> {means[j][trial]:.2f} - {means[i][trial]:.2f} = {diff:.2f}")
                if adj_p_value < 0.05 and strategies[i] == 'Supervised':
                    differences[strategies[j]].append(diff)
                            
    print(f'\nDifferences: {differences}')
    print(f'\nAverage Difference: ({np.mean(differences["SimCLR"]):.2f}, {np.mean(differences["SupCon"]):.2f})\n')
    
def plot_als_results(files_paths:list, model_names:list):
    
    strategies = ['Supervised', 'SimCLR', 'SupCon']
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    pallete = sns.color_palette(n_colors=8)
    blue, orange, green, red, gray = pallete[0], pallete[1], pallete[2], pallete[3], pallete[7]
    colors = [orange, green, blue]
    
    def configure_axis(ax, model):
        ax.set_title(model, fontsize=24)
        ax.set_xlabel("Number of Trials", fontsize=22)
        ax.set_ylabel("Average Accuracy", fontsize=22)
        ax.set_ylim([0, 105])
        ax.set_xticks(range(1, 11))
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    for ax_numbr, model in enumerate(model_names):
        blocks = {x: None for x in strategies}
        means = []
        for file_path, strategy, color in zip(files_paths[ax_numbr], strategies, colors):
            block = read_als_file(file_path)
            # print(len(block))
            blocks[strategy] = block
            
            mean = np.mean(block, axis=0)
            means.append(mean)
            
            axes[ax_numbr].plot(range(1,11), mean, label=strategy, color=color, linewidth=2)
         
        configure_axis(axes[ax_numbr], model)
        models_legend = axes[ax_numbr].legend(fontsize=18, loc='upper left')   
        red_line = Line2D([0], [0], color=red, lw=2, linestyle='-', label='p < 0.05')
        black_line = Line2D([0], [0], color=gray, lw=2, linestyle='-', label='p < 0.01')
        axes[ax_numbr].legend(handles=[black_line, red_line], loc='upper right', fontsize=16,
                              title="Significance", title_fontsize=18)
        axes[ax_numbr].add_artist(models_legend)
        
        trial_results = {trial: [np.array(data)[:, trial] for data in blocks.values()] for trial in range(10)}
        # print(trial_results)
        p_values = []
        friedman_results = []
        significant_trials = []

        # Perform Friedman's test across all trials
        for trial, data in trial_results.items():
            stat, p_value = friedmanchisquare(*data)
            friedman_results.append((trial + 1, stat, p_value))
            p_values.append(p_value)

        # Wilcoxon post-hoc test
            if p_value < 0.05:
                p_values_wilcoxon = []
                pairs = list(itertools.combinations(range(len(data)), 2))  
                # print(pairs)
                adjusted_p_values = []
                
                for (i, j) in pairs:
                    stat, p_value_wilcoxon = wilcoxon(data[i], data[j])
                    p_values_wilcoxon.append((i, j, p_value_wilcoxon))
                        
                # Apply Benjamini-Hochberg correction
                sorted_p_values = sorted(p_values_wilcoxon, key=lambda x: x[2])  # Sort by p-value
                m = len(sorted_p_values)  # Total number of tests
                    
                for k, (i, j, p) in enumerate(sorted_p_values):
                    adjusted_p = p * m / (k + 1)
                    adjusted_p = min(adjusted_p, 1)  # Ensure adjusted p-value does not exceed 1
                    adjusted_p_values.append((i, j, adjusted_p))
    
                    
                print(f"Post-hoc Wilcoxon tests for trial {trial + 1}:")
                for (i, j, adj_p_value) in adjusted_p_values:
                    print(f"Wilcoxon test between {strategies[i]} and {strategies[j]}: p-value = {adj_p_value}")
                    significant_trials.append((trial + 1, i, j, adj_p_value)) if adj_p_value < 0.05 else None
                        
        for trial, i, j, p_value in significant_trials:
            if p_value < 0.05 and i == 0 and j == 2:  # For significance at p < 0.05
                # Get the values for strategy 0 and 2 at this trial
                y_val_i = means[0][trial - 1]  # Supervised (Strategy 0)
                y_val_j = means[2][trial - 1]  # SupCon (Strategy 2)
                
                # Calculate a small buffer to extend the line beyond the two points
                y_buffer = (max(means[0]) - min(means[0])) * 0.05  # Extend 5% of the range
                
                # Set the line color based on the p-value
                line_color = gray if p_value < 0.01 else red
                
                # Plot a vertical line from y_val_0 to y_val_2, with a buffer above and below
                axes[ax_numbr].plot([trial, trial], 
                                    [min(y_val_i, y_val_j) - y_buffer, max(y_val_i, y_val_j) + y_buffer], 
                                    color=line_color, linestyle='-', alpha=0.7, linewidth=2)       
    
    # plt.suptitle(f'ALS Dataset \nAverage Accuracy across Number of Trials for each Model', fontsize=18)    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    
    
def plot_intra_cross(file_paths:list, model, retraining_chars = [1, 5, 10, 25, 40, 85], dataset='BCI'):
    strategies = ['SimCLR Cross', 'SimCLR Intra']
    
    # Create a larger figure for publication-quality plots
    if dataset=='ALS':
        fig, axes = plt.subplots(1, len(retraining_chars), figsize=(18, 6)) 
        n_trials = range(10)
        x_values = range(1,11)
        x_label = x_values
    else:
        fig, axes = plt.subplots(2, len(retraining_chars) // 2, figsize=(18, 12)) 
        axes = axes.flatten()
        n_trials = range(15)
        x_values = range(1,16)
        x_label = [2, 4, 6, 8, 10, 12, 14]
    
    retraining_char_dict = {i: [] for i in retraining_chars}
    # print(retraining_char_dict)
    
    for file_path in file_paths:
        if dataset=='BCI':
            blocks = read_and_separate_blocks(file_path, dataset)
            for n_char in retraining_chars:
                retraining_char_dict[n_char].append(blocks[n_char])
        else:
            blocks = read_als_file(file_path)
            retraining_char_dict[1].append(blocks)

    for retraining_idx, (retraining_char, reatraining_data) in enumerate(retraining_char_dict.items()):
        
        means = [np.mean(d, axis=0) for d in reatraining_data]
        std_deviations = [np.std(d, axis=0) for d in reatraining_data]
        
        print(f'Analysing data from {retraining_char} Retraining Chars')
        trial_results = {trial: [np.array(model)[:, trial] for model in reatraining_data] for trial in n_trials}
        
        stat_results = []
        
        # Loop through each trial to perform statistical tests
        for trial, data in trial_results.items():
            print(f'\nN Trials -> {trial+1}')
            trial_data = [d for d in data]

            # Check normality and homogeneity of variance.
            normality = all(shapiro(d).pvalue > 0.05 for d in trial_data)
            homogeneity = levene(*trial_data).pvalue > 0.05
            print(f'\nIs data Normally Distributed? - {normality}')
            print(f'Is Variance Homogeneous? - {homogeneity}')

            # Perform the Mann-Whitney U Test
            stat, p_value = mannwhitneyu(*trial_data, alternative='two-sided')
            print(f'\nMann Whitney U Test results: (F-stat = {stat}, p_value = {p_value})')
            stat_results.append((trial + 1, stat, p_value))
    
        pallete = sns.color_palette(n_colors=8)
        blue, orange, green, red, gray = pallete[0], pallete[1], pallete[2], pallete[3], pallete[7]
        colors = [orange, green, blue]
        ax = axes[retraining_idx] if dataset == 'BCI' else axes
        significance_legend_loc = 'upper right' if retraining_idx in [0, 1] else 'lower right'
        
        # Plot results
        for strategy_idx, (strategy, color) in enumerate(zip(strategies, colors)):
            ax.plot(x_values, means[strategy_idx], label=strategy, color=color, linewidth=2)
        
        # Only add vertical lines for significant differences between strategy 0 and 2
        for trial, stat, p_value in stat_results:
            if p_value < 0.05:  # For significance at p < 0.05
                # Get the values for strategy 0 and 2 at this trial
                y_val_i = means[0][trial - 1] 
                y_val_j = means[1][trial - 1]  
                
                # Calculate a small buffer to extend the line beyond the two points
                y_buffer = (max(means[0]) - min(means[0])) * 0.05  # Extend 5% of the range
                
                # Set the line color based on the p-value
                line_color = gray if p_value < 0.01 else red
                
                # Plot a vertical line from y_val_0 to y_val_2, with a buffer above and below
                ax.plot([trial, trial], [min(y_val_i, y_val_j) - y_buffer, max(y_val_i, y_val_j) + y_buffer], 
                        color=line_color, linestyle='-', alpha=0.7, linewidth=2)
        
        ax.set_title(f"N = {retraining_chars[retraining_idx]}", fontsize=22) if dataset=='BCI' else None
        ax.set_xlabel("Number of Trials", fontsize=22)
        ax.set_ylabel("Average Accuracy", fontsize=22)
        ax.set_ylim([0, 105])
        ax.set_xticks(x_label)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        main_legend = ax.legend(fontsize=18, loc='upper left')
        # Create custom legend handles for the vertical lines
        red_line = Line2D([0], [0], color=red, lw=2, linestyle='-', label='p < 0.05')
        black_line = Line2D([0], [0], color=gray, lw=2, linestyle='-', label='p < 0.01')
        ax.legend(handles=[black_line, red_line], loc= significance_legend_loc , fontsize=16, 
               title="Significance", title_fontsize=18)
        ax.add_artist(main_legend)
    
    # Adjust layout for better spacing
    # plt.suptitle(f'{model} SimCLR Cross vs. {model} SimCLR Intra\n{dataset} Dataset Results', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust title space
    
    plt.show()
    
    
def plot_baseline_and_studied(file_paths: list, model_names: list, dataset = 'BCI'):
    # Setup the figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    palette = sns.color_palette(n_colors=8)
    blue, orange, green, red, purple, gray = palette[0], palette[1], palette[2], palette[3], palette[4], palette[7]

    model_colors = [blue, orange]

    for ax, idx in enumerate(range(0, len(file_paths), 2)):
        
        model_files = [file_paths[idx], file_paths[idx+1]]
        models = [model_names[idx], model_names[idx+1]]
        # Prepare data for statistical analysis
        all_data = {model_name: [] for model_name in models}
        trial_indices = range(1, 16) if dataset == 'BCI' else range(1,11)
        
        means = []
        
        ax2 = axes[ax].twinx()
        ax2.set_ylabel("ITR (bits/min)", fontsize=14, color='gray')
        ax2.set_ylim([0, 13])
        ax2.set_yticks(range(0, 13, 3))
        ax2.tick_params(axis='y', labelsize=12, colors='gray')
        ax2.grid(False)

        for file_path, model_name, color in zip(model_files, models, model_colors):
            if dataset == 'BCI':
                block = read_and_separate_blocks(file_path, dataset='BCI')
                # print(block)
                all_data[model_name] = block[85]  # Assuming block[85] holds the required accuracy data
            elif dataset == 'ALS':
                block = read_als_file(file_path)
                all_data[model_name] = block
            
            mean = np.mean(all_data[model_name], axis=0)
            means.append(mean)
            axes[ax].plot(trial_indices, mean, label=model_name, color=color, linewidth=2)
            
            itr_values = [compute_ITR(acc/100, trial+1, 'BCI') for trial, acc in enumerate(mean)]
            ax2.plot(trial_indices, itr_values, label=f'{model_name} ITR', color=color, linestyle='dotted')
            print(f'\n{model_name}: \nMean = {mean} \nITR = {itr_values}')
            # print(f'ITR = {itr_values}')

        trial_results = {trial-1: [np.array(all_data[model_name])[:, trial-1] for model_name in models] for trial in trial_indices}
        stat_results = []
        p_values = []

        # Loop through each trial to perform statistical tests
        for trial, data in trial_results.items():
            print(f'\nN Trials -> {trial+1}')
            trial_data = [d for d in data]

            # Check normality and homogeneity of variance.
            normality = all(shapiro(d).pvalue > 0.05 for d in trial_data)
            homogeneity = levene(*trial_data).pvalue > 0.05
            print(f'\nIs data Normally Distributed? - {normality}')
            print(f'Is Variance Homogeneous? - {homogeneity}')

            # Perform the Mann-Whitney U Test
            stat, p_value = mannwhitneyu(*trial_data, alternative='two-sided')
            print(f'\nMann Whitney U Test results: (F-stat = {stat}, p_value = {p_value})')
            stat_results.append((trial + 1, stat, p_value, "Mann Whitney U Test"))

            # Plot significance lines if p-value is less than 0.05 or 0.01
            if p_value < 0.05:
                # Find the values of the two models a this trial
                
                y_val_i = means[0][trial] 
                y_val_j = means[1][trial] 

                y_buffer = (max(means[0]) - min(means[0])) * 0.05
                
                # Set the line color based on p-value
                line_color = "gray" if p_value < 0.01 else "red"

                # Plot the significance line between the two models' values at this trial
                axes[ax].plot([trial+1, trial+1], [min(y_val_i, y_val_j) - y_buffer, max(y_val_i, y_val_j) + y_buffer], 
                            color=line_color, linestyle='-', alpha=0.7, linewidth=2)
            

        # Configure axis and title
        axes[ax].set_xlabel("Number of Trials", fontsize=14)
        axes[ax].set_ylabel("Average Accuracy", fontsize=14)
        axes[ax].set_ylim([0, 105])
        axes[ax].set_xticks(trial_indices)
        axes[ax].tick_params(axis='both', labelsize=12)
        axes[ax].grid(True, linestyle='--', alpha=0.6)
        axes[ax].legend(fontsize=12)
        axes[ax].set_title(f'{models[0]}', fontsize=18)

    red_line = Line2D([0], [0], color=red, lw=2, linestyle='-', label='p < 0.05')
    black_line = Line2D([0], [0], color=gray, lw=2, linestyle='-', label='p < 0.01')
    fig.legend(handles=[black_line, red_line], loc='upper right', fontsize=12, title="Significance", title_fontsize=14, bbox_to_anchor=(0.99, 0.99))
    
    plt.suptitle('Baseline vs. Pre-Trained Models Performance Comparison\n on the BCI Comp Dataset', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    plt.show()