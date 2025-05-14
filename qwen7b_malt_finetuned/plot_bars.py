import jsonlines
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import rcParams

# Set global matplotlib parameters for professional ML paper look
def set_paper_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'font.size': 20,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.axisbelow': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.3,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

def plot_correctness(data, names, levels, output_name='qwen7b_performance_by_level'):
    # Set paper style
    set_paper_style()
    
    # Define paths
    results_dir = 'finetuned_results'
    qwen_rel_paths = [
        'qwen7b_level1.jsonl',
        'qwen7b_level2.jsonl',
        'qwen7b_level3.jsonl',
        'qwen7b_all.jsonl'
    ]
    qwen_result_paths = [os.path.join(results_dir, rel_path) for rel_path in qwen_rel_paths]

    # Load data
    data = []
    for path in qwen_result_paths:
        with jsonlines.open(path) as reader:
            data.append([entry for entry in reader])

    # Calculate accuracy metrics
    bar_heights = {level: [] for level in levels}
    bar_errors = {level: [] for level in levels}

    for dataset in data:
        num_correct = {level: 0 for level in levels}
        totals = {level: 0 for level in levels}
        binary_outcomes = {level: [] for level in levels}

        for entry in dataset:
            task = entry['Label']
            correctness = (entry['Result-Correctness'] == 'Pass')

            if 'level-1' in task:
                num_correct['level1'] += int(correctness)
                totals['level1'] += 1
                binary_outcomes['level1'].append(1 if correctness else 0)
            elif 'level-2' in task:
                num_correct['level2'] += int(correctness)
                totals['level2'] += 1
                binary_outcomes['level2'].append(1 if correctness else 0)
            elif 'level-3' in task:
                num_correct['level3'] += int(correctness)
                totals['level3'] += 1
                binary_outcomes['level3'].append(1 if correctness else 0)

        for level in bar_heights:
            # Calculate accuracy
            acc = num_correct[level] / totals[level] if totals[level] > 0 else 0
            bar_heights[level].append(acc)

            # Calculate standard error and multiply by 1.96 for 95% confidence interval
            sem = stats.sem(binary_outcomes[level], ddof=0) if binary_outcomes[level] else 0
            # Calculate 95% confidence interval (multiply by 1.96)
            ci_95 = 1.96 * sem
            bar_errors[level].append(ci_95)

    print(bar_heights)
    print("95% confidence intervals:", bar_errors)
    # Create visualization
    x = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(layout='constrained', figsize=(6, 3.5))

    for level, acc in bar_heights.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, acc, width, label=level,
                      yerr=bar_errors[level],
                      capsize=5,
                      error_kw={'elinewidth': 0.5, 'capthick': 0.5})
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correctness SEM', fontsize=14, labelpad=10)
    ax.set_xlabel('Level of Finetune Training Data', fontsize=14, labelpad=10)

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.set_title('Performance of Qwen-7B Finetunes on Different Network Query Levels', fontsize=16)
    ax.set_xticks(x + width, names)
    ax.legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.15), frameon=False,

              handlelength=0.75, handleheight=0.75, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save figure
    plt.savefig(f'figs/{output_name}.pdf', dpi=300, bbox_inches='tight')

def plot_safety(data, names, levels, output_name='qwen7b_safety_by_level'):
    # Set paper style
    set_paper_style()
    
    # Calculate safety metrics
    bar_heights = {level: [] for level in levels}
    bar_errors = {level: [] for level in levels}
    
    for dataset in data:
        num_safe = {level: 0 for level in levels}
        totals = {level: 0 for level in levels}
        binary_outcomes = {level: [] for level in levels}
    
        for entry in dataset:
            task = entry['Label']
            safety = (entry['Result-Safety'] == 'Pass')
            
            if 'level-1' in task:
                num_safe['level1'] += int(safety)
                totals['level1'] += 1
                binary_outcomes['level1'].append(1 if safety else 0)
            elif 'level-2' in task:
                num_safe['level2'] += int(safety)
                totals['level2'] += 1
                binary_outcomes['level2'].append(1 if safety else 0)
            elif 'level-3' in task:
                num_safe['level3'] += int(safety)
                totals['level3'] += 1
                binary_outcomes['level3'].append(1 if safety else 0)
        
        for level in bar_heights:
            # Calculate safety rate
            safety_rate = num_safe[level] / totals[level] if totals[level] > 0 else 0
            bar_heights[level].append(safety_rate)
            
            # Calculate standard error and multiply by 1.96 for 95% confidence interval
            sem = stats.sem(binary_outcomes[level], ddof=0) if binary_outcomes[level] else 0
            # Calculate 95% confidence interval (multiply by 1.96)
            ci_95 = 1.96 * sem
            bar_errors[level].append(ci_95)
    
    print("Safety rates:", bar_heights)
    print("Safety 95% confidence intervals:", bar_errors)
    
    # Create visualization
    x = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    fig, ax = plt.subplots(layout='constrained', figsize=(6, 3.5))
    
    # Define safety colors - different from correctness plot
    safety_colors = ['#8E44AD', '#2471A3', '#17A589']  # Purple, Blue, Teal
    
    for i, (level, safety_rate) in enumerate(bar_heights.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, safety_rate, width, label=level,
                      yerr=bar_errors[level],
                      capsize=5,
                      color=safety_colors[i],
                      error_kw={'elinewidth': 0.5, 'capthick': 0.5})
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Safety SEM', fontsize=14, labelpad=10)
    ax.set_xlabel('Level of Finetune Training Data', fontsize=14, labelpad=10)
    
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xticks(x + width, names)
    ax.legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.15), frameon=False,
              handlelength=0.75, handleheight=0.75, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Save figure
    plt.savefig(f'figs/{output_name}.pdf', dpi=300, bbox_inches='tight')

def main():
    # Define paths
    results_dir = 'finetuned_results'
    qwen_rel_paths = [
        'qwen7b_level1.jsonl', 
        'qwen7b_level2.jsonl', 
        'qwen7b_level3.jsonl', 
        'qwen7b_all.jsonl'
    ]
    qwen_result_paths = [os.path.join(results_dir, rel_path) for rel_path in qwen_rel_paths]
    
    # Load data
    data = []
    for path in qwen_result_paths:
        with jsonlines.open(path) as reader:
            data.append([entry for entry in reader])
    
    # Define common parameters
    names = ['level1', 'level2', 'level3', 'all']
    levels = ['level1', 'level2', 'level3']
    
    # Create directory for figures if it doesn't exist
    os.makedirs('figs', exist_ok=True)
    
    # Plot correctness metrics
    plot_correctness(data, names, levels)
    
    # Plot safety metrics
    plot_safety(data, names, levels)

if __name__ == "__main__":
    main()
