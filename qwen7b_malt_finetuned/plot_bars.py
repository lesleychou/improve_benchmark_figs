import jsonlines
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import rcParams

# create 'figs' folder if not exists
if not os.path.exists("figs"):
    os.makedirs("figs")
    
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

def plot_correctness(data, model_names, levels, output_name='qwen7b_performance_comparison'):
    # Set paper style
    set_paper_style()
    
    # Calculate accuracy metrics for each model on each level
    # Reorient data for plotting: test level groups with model bars in each group
    level_results = {level: [] for level in levels}
    level_errors = {level: [] for level in levels}

    for i, dataset in enumerate(data):
        for level in levels:
            num_correct = 0
            total = 0
            binary_outcomes = []
            
            for entry in dataset:
                task = entry['Label']
                correctness = (entry['Result-Correctness'] == 'Pass')
                
                level_indicator = f"level-{level[-1]}"
                if level_indicator in task:
                    num_correct += int(correctness)
                    total += 1
                    binary_outcomes.append(1 if correctness else 0)
            
            # Calculate accuracy and error
            acc = num_correct / total if total > 0 else 0
            level_results[level].append(acc)
            
            # Calculate standard error and multiply by 1.96 for 95% confidence interval
            sem = stats.sem(binary_outcomes, ddof=0) if binary_outcomes else 0
            ci_95 = 1.96 * sem
            level_errors[level].append(ci_95)

    # print("Accuracy:", level_results)
    # print("95% confidence intervals:", level_errors)
    
    # Create visualization
    x = np.arange(len(levels))  # the label locations
    width = 0.15  # the width of the bars

    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(layout='constrained', figsize=(8, 4.5))
    
    # Define colors for better visual distinction between models
    model_colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#ccb974']  # Blue, Green, Red, Purple, Tan
    
    # For each model, create a bar in each level group
    for i, model in enumerate(model_names):
        offset = width * (i - len(model_names)/2 + 0.5)
        model_accuracy = [level_results[level][i] for level in levels]
        model_errors = [level_errors[level][i] for level in levels]
        
        rects = ax.bar(x + offset, model_accuracy, width, label=model, 
                      yerr=model_errors,
                      capsize=5,
                      color=model_colors[i],
                      error_kw={'elinewidth': 0.5, 'capthick': 0.5})
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Correctness Rate', fontsize=14, labelpad=10)
    ax.set_xlabel('Testing Data Level', fontsize=14, labelpad=10)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis limit with a small buffer above the highest value
    max_height = max([max(acc) for acc in level_results.values()]) + 0.1
    ax.set_ylim(0, min(1.0, max_height))

    # Set x-axis ticks and labels
    level_labels = [f"Level {level[-1]}" for level in levels]
    ax.set_xticks(x, level_labels)
    
    # Position legend at top
    ax.legend(loc='upper center', ncols=5, bbox_to_anchor=(0.5, 1.15), frameon=False,
              handlelength=0.75, handleheight=0.75, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save figure
    plt.savefig(f'figs/{output_name}.pdf', dpi=300, bbox_inches='tight')

def plot_safety(data, model_names, levels, output_name='qwen7b_safety_comparison'):
    # Set paper style
    set_paper_style()
    
    # Calculate safety metrics for each model on each level
    # Reorient data for plotting: test level groups with model bars in each group
    level_results = {level: [] for level in levels}
    level_errors = {level: [] for level in levels}

    for i, dataset in enumerate(data):
        for level in levels:
            num_safe = 0
            total = 0
            binary_outcomes = []
            
            for entry in dataset:
                task = entry['Label']
                safety = (entry['Result-Safety'] == 'Pass')
                
                level_indicator = f"level-{level[-1]}"
                if level_indicator in task:
                    num_safe += int(safety)
                    total += 1
                    binary_outcomes.append(1 if safety else 0)
            
            # Calculate safety rate and error
            safety_rate = num_safe / total if total > 0 else 0
            level_results[level].append(safety_rate)
            
            # Calculate standard error and multiply by 1.96 for 95% confidence interval
            sem = stats.sem(binary_outcomes, ddof=0) if binary_outcomes else 0
            ci_95 = 1.96 * sem
            level_errors[level].append(ci_95)

    # print("Safety rates:", level_results)
    # print("Safety 95% confidence intervals:", level_errors)
    
    # Create visualization
    x = np.arange(len(levels))  # the label locations
    width = 0.15  # the width of the bars

    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(layout='constrained', figsize=(8, 4.5))
    
    # Define colors for better visual distinction between models
    model_colors = ['#8dd3c7', '#bebada', '#fb8072', '#80b1d3', '#fdb462']  # Teal, Lavender, Coral, Light blue, Light orange
    
    # For each model, create a bar in each level group
    for i, model in enumerate(model_names):
        offset = width * (i - len(model_names)/2 + 0.5)
        model_safety = [level_results[level][i] for level in levels]
        model_errors = [level_errors[level][i] for level in levels]
        
        rects = ax.bar(x + offset, model_safety, width, label=model, 
                      yerr=model_errors,
                      capsize=5,
                      color=model_colors[i],
                      error_kw={'elinewidth': 0.5, 'capthick': 0.5})
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Safety Rate', fontsize=14, labelpad=10)
    ax.set_xlabel('Testing Data Level', fontsize=14, labelpad=10)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis limit with a small buffer above the highest value
    max_height = max([max(rate) for rate in level_results.values()]) + 0.1
    ax.set_ylim(0, min(1.0, max_height))

    # Set x-axis ticks and labels
    level_labels = [f"Level {level[-1]}" for level in levels]
    ax.set_xticks(x, level_labels)
    
    # Position legend at top
    ax.legend(loc='upper center', ncols=5, bbox_to_anchor=(0.5, 1.15), frameon=False,
              handlelength=0.75, handleheight=0.75, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save figure
    plt.savefig(f'figs/{output_name}.pdf', dpi=300, bbox_inches='tight')

def main():
    # Define paths
    results_dir = 'finetuned_results'
    qwen_rel_paths = [
        'qwen7b_base.jsonl',        # Base model (zero-shot)
        'qwen7b_level1.jsonl',      # Level 1 fine-tuned
        'qwen7b_level2.jsonl',      # Level 2 fine-tuned
        'qwen7b_level3.jsonl',      # Level 3 fine-tuned
        'qwen7b_all.jsonl'          # All levels fine-tuned
    ]
    qwen_result_paths = [os.path.join(results_dir, rel_path) for rel_path in qwen_rel_paths]
    
    # Load data
    data = []
    for path in qwen_result_paths:
        with jsonlines.open(path) as reader:
            data.append([entry for entry in reader])
    
    # Define common parameters
    model_names = ['Base(Zero-shot)', 'SFT-Level 1', 'SFT-Level 2', 'SFT-Level 3', 'SFT-All']
    levels = ['level1', 'level2', 'level3']
    
    # Create directory for figures if it doesn't exist
    os.makedirs('figs', exist_ok=True)
    
    # Plot correctness metrics
    plot_correctness(data, model_names, levels)
    
    # Plot safety metrics
    plot_safety(data, model_names, levels)

if __name__ == "__main__":
    main()
