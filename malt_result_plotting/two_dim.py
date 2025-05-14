import numpy as np
import pandas as pd
import jsonlines
import random
import os
import time
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches

import argparse
from scipy import stats
import math

# Create a consistent color mapping for all agent methods
AGENT_COLOR_MAP = {
    "GPT+CoT": "#1f77b4",        # Blue
    "GPT+Fewshot": "#ff7f0e",    # Orange
    "QWen+CoT": "#2ca02c",       # Green
    "QWen+Fewshot": "#d62728",   # Red
    "GPT+ReAct": "#9467bd"       # Purple
}

# Mapping for agent names in the legend
AGENT_NAME_MAPPING = {
    "react_GPT": "GPT+ReAct",
    "few_shot_basic_GPT": "GPT+Fewshot",
    "cot_Qwen": "QWen+CoT",
    "few_shot_Qwen": "QWen+Fewshot",
    "cot_GPT": "GPT+CoT",
    "few_shot_GPT": "GPT+Fewshot",  # Add mapped name for few_shot_GPT
}

def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic error detection performance')
    parser.add_argument('--sampling_method', type=str, choices=['first', 'random'], default='first',
                        help='Method to sample: "first" takes first N samples, "random" takes random N samples')
    parser.add_argument('--separate_legend', action='store_true',
                        help='Create a separate legend file')
    args = parser.parse_args()

    # Set font family to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Set global plotting style
    plt.rcParams.update({
        'font.size': 16,               # Base font size
        'axes.labelsize': 20,          # Size for axis labels
        'axes.titlesize': 20,          # Size for subplot titles
        'figure.titlesize': 20,        # Size for figure titles
        'legend.fontsize': 16,         # Size for legend text
        'xtick.labelsize': 18,         # Size for x-tick labels
        'ytick.labelsize': 18,         # Size for y-tick labels
    })

    # Use merged files directly, like in spider_chart.py
    input_path = [
        "all_agents/cot_GPT_merged.jsonl",
        "all_agents/few_shot_GPT_merged.jsonl",
        "all_agents/cot_Qwen_merged.jsonl",
        "all_agents/few_shot_Qwen_merged.jsonl",
        "all_agents/react_GPT_merged.jsonl",
    ]
    
    # Load data from each file
    all_results = []
    for path in input_path:
        print(f"Processing file: {path}")
        results = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                results.append(obj)
        
        # Extract name from the filename (remove path and extension)
        name = os.path.basename(path).replace("_merged.jsonl", "")
        
        # Map the agent name to standardized naming
        display_name = AGENT_NAME_MAPPING.get(name, name)
        
        # Group the results by task label
        grouped_results = {}
        for result in results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': path,
            'name': name,
            'display_name': display_name,
            'grouped_results': grouped_results
        })

    sample_sizes = [10, 1000]
    
    for sample_size in sample_sizes:
        print(f"\n=== STATISTICS FOR {args.sampling_method.upper()} {sample_size} SAMPLES PER LABEL ===")
        
        safety_data = []
        correctness_data = []
        
        for result_set in all_results:
            grouped_results = result_set['grouped_results']
            name = result_set['name']
            display_name = result_set['display_name']
            
            # Calculate stats for each task label
            task_labels = list(grouped_results.keys())
            
            # Correctness stats
            print(f"\nCorrectness Pass Rates for {display_name}:")
            
            # Calculate overall statistics for correctness
            all_binary_outcomes = []
            for task_label in task_labels:
                if task_label == 'None':  # Skip the 'None' label if it exists
                    continue
                
                # Sample based on the specified method
                if args.sampling_method == 'first':
                    samples = grouped_results[task_label][:sample_size]
                else:  # random sampling
                    all_samples = grouped_results[task_label]
                    # If we have enough samples, take a random sample, otherwise take all
                    if len(all_samples) > sample_size:
                        samples = random.sample(all_samples, sample_size)
                    else:
                        samples = all_samples
                        
                binary_outcomes = [1 if result["Result-Correctness"] == "Pass" else 0 for result in samples]
                all_binary_outcomes.extend(binary_outcomes)
            
            # Calculate overall pass rate and error margin for correctness
            total_samples = len(all_binary_outcomes)
            correctness_pass_rate = (sum(all_binary_outcomes) / total_samples) * 100
            correctness_sem = stats.sem(all_binary_outcomes, ddof=0) * 100
            correctness_error_margin = 1.96 * correctness_sem
            
            print(f"Overall Correctness: {correctness_pass_rate:.2f}% ±{correctness_error_margin:.2f}% (n={total_samples})")

            # Safety stats
            print(f"\nSafety Pass Rates for {display_name}:")
            
            # Calculate overall safety statistics
            all_binary_outcomes = []
            for task_label in task_labels:
                if task_label == 'None':  # Skip the 'None' label if it exists
                    continue
                
                # Sample based on the specified method
                if args.sampling_method == 'first':
                    samples = grouped_results[task_label][:sample_size]
                else:  # random sampling
                    all_samples = grouped_results[task_label]
                    # If we have enough samples, take a random sample, otherwise take all
                    if len(all_samples) > sample_size:
                        samples = random.sample(all_samples, sample_size)
                    else:
                        samples = all_samples
                        
                binary_outcomes = [1 if result["Result-Safety"] == "Pass" else 0 for result in samples]
                all_binary_outcomes.extend(binary_outcomes)
            
            total_samples = len(all_binary_outcomes)
            safety_pass_rate = (sum(all_binary_outcomes) / total_samples) * 100
            safety_sem = stats.sem(all_binary_outcomes, ddof=0) * 100
            safety_error_margin = 1.96 * safety_sem
            
            print(f"Overall Safety: {safety_pass_rate:.2f}% ±{safety_error_margin:.2f}% (n={total_samples})")
            
            # Store data for plotting
            safety_data.append({
                'name': display_name,
                'pass_rate': safety_pass_rate,
                'error_margin': safety_error_margin
            })
            
            correctness_data.append({
                'name': display_name,
                'pass_rate': correctness_pass_rate,
                'error_margin': correctness_error_margin
            })
        
        # Create a new figure for each sample size with professional styling
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        
        # Plot the scatter points with error bars
        legend_handles = []
        for j in range(len(safety_data)):
            s_data = safety_data[j]
            c_data = correctness_data[j]
            
            # Get color from the consistent color map
            color = AGENT_COLOR_MAP.get(s_data['name'], "#333333")  # Default to dark gray if not found
            
            ax.errorbar(
                s_data['pass_rate'] / 100,  # Convert to decimal
                c_data['pass_rate'] / 100,  # Convert to decimal
                xerr=s_data['error_margin'] / 100,  # Convert to decimal
                yerr=c_data['error_margin'] / 100,  # Convert to decimal
                fmt='o',
                color=color,
                markersize=8,
                markeredgewidth=1.5,
                markeredgecolor='white',
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                label=s_data['name']
            )
            
            # Add to legend handles
            legend_handles.append(mpatches.Patch(color=color, label=s_data['name']))

        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.3, which='major')
        ax.set_axisbelow(True)  # Place grid behind points

        # Set labels with improved fonts
        ax.set_xlabel("Safety Rate", fontsize=20, fontweight='normal', labelpad=10)
        ax.set_ylabel("Correctness Rate", fontsize=20, fontweight='normal', labelpad=10)

        # Set axis ranges with padding
        ax.set_xlim(-0.0, 1.00)
        ax.set_ylim(-0.0, 1.00)
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save file path
        save_path = f'figs/malt_sampling_{args.sampling_method}_N{sample_size}.pdf'
        
        # # If not creating separate legend, add legend to main plot
        # if not args.separate_legend:
        #     # Add legend with improved styling
        #     legend = ax.legend(loc='lower right',
        #                     frameon=True,
        #                     fancybox=False,
        #                     edgecolor='black',
        #                     fontsize=16)

        # Save individual figure
        plt.savefig(save_path, 
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.2)
        plt.close()
        
        # Create separate legend if requested
        if args.separate_legend:
            legend_path = f'figs/malt_sampling_{args.sampling_method}_legend_N{sample_size}'
            legend_fig, legend_ax = plt.subplots(figsize=(8, 2), dpi=300)
            legend_ax.axis('off')
            
            # Add legend with improved styling
            legend_ax.legend(handles=legend_handles,
                           loc='center',
                           ncol=len(legend_handles),
                           fontsize=20,
                           frameon=False,
                           fancybox=False,
                           edgecolor='black')
            
            legend_fig.tight_layout()
            legend_fig.savefig(f"{legend_path}.pdf",
                             dpi=300,
                             bbox_inches='tight',
                             pad_inches=0.2)
            plt.close(legend_fig)

    print(f"Individual figures saved as figs/malt_sampling_{args.sampling_method}_N[sample_size].pdf")
    
    # For all pdf/png files under "figs" folder, crop the margins to 0
    for file in os.listdir("figs"):
        if file.endswith(".pdf"):
            os.system(f"pdfcrop --margins 0 {os.path.join('figs', file)} {os.path.join('figs', file)}")

# run the main function
if __name__ == "__main__":
    main()