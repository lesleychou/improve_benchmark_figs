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

# create 'figs' folder if not exists
if not os.path.exists("figs"):
    os.makedirs("figs")
    
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

# Example usage:
# python eval_with_spider_charts.py --sampling_method random

def main():
    parser = argparse.ArgumentParser(description='Generate spider charts for semantic error detection performance')
    parser.add_argument('--sampling_method', type=str, choices=['first', 'random'], default='first',
                        help='Method to sample: "first" takes first N samples, "random" takes random N samples')
    args = parser.parse_args()

    # Set global plotting style
    plt.style.use('seaborn-v0_8-white')  # Clean, professional base style
    plt.rcParams.update({
        'font.size': 16,                   # Base font size
        'axes.labelsize': 16,              # Size for axis labels
        'axes.titlesize': 16,              # Size for subplot titles
        'figure.titlesize': 16,            # Size for figure titles
        'legend.fontsize': 16,             # Size for legend text
        'xtick.labelsize': 16,             # Size for x-tick labels
        'ytick.labelsize': 10,             # Size for y-tick labels
    })

    # Use merged files directly
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

    # Use a fixed sample size for the spider charts
    sample_size = 100
    
    # Set up data structures for spider charts
    agent_display_names = [result_set['display_name'] for result_set in all_results]
    # Get all task labels excluding the first one (which is typically 'None' or similar)
    all_task_labels = set()
    for result_set in all_results:
        for label in result_set['grouped_results'].keys():
            if label != 'None':  # Skip the 'None' label if it exists
                # Remove "capacity planning, " from the label
                cleaned_label = label.replace("capacity planning, ", "")
                all_task_labels.add(cleaned_label)
    task_labels = sorted(list(all_task_labels))

    # Prepare data for spider charts
    correctness_data = {agent: {} for agent in agent_display_names}
    safety_data = {agent: {} for agent in agent_display_names}
    latency_data = {agent: {} for agent in agent_display_names}
    
    # Calculate stats for each agent and task label
    for i, result_set in enumerate(all_results):
        agent_display_name = result_set['display_name']
        grouped_results = result_set['grouped_results']
        
        for task_label in task_labels:
            # Add back "capacity planning, " to match the original label in the data
            original_label = f"capacity planning, {task_label}"
            if original_label in grouped_results:
                # Sample based on the specified method
                if args.sampling_method == 'first':
                    samples = grouped_results[original_label][:sample_size]
                else:  # random sampling
                    all_samples = grouped_results[original_label]
                    # If we have enough samples, take a random sample, otherwise take all
                    if len(all_samples) > sample_size:
                        samples = random.sample(all_samples, sample_size)
                    else:
                        samples = all_samples
                
                # Calculate correctness pass rate
                correctness_binary = [1 if result["Result-Correctness"] == "Pass" else 0 for result in samples]
                correctness_pass_rate = (sum(correctness_binary) / len(correctness_binary)) * 100
                correctness_data[agent_display_name][task_label] = correctness_pass_rate
                
                # Calculate safety pass rate
                safety_binary = [1 if result["Result-Safety"] == "Pass" else 0 for result in samples]
                safety_pass_rate = (sum(safety_binary) / len(safety_binary)) * 100
                safety_data[agent_display_name][task_label] = safety_pass_rate

                # Calculate average latency
                latencies = [float(result["Result-Latency"]) for result in samples]
                avg_latency = sum(latencies) / len(latencies)
                latency_data[agent_display_name][task_label] = avg_latency
            else:
                # If no data for this task label, set to 0
                correctness_data[agent_display_name][task_label] = 0
                safety_data[agent_display_name][task_label] = 0
                latency_data[agent_display_name][task_label] = 0
    
    # Create spider charts
    create_spider_chart(correctness_data, task_labels, "Correctness Pass Rate (%)", 
                        f"figs/malt_correctness_spider_merged.pdf")
    create_spider_chart(safety_data, task_labels, "Safety Pass Rate (%)", 
                        f"figs/malt_safety_spider_merged.pdf")
    create_spider_chart(latency_data, task_labels, "Average Latency (seconds)", 
                        f"figs/malt_latency_spider_merged.pdf",
                        is_latency=True)

    print(f"Spider charts saved to figs/malt_correctness_spider_merged.pdf, figs/malt_safety_spider_merged.pdf, and figs/malt_latency_spider_merged.pdf")

    # for all pdf file under "figs" folder, crop the margins to 0
    for file in os.listdir("figs"):
        if file.endswith(".pdf"):
            os.system(f"pdfcrop --margins 0 {os.path.join('figs', file)} {os.path.join('figs', file)}")


def create_spider_chart(data, categories, title, output_path, is_latency=False):
    """Create a polygon-style spider chart with the given data."""
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot with specific figure size for paper
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # Set font family to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Set the category labels with consistent formatting
    formatted_categories = [cat.replace(',', '\n') for cat in categories]
    plt.xticks(angles[:-1], formatted_categories)
    
    # Set consistent text alignment for the category labels
    for angle, label in zip(angles[:-1], ax.get_xticklabels()):
        if angle == 0:
            label.set_horizontalalignment('left')
        elif angle == np.pi:
            label.set_horizontalalignment('right')
        elif angle in (np.pi / 2, 3 * np.pi / 2):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi / 2:
            label.set_horizontalalignment('left')
        elif np.pi / 2 < angle < np.pi:
            label.set_horizontalalignment('right')
        elif np.pi < angle < 3 * np.pi / 2:
            label.set_horizontalalignment('right')
        else:
            label.set_horizontalalignment('left')
    
    # Set y-limits and ticks based on data type with consistent formatting
    if is_latency:
        # Set fixed max value at 25 for consistent display
        y_max = 25
        # Create 6 ticks from 0 to 25 (inclusive)
        y_ticks = np.linspace(0, y_max, 6)  # Will create [0, 5, 10, 15, 20, 25]
        # Set y-limits to match the max value
        ax.set_ylim(0, y_max)
        # Display the tick labels with consistent formatting
        plt.yticks(y_ticks, [f"{tick:.1f}s" for tick in y_ticks], color="black")
    else:
        ax.set_ylim(0, 100)
        plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="black")
    
    # Set radial axis label position
    ax.set_rlabel_position(0)
    
    # Remove the circular grid and spines
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    
    # Draw polygon grid lines with more professional styling
    grid_values = [20, 40, 60, 80, 100] if not is_latency else y_ticks
    for i, grid_val in enumerate(grid_values):
        alpha = 1 if i == len(grid_values) - 1 else 0.15
        polygon_points = [(a, grid_val) for a in angles]
        ax.plot([p[0] for p in polygon_points], [p[1] for p in polygon_points], 
                '-', color='gray' if alpha != 1 else 'black', alpha=alpha, linewidth=0.7, clip_on=False)
    
    # Draw axis lines with consistent styling
    for i in range(N):
        ax.plot([angles[i], angles[i]], [0, ax.get_ylim()[1]], 
                color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=-10)
    
    # Plot each agent with improved styling
    legend_patches = []
    for i, (agent_name, agent_data) in enumerate(data.items()):
        values = [agent_data.get(cat, 0) for cat in categories]
        values += values[:1]  # Close the loop
        
        # Get color from the consistent color map
        color = AGENT_COLOR_MAP.get(agent_name, "#333333")  # Default to dark gray if not found
        
        # Plot line with higher z-order to ensure it's above the fill
        ax.plot(angles, values, linewidth=1.5, linestyle='-', color=color, zorder=2, clip_on=False)
        # Removed the fill line to keep only the outline
        
        legend_patches.append(mpatches.Patch(color=color, label=agent_name))
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save figure with higher quality settings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)

# run the main function
if __name__ == "__main__":
    main()
