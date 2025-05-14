import json
import os
import itertools
import matplotlib.pyplot as plt
from scipy import stats
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as mpatches

def summary_different_agent(directory, number_query):
    summary_results = {}

    # Define the 15 error types
    error_types = [
        "remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress",
        "remove_ingress+add_ingress", "remove_ingress+change_port", "remove_ingress+change_protocol",
        "add_ingress+change_port", "add_ingress+change_protocol", "change_port+change_protocol",
        "change_port+add_egress", "change_protocol+add_egress", "remove_ingress+add_egress",
        "add_ingress+add_egress"
    ]
    # Load error_config.json
    error_config_path = os.path.join(directory, "error_config.json")
    if not os.path.exists(error_config_path):
        raise FileNotFoundError(f"error_config.json not found in {directory}")
    
    with open(error_config_path, "r") as config_file:
        error_config = json.load(config_file)

    details = error_config["details"]

    # Iterate through all folders in the given directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Initialize counters
            total_queries = 0
            total_success = 0
            total_safety = 0
            success_rates = []
            safety_rates = []

            # Process each error type
            for error_type in error_types:
                # Collect all matching JSON files for the current error type
                matching_files = [
                    f for f in os.listdir(folder_path)
                    if f.startswith(f"{error_type}_result_") and f.endswith(".json")
                ]

                # Extract and sort files by their numeric index
                indexed_files = []
                for file_name in matching_files:
                    # Use re.escape to handle special characters in error_type
                    match = re.search(rf"{re.escape(error_type)}_result_(\d+)\.json$", file_name)
                    if match:
                        index = int(match.group(1))
                        indexed_files.append((index, file_name))
                indexed_files.sort(key=lambda x: x[0])  # Sort by index

                # Select the first `number_query` files based on their index
                selected_files = [file_name for _, file_name in indexed_files[:number_query]]

                # Process each selected file
                for file_name in selected_files:
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as file:
                        data = json.load(file)

                        # Update counters
                        total_queries += 1
                        success = 1
                        if "No mismatches found" in data[-1].get("mismatch_summary", ""):
                            total_success += 1
                        else:
                            success = 0

                        # Check safety
                        safe = True
                        previous_mismatch_count = float('inf')
                        for entry in data:
                            mismatch_summary = entry.get("mismatch_summary", "")
                            mismatch_count = mismatch_summary.count("Mismatch")
                            if mismatch_count > previous_mismatch_count:
                                safe = False
                                break
                            previous_mismatch_count = mismatch_count
                        if safe:
                            total_safety += 1

                        # Create binary lists for success and safety counts
                        success_binary_outcomes = [1] * total_success + [0] * (total_queries - total_success)
                        safety_binary_outcomes = [1] * total_safety + [0] * (total_queries - total_safety)

                        # Calculate standard error of mean (SEM) for success rates
                        if len(success_binary_outcomes) > 1:
                            success_rates.append(stats.sem(success_binary_outcomes, ddof=0) * 100)

                        # Calculate SEM for safety rates
                        if len(safety_binary_outcomes) > 1:
                            safety_rates.append(stats.sem(safety_binary_outcomes, ddof=0) * 100)

            # Compute 95% confidence interval (1.96 * SEM) for percentages
            success_margin = 1.96 * (sum(success_rates) / len(success_rates)) if success_rates else 0
            safety_margin = 1.96 * (sum(safety_rates) / len(safety_rates)) if safety_rates else 0

            # Store results for each experiment folder
            summary_results[folder] = {
                "total_queries": total_queries,
                "success_rate": (total_success / total_queries) * 100 if total_queries > 0 else 0,
                "safety_rate": (total_safety / total_queries) * 100 if total_queries > 0 else 0,
                "success_margin": success_margin,
                "safety_margin": safety_margin
            }

    # Print and return the summary results
    print(json.dumps(summary_results, indent=4))
    return summary_results

def plot_summary_results(directory_path, number_query):
    """
    Reads experiment results from multiple folders, plots success vs. safety,
    and saves the figure inside the directory.

    Parameters:
        directory_path (str): Path to the directory containing experiment folders.
        number_query (int): Number of queries used for each error type.

    Saves:
        summary_plot_{number_query}.png inside directory_path.
    """
    # Get summary results
    summary_results = summary_different_agent(directory_path, number_query)

    # Set font family to Arial
    plt.rcParams['font.family'] = 'Arial'

    # Create figure with higher DPI and specific size
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    
    # Professional color palette - Scientific color scheme
    colors = ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67']

    # Iterate through each folder's summary and plot points
    for i, (folder, stats) in enumerate(summary_results.items()):
        x = stats["safety_rate"] / 100  # X-axis: Safety rate (converted to 0-1 scale)
        y = stats["success_rate"] / 100  # Y-axis: Success rate (converted to 0-1 scale)
        x_err = stats["safety_margin"] / 100  # Error bar for safety rate (converted to 0-1 scale)
        y_err = stats["success_margin"] / 100  # Error bar for success rate (converted to 0-1 scale)

        # Plot points and error bars with improved styling
        ax.errorbar(x, y, 
                   xerr=x_err, 
                   yerr=y_err,
                   fmt='o',
                   color=colors[i % len(colors)],
                   markersize=8,
                   markeredgewidth=1.5,
                   markeredgecolor='white',
                   capsize=5,
                   capthick=1.5,
                   elinewidth=1.5,
                   label=folder)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3, which='major')
    ax.set_axisbelow(True)  # Place grid behind points
    
    # Set labels with improved fonts
    ax.set_xlabel("Safety Rate", fontsize=20, fontweight='normal', labelpad=10)
    ax.set_ylabel("Success Rate", fontsize=20, fontweight='normal', labelpad=10)

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
    
    # Get legend handles for separate legend
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    
    # Save the chart with high quality
    save_path = os.path.join("figs", f"k8s_summary_plot_{number_query}.pdf")
    plt.savefig(save_path, 
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)
    
    plt.close()
    
    # Create separate legend figure
    legend_path = os.path.join("figs", f"k8s_summary_plot_legend_{number_query}")
    legend_fig, legend_ax = plt.subplots(figsize=(8, 2), dpi=300)
    legend_ax.axis('off')
    
    # Add legend with improved styling
    legend_ax.legend(legend_handles, legend_labels,
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
    
    print(f"Plot saved at: {save_path}")


def plot_spider_charts_for_agents(save_result_path, number_query):
    """
    Create separate spider charts for safety, success, and iteration rates by error type,
    comparing results from multiple agents.

    Args:
        save_result_path (str): Root directory path containing agent result JSON files.
        number_query (int): Number of queries to analyze and plot for each error type.
    """
    # Error type abbreviation mapping - you can customize this for the k8s context
    # Example mapping based on common error types in k8s
    error_abbrev = {
        "remove_ingress": "level-1\nRI",
        "add_ingress": "level-1\nAI",
        "change_port": "level-1\nCP",
        "change_protocol": "level-1\nCPR",
        "add_egress": "level-1\nAE",
        "remove_ingress+add_ingress": "level-2\nRI+AI",
        "remove_ingress+change_port": "level-2\nRI+CP",
        "remove_ingress+change_protocol": "level-2\nRI+CPR",
        "add_ingress+change_port": "level-2\nAI+CP",
        "add_ingress+change_protocol": "level-2\nAI+CPR",
        "change_port+change_protocol": "level-2\nCP+CPR",
        "change_port+add_egress": "level-3\nCP+AE",
        "change_protocol+add_egress": "level-3\nCPR+AE",
        "remove_ingress+add_egress": "level-3\nRI+AE",
        "add_ingress+add_egress": "level-3\nAI+AE"
    }
    
    # Set global plotting style
    plt.rcParams.update({
        'font.size': 16,                   # Base font size
        'axes.labelsize': 16,              # Size for axis labels
        'axes.titlesize': 16,              # Size for subplot titles
        'figure.titlesize': 16,            # Size for figure titles
        'legend.fontsize': 16,             # Size for legend text
        'xtick.labelsize': 16,             # Size for x-tick labels
        'ytick.labelsize': 10,             # Size for y-tick labels
    })
    
    # Set font family to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Dictionary to store results by agent and error type
    agent_results = {}
    
    # Professional color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf']  # Professional color scheme

    # Iterate through each agent directory
    for agent in os.listdir(save_result_path):
        agent_path = os.path.join(save_result_path, agent)
        if not os.path.isdir(agent_path):
            continue

        # Load the test results summary JSON file for this agent
        result_file = os.path.join(agent_path, "test_results_summary.json")
        if not os.path.exists(result_file):
            print(f"Result JSON not found for {agent}")
            continue

        with open(result_file, "r") as f:
            results = json.load(f)
            
        # Initialize agent results
        if agent not in agent_results:
            agent_results[agent] = {}

        # Collect success, safety, and iteration metrics for each error type
        for error_type, error_data in results.items():
            if error_type not in agent_results[agent]:
                agent_results[agent][error_type] = {
                    "success": [],
                    "safety": [],
                    "iteration": []
                }
            
            # Extract metrics
            success_rate = error_data["successful_rate"]
            safety_rate = error_data["safety_rate"]
            # Assuming there's an average_iteration field, else use a default
            avg_iteration = error_data.get("average_iteration", 0)
            
            # Normalize iteration to 0-160 scale (assuming max of 16 iterations would be 160%)
            # You may need to adjust this scaling based on your actual iteration ranges
            normalized_iteration = min(avg_iteration * 10, 160)
            
            # Store the metrics
            agent_results[agent][error_type]["success"].append(success_rate * 100)      # Convert to percentage 
            agent_results[agent][error_type]["safety"].append(safety_rate * 100)        # Convert to percentage
            agent_results[agent][error_type]["iteration"].append(normalized_iteration)  # Already normalized

    # Get all unique error types
    all_error_types = set()
    for agent_data in agent_results.values():
        all_error_types.update(agent_data.keys())
    
    # Sort categories by level 
    def get_level(error_type):
        if error_type in error_abbrev:
            abbrev = error_abbrev[error_type]
            if "level-1" in abbrev:
                return 1
            elif "level-2" in abbrev:
                return 2
            elif "level-3" in abbrev:
                return 3
        return 4  # For any unknown categories
    
    categories = sorted(list(all_error_types), key=get_level)
    
    # Create abbreviated category labels
    category_labels = [error_abbrev.get(cat, cat) for cat in categories]
    
    # Create three separate spider charts (success, safety, and iteration)
    for metric in ["Success Rate", "Safety Rate", "Iteration"]:
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the plot with specific figure size for paper
        fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(projection='polar'))
        
        # Set the category labels with consistent formatting
        plt.xticks(angles[:-1], category_labels, fontsize=12, va='center')
        
        # Set the alignments of the labels
        for angle, label in zip(angles[:-1], ax.get_xticklabels()):
            if angle in (0, np.pi):
                label.set_horizontalalignment('left')
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
        
        # Set y-limits and ticks
        if metric == "Iteration":
            # Use the original scale for iteration
            ax.set_ylim(0, 160)
            plt.yticks([0, 40, 80, 120, 160], ["0", "4", "8", "12", "16"], color="black", ha='right', va='bottom')
            
            # Set radial axis label position
            ax.set_rlabel_position(angles[1] * 180 / np.pi)
        else:
            ax.set_ylim(0, 100)
            plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", ""], color="black", ha='left', va='bottom')

            # Set radial axis label position
            ax.set_rlabel_position(angles[1] * 180 / np.pi)
        
        # Remove the circular grid and spines
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        
        # Draw polygon grid lines with more professional styling
        grid_values = [20, 40, 60, 80, 100] if metric != "Iteration" else [20, 40, 80, 120, 160]
            
        for i, grid_val in enumerate(grid_values):
            alpha = 1 if i == len(grid_values) - 1 else 0.15
            polygon_points = [(a, grid_val) for a in angles]
            ax.plot([p[0] for p in polygon_points], [p[1] for p in polygon_points], 
                    '-', color='gray' if alpha != 1 else 'black', alpha=alpha, linewidth=0.7,
                    clip_on=False)
        
        # Draw axis lines with consistent styling
        for i in range(N):
            ax.plot([angles[i], angles[i]], [0, ax.get_ylim()[1]], 
                    color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=-10)
        
        # Plot each agent with improved styling
        legend_patches = []
        for idx, (agent, agent_data) in enumerate(agent_results.items()):
            rates = []
            for errortype in categories:
                if errortype in agent_data:
                    if metric == "Success Rate":
                        rate = np.mean(agent_data[errortype]["success"])
                    elif metric == "Safety Rate":
                        rate = np.mean(agent_data[errortype]["safety"])
                    else:  # Iteration
                        rate = np.mean(agent_data[errortype]["iteration"])
                else:
                    rate = 0
                rates.append(rate)
            
            # Close the plot by appending the first value
            values = np.concatenate((rates, [rates[0]]))
            
            color = colors[idx % len(colors)]
            # Plot line with higher z-order to ensure it's above the fill
            ax.plot(angles, values, linewidth=1.5, linestyle='-', color=color, zorder=2, clip_on=False)
            # Lighter fill
            # ax.fill(angles, values, color=color, alpha=0.05, zorder=1)
            
            legend_patches.append(mpatches.Patch(color=color, label=agent))
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        
        # Save figure with higher quality settings
        metric_name = metric.lower().replace(' ', '_')
        output_path = os.path.join("figs", f"k8s_spider_chart_{metric_name}_by_agent")
        
        # Save as PDF with high quality
        plt.savefig(f"{output_path}.pdf",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.2)
        
        plt.close()
        
        # Create the legend as a separate figure
        legend_path = os.path.join("figs", f"k8s_spider_chart_legend")
        
        legend_fig, legend_ax = plt.subplots(figsize=(8, 2), dpi=300)
        legend_ax.axis('off')
        legend_ax.legend(handles=legend_patches,
                       loc='center',
                       frameon=False,
                       edgecolor='none',
                       facecolor='white',
                       ncol=len(legend_patches))
        legend_fig.tight_layout()
        
        # Save legend as PDF
        legend_fig.savefig(f"{legend_path}.pdf",
                         dpi=300,
                         bbox_inches='tight',
                         pad_inches=0.2)
        
        print(f"Spider chart for {metric} by agent saved to {output_path}.pdf")
        
    # Print the abbreviation mapping for reference once at the end
    print("\nError Type Abbreviations:")
    for cat, abbrev in zip(categories, category_labels):
        print(f"{abbrev}: {cat}")

if __name__ == "__main__":
    plot_spider_charts_for_agents("20250426_045818", 150)
    
    plot_summary_results("20250426_045818", 10)
    plot_summary_results("20250426_045818", 150)
    
    # Crop all PDF files to remove margins
    for file in os.listdir("figs"):
        if file.endswith(".pdf"):
            os.system(f"pdfcrop --margins 0 {os.path.join('figs', file)} {os.path.join('figs', file)}")