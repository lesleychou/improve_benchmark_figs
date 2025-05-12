import os
import json
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.patches as mpatches

def static_summarize_results(json_folder, output_file):
    """Summarize results from multiple JSON files in a folder and write to a new JSON file."""
    total_success = 0
    total_iterations = 0
    total_average_time = 0
    total_safe_files = 0
    total_packet_loss_reduction = 0
    total_files_with_reduction = 0

    # Get all JSON file paths in the folder
    json_files = [
        os.path.join(json_folder, f)
        for f in os.listdir(json_folder)
        if f.endswith(".json") and os.path.isfile(os.path.join(json_folder, f))
    ]
    total_files = len(json_files)

    if total_files == 0:
        raise ValueError("No valid JSON files found in the specified folder.")

    for json_path in json_files:
        with open(json_path, "r") as file:
            data = json.load(file)

        # Analyze the last entry for success
        last_entry = data[-1]
        success = 1 if last_entry.get("packet_loss", -1) == 0 else 0

        # Count total successes
        total_success += success

        # Calculate average elapsed time and iterations
        elapsed_times = [entry["elapsed_time"] for entry in data if "elapsed_time" in entry]
        average_time = sum(elapsed_times) / len(elapsed_times)

        # Check if the file is safe (packet_loss is non-increasing)
        packet_losses = [entry["packet_loss"] for entry in data if "packet_loss" in entry]
        is_safe = all(x >= y for x, y in zip(packet_losses, packet_losses[1:]))
        total_safe_files += 1 if is_safe else 0

        # Calculate packet loss reduction
        if len(packet_losses) >= 2:
            initial_packet_loss = packet_losses[0]
            final_packet_loss = packet_losses[-1]
            packet_loss_reduction = initial_packet_loss - final_packet_loss
            total_packet_loss_reduction += packet_loss_reduction
            total_files_with_reduction += 1

        total_average_time += average_time
        total_iterations += len(elapsed_times)

    # Calculate overall metrics
    success_rate = total_success / total_files
    overall_average_time = total_average_time / total_files
    average_iterations = total_iterations / total_files
    safety_rate = total_safe_files / total_files

    # Calculate average packet loss reduction
    average_packet_loss_reduction = (
        total_packet_loss_reduction / total_files_with_reduction
        if total_files_with_reduction > 0 else 0
    )

    # Prepare the summary
    summary = {
        "success_rate": round(success_rate, 2),
        "overall_average_time": round(overall_average_time, 2),
        "average_iterations": round(average_iterations, 2),
        "safety_rate": round(safety_rate, 2),
        "average_packet_loss_reduction": round(average_packet_loss_reduction, 2)
    }

    # Write the summary to the output file
    with open(output_file, "w") as out_file:
        json.dump(summary, out_file, indent=4)


def plot_results(save_result_path, sample_num):
    """
    Plot the success rate and safety rate for each promptagent with publication-ready styling.

    Args:
        save_result_path (str): Root directory path.
        sample_num (int): Number of samples to select for each error type.
    """
    summary_results = {}

    # Iterate through each promptagent folder
    for promptagent in os.listdir(save_result_path):
        promptagent_path = os.path.join(save_result_path, promptagent)
        if not os.path.isdir(promptagent_path):
            continue

        # Load the result JSON file
        result_path = os.path.join(save_result_path, f"{promptagent}.json")
        if not os.path.exists(result_path):
            print(f"Result JSON not found for {promptagent}")
            continue

        with open(result_path, "r") as f:
            results = json.load(f)

        # Group by error type
        errortype_groups = {}
        for result in results:
            errortype = result["detail"]["errortype"]

            # If errortype is a list, convert it to a string or tuple
            if isinstance(errortype, list):
                errortype = tuple(errortype)  # Convert to tuple to ensure hashability

            if errortype not in errortype_groups:
                errortype_groups[errortype] = []
            errortype_groups[errortype].append(result)

        # Select only the top sample_num results for each error type
        filtered_results = []
        for errortype, group in errortype_groups.items():
            filtered_results.extend(group[:sample_num])
        print(f"Filtered results for {promptagent}: {len(filtered_results)} entries")

        # Statistics
        total_queries = len(filtered_results)
        total_success = sum(1 for result in filtered_results if result["success"] == 1)
        total_safe = sum(1 for result in filtered_results if result["safe"] == 1)

        # Calculate success rate and safety rate
        success_rate = (total_success / total_queries) * 100 if total_queries > 0 else 0
        safety_rate = (total_safe / total_queries) * 100 if total_queries > 0 else 0

        # Calculate standard error (SEM)
        success_binary_outcomes = [1] * total_success + [0] * (total_queries - total_success)
        safety_binary_outcomes = [1] * total_safe + [0] * (total_queries - total_safe)

        success_sem = stats.sem(success_binary_outcomes, ddof=0) * 100 if len(success_binary_outcomes) > 1 else 0
        safety_sem = stats.sem(safety_binary_outcomes, ddof=0) * 100 if len(safety_binary_outcomes) > 1 else 0

        # Calculate 95% confidence intervals
        success_margin = 1.96 * success_sem
        safety_margin = 1.96 * safety_sem

        # Save statistics
        summary_results[promptagent] = {
            "success_rate": success_rate,
            "safety_rate": safety_rate,
            "success_margin": success_margin,
            "safety_margin": safety_margin
        }
        print(f"Processed {promptagent}: Success Rate = {success_rate:.2f}%, Safety Rate = {safety_rate:.2f}%,success_margin = {success_margin:.2f}, safety_margin = {safety_margin:.2f}")

    # Create figure with higher DPI and specific size
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # Professional color palette - Option 1: Scientific color scheme
    colors = ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67']

    # Plot each point
    for i, (folder, folder_stats) in enumerate(summary_results.items()):
        x = folder_stats["safety_rate"] / 100
        y = folder_stats["success_rate"] / 100
        x_err = folder_stats["safety_margin"] / 100
        y_err = folder_stats["success_margin"] / 100

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

    # Set labels and title with improved fonts
    ax.set_xlabel("Safety Rate", fontsize=20, fontweight='bold')
    ax.set_ylabel("Success Rate", fontsize=20, fontweight='bold')
    # ax.set_title(f"Success vs. Safety Analysis\n(Top {sample_num} samples per error type)",
    #              fontsize=20,
    #              fontweight='bold',
    #              pad=20)

    # Set axis ranges with padding
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Add legend with improved styling
    legend = ax.legend(loc='upper left',
                      fontsize=20,
                      frameon=True,
                      fancybox=False,
                      edgecolor='black')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the chart with high quality
    output_image_path = os.path.join("figs", f"summary_plot_top_{sample_num}.png")
    plt.savefig(output_image_path,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

    print(f"Plot saved to {output_image_path}")


def plot_spider_charts(save_result_path, sample_num):
    """
    Create spider/radar charts for safety and success rates based on error types,
    with different agents plotted on the same chart for comparison.

    Args:
        save_result_path (str): Root directory path.
        sample_num (int): Number of samples to select for each error type.
    """


    # Error type abbreviation mapping
    error_abbrev = {
        "disable_routing": "level-1\nDR",
        "disable_interface": "level-1\nDI",
        "remove_ip": "level-1\nRI",
        "drop_traffic_to_from_subnet": "level-1\nDT",
        "wrong_routing_table": "level-1\nWR",
        # For combined error types
        "disable_routing+disable_interface": "level-2\nDR+DI",
        "disable_routing+remove_ip": "level-2\nDR+RI",
        "disable_routing+drop_traffic_to_from_subnet": "level-3\nDR+DT",
        "disable_routing+wrong_routing_table": "level-2\nDR+WR",
        "disable_interface+remove_ip": "level-3\nDI+RI",
        "disable_interface+drop_traffic_to_from_subnet": "level-2\nDI+DT",
        "disable_interface+wrong_routing_table": "level-3\nDI+WR",
        "remove_ip+drop_traffic_to_from_subnet": "level-3\nRI+DT",
        "remove_ip+wrong_routing_table": "level-2\nRI+WR",
        "drop_traffic_to_from_subnet+wrong_routing_table": "level-2\nDT+WR"
    }

    # Set global plotting style
    # plt.style.use('seaborn-v0_8-white')  # Clean, professional base style
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

    # Process each promptagent folder
    for promptagent in os.listdir(save_result_path):
        promptagent_path = os.path.join(save_result_path, promptagent)
        if not os.path.isdir(promptagent_path):
            continue

        # Load the result JSON file
        result_path = os.path.join(save_result_path, f"{promptagent}.json")
        if not os.path.exists(result_path):
            print(f"Result JSON not found for {promptagent}")
            continue

        with open(result_path, "r") as f:
            results = json.load(f)

        # Initialize agent results
        if promptagent not in agent_results:
            agent_results[promptagent] = {}

        # Group results by error type for this agent
        for result in results:
            errortype = result["detail"]["errortype"]
            if isinstance(errortype, list):
                errortype = "+".join(errortype)  # Convert list to string

            if errortype not in agent_results[promptagent]:
                agent_results[promptagent][errortype] = {
                    "success": [],
                    "safety": [],
                    "iterations": []
                }

            agent_results[promptagent][errortype]["success"].append(result["success"])
            agent_results[promptagent][errortype]["safety"].append(result["safe"])

            # Calculate iterations from packet loss data
            if "packet_loss" in result:
                iterations = len(result["packet_loss"])
                agent_results[promptagent][errortype]["iterations"].append(iterations)

    # Get all unique error types
    all_error_types = set()
    for agent_data in agent_results.values():
        all_error_types.update(agent_data.keys())
    categories = sorted(list(all_error_types))

    # Create abbreviated category labels
    category_labels = [error_abbrev.get(cat, cat) for cat in categories]

    # Create spider charts for success rate, safety rate, and iterations
    for metric in ["Success Rate", "Safety Rate", "Iterations"]:
        # Number of variables
        N = len(categories)

        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Create the plot with specific figure size for paper
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

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

        # Set y-limits and ticks based on metric
        if metric == "Iterations":
            # Find the max average iterations to set scale appropriately
            max_iterations = 0
            for agent_data in agent_results.values():
                for error_data in agent_data.values():
                    if error_data.get("iterations"):
                        max_iterations = max(max_iterations, max(5, np.mean(error_data["iterations"])))

            # Round up to the nearest 5 for a cleaner scale
            max_y = min(30, max(10, 5 * np.ceil(max_iterations / 5)))
            tick_step = max(1, int(max_y / 5))
            ticks = [i * tick_step for i in range(6)]

            ax.set_ylim(0, max_y)
            plt.yticks(ticks, [str(t) if t != max_y else "" for t in ticks], color="black", ha='right', va='bottom')

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
        if metric == "Iterations":
            grid_values = ticks
        else:
            grid_values = [20, 40, 60, 80, 100]

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
                        rate = np.mean(agent_data[errortype]["success"]) * 100
                    elif metric == "Safety Rate":
                        rate = np.mean(agent_data[errortype]["safety"]) * 100
                    else:  # Iterations
                        if agent_data[errortype].get("iterations"):
                            rate = np.mean(agent_data[errortype]["iterations"])
                        else:
                            rate = 0
                else:
                    rate = 0
                rates.append(rate)

            # Close the plot by appending the first value
            values = np.concatenate((rates, [rates[0]]))

            color = colors[idx % len(colors)]
            # Plot line with higher z-order to ensure it's above the fill
            ax.plot(angles, values, linewidth=1.5, linestyle='-', color=color, zorder=2, clip_on=False)
            # ax.fill(angles, values, color=color, alpha=0.05, zorder=1)

            legend_patches.append(mpatches.Patch(color=color, label=agent))

        # Add legend with improved positioning and styling
        # legend = plt.legend(handles=legend_patches,
        #                   loc='lower left',
        #                   frameon=True,
        #                   edgecolor='none',
        #                   facecolor='white',
        #                   framealpha=0.8)

        # Adjust layout to prevent text cutoff
        plt.tight_layout()

        # Save figure with higher quality settings
        metric_filename = metric.lower().replace(' ', '_')
        output_path = os.path.join("figs", f"route_spider_chart_{metric_filename}_by_agent")

        # # Save as PDF for better quality
        # plt.savefig(f"{output_path}.pdf",
        #             dpi=300,
        #             bbox_inches='tight',
        #             pad_inches=0.2)

        # Also save as PNG for easy viewing
        plt.savefig(f"{output_path}.png",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.2)

        plt.close()


        # Create the legend as a separate figure
        # Shared legend between all spider charts
        legend_path = os.path.join("figs", f"route_spider_chart_legend")

        legend_fig, legend_ax = plt.subplots(figsize=(8, 2), dpi=300)
        legend_ax.axis('off')
        legend_ax.legend(handles=legend_patches,
                         loc='center',
                         frameon=False,
                         edgecolor='none',
                         facecolor='white',
                         ncol=len(legend_patches)
                         )
        legend_fig.tight_layout()

        # # Save as PDF for better quality
        # plt.savefig(f"{legend_path}.pdf",
        #             dpi=300,
        #             bbox_inches='tight',
        #             pad_inches=0.2)

        legend_fig.savefig(f"{legend_path}.png",
                           dpi=300,
                           bbox_inches='tight',
                           pad_inches=0.2)

        print(f"Spider chart for {metric} by agent saved to {output_path}.png")

    # Print the abbreviation mapping for reference
    print("\nError Type Abbreviations:")
    for cat, abbrev in zip(categories, category_labels):
        print(f"{abbrev}: {cat}")

if __name__ == "__main__":
    save_result_path = "all_agents"

    # plot_results(save_result_path, 10)
    # plot_results(save_result_path, 150)
    plot_spider_charts(save_result_path, 150)
