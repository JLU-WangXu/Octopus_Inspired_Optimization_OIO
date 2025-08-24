#!/usr/bin/env python3
"""
Algorithm Performance Visualization Script
==========================================
This script generates a professional-quality boxplot to compare the performance
distribution of various optimization algorithms based on experimental data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

def setup_matplotlib_style():
    """Sets a professional and clean style for the plots."""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 300
    plt.style.use('seaborn-v0_8-whitegrid')

def create_performance_boxplot(data: dict, output_dir: Path):
    """
    Generates and saves a performance comparison boxplot.

    Args:
        data (dict): A dictionary containing the performance data for all algorithms.
        output_dir (Path): The directory where the output plot will be saved.
    """
    print("📊 Generating performance box plot...")

    # 1. Extract the necessary data for plotting
    # We need the raw data from all runs for each algorithm to create a boxplot
    all_performance_data = data.get('algorithm_all_data')
    if not all_performance_data:
        print("⚠️  Warning: 'algorithm_all_data' key not found or is empty. Cannot generate plot.")
        return

    # 2. Prepare data and labels for the plot
    # Sort algorithms to ensure a consistent order, placing 'OIO' first for emphasis
    alg_names = sorted([name for name in all_performance_data.keys() if name != 'OIO'])
    if 'OIO' in all_performance_data:
        alg_names.insert(0, 'OIO')
    
    plot_data = [all_performance_data[name] for name in alg_names]

    # 3. Identify the best-performing algorithm to highlight it
    # Based on the data, lower values indicate better performance (e.g., lower error)
    medians = [np.median(d) for d in plot_data]
    best_alg_idx = np.argmin(medians)

    # 4. Create the plot
    fig, ax = plt.subplots(figsize=(16, 9))
    box = ax.boxplot(plot_data, labels=alg_names, patch_artist=True, showmeans=True)

    # 5. Apply professional styling
    # Highlight the best-performing algorithm with a distinct color
    colors = ['#FF6B35' if i == best_alg_idx else '#4A90A4' for i in range(len(alg_names))]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    
    # Improve median and mean line visibility
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    for mean in box['means']:
        mean.set_markerfacecolor('white')
        mean.set_markeredgecolor('black')

    # Set titles and labels
    ax.set_title('Algorithm Performance Comparison', fontsize=18, fontweight='bold')
    ax.set_ylabel('Performance Value Distribution (Log Scale)', fontsize=14, fontweight='bold')
    
    # Use a logarithmic scale for the y-axis due to the vast range in data values
    ax.set_yscale('log')
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, which="both", ls="--", c='0.65') # Grid for both major and minor ticks on log scale

    # 6. Save the final plot
    plt.tight_layout()
    output_path = output_dir / 'performance_comparison_boxplot.png'
    plt.savefig(output_path)
    plt.close() # Close the figure to free up memory
    
    print(f"✅ Performance box plot saved successfully to '{output_path}'")

# ============================================================================
# MAIN EXECUTION WORKFLOW
# ============================================================================

def main():
    """
    Main function to load data and orchestrate the visualization process.
    """
    setup_matplotlib_style()
    
    # Define file and directory paths
    data_filename = 'boxplot_data_20250818_032342.json'
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🔬 Algorithm Performance Visualization 🔬")
    print("="*80)

    # --- Load Data ---
    try:
        print(f"📂 Attempting to load data from '{data_filename}'...")
        with open(data_filename, 'r') as f:
            # The provided JSON has the actual data nested under a 'data' key
            loaded_json = json.load(f)
            experiment_data = loaded_json.get('data')
            if not experiment_data:
                raise KeyError("The top-level 'data' key was not found in the JSON file.")
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Data file '{data_filename}' not found. Please ensure it is in the same directory.")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error: Could not read or parse the data file. Details: {e}")
        return

    # --- Generate Visualizations ---
    create_performance_boxplot(experiment_data, output_dir)
    
    print("\n🎉 Visualization script finished.")

if __name__ == "__main__":
    main()