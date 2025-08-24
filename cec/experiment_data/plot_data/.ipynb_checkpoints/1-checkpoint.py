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
    all_performance_data = data.get('algorithm_all_data')
    if not all_performance_data:
        print("⚠️  Warning: 'algorithm_all_data' key not found or is empty. Cannot generate plot.")
        return

    # 2. Prepare data and labels for the plot
    alg_names = sorted([name for name in all_performance_data.keys() if name != 'OIO'])
    if 'OIO' in all_performance_data:
        alg_names.insert(0, 'OIO')
    
    plot_data = [all_performance_data[name] for name in alg_names]

    # 3. Identify the best-performing algorithm to highlight it
    medians = [np.median(d) for d in plot_data]
    best_alg_idx = np.argmin(medians)

    # 4. Create the plot, increasing height for better visibility
    fig, ax = plt.subplots(figsize=(16, 10))

    # Draw the boxplot, hiding extreme outliers to focus the view
    box = ax.boxplot(plot_data, labels=alg_names, patch_artist=True, showmeans=True, showfliers=False)

    # --- MODIFICATION 1: RE-ADD OUTLIER MARKERS AND ANNOTATION ---
    # After drawing, get the y-axis limit to determine what's "off-scale"
    y_axis_top = ax.get_ylim()[1]

    # Manually check for and mark any off-scale outliers
    for i, data_points in enumerate(plot_data):
        q1 = np.percentile(data_points, 25)
        q3 = np.percentile(data_points, 75)
        iqr = q3 - q1
        upper_whisker = q3 + 1.5 * iqr
        
        has_off_scale_outliers = any(p > upper_whisker and p > y_axis_top for p in data_points)
        if has_off_scale_outliers:
            ax.plot(i + 1, y_axis_top, 'v', color='red', markersize=10, clip_on=False)

    # Add the annotation text box to explain the markers
    ax.text(0.98, 0.97, '▼ Red triangles indicate outliers exist beyond the Y-axis range',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. Apply professional styling
    colors = ['#FF6B35' if i == best_alg_idx else '#4A90A4' for i in range(len(alg_names))]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    
    # --- MODIFICATION 2: CHANGE MEDIAN LINE TO RED ---
    # Change the median line to a thicker red for emphasis and style consistency
    for median in box['medians']:
        median.set_color('red') # Changed from 'black' to 'red'
        median.set_linewidth(2) # Made slightly thicker

    for mean in box['means']:
        mean.set_markerfacecolor('white')
        mean.set_markeredgecolor('black')

    # Set titles and labels
    ax.set_title('Algorithm Performance Comparison (Core Distribution)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Performance Value Distribution (Log Scale)', fontsize=14, fontweight='bold')
    
    # Use a logarithmic scale for the y-axis
    ax.set_yscale('log')
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, which="both", ls="--", c='0.65')

    # 6. Save the final plot
    plt.tight_layout()
    output_path = output_dir / 'performance_comparison_final.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ Final focused box plot saved successfully to '{output_path}'")

# ============================================================================
# MAIN EXECUTION WORKFLOW
# ============================================================================

def main():
    """
    Main function to load data and orchestrate the visualization process.
    """
    setup_matplotlib_style()
    
    data_filename = 'boxplot_data_20250818_032342.json'
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🔬 Algorithm Performance Visualization 🔬")
    print("="*80)

    try:
        print(f"📂 Attempting to load data from '{data_filename}'...")
        with open(data_filename, 'r') as f:
            loaded_json = json.load(f)
            experiment_data = loaded_json.get('data')
            if not experiment_data:
                raise KeyError("The top-level 'data' key was not found in the JSON file.")
        print("✅ Data loaded successfully.")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error: Could not read or parse the data file. Details: {e}")
        return

    create_performance_boxplot(experiment_data, output_dir)
    
    print("\n🎉 Visualization script finished.")

if __name__ == "__main__":
    main()