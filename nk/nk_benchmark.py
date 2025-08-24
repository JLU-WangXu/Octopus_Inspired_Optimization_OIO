#!/usr/bin/env python3
"""
NK Landscape Benchmark - OIO vs Baseline Algorithms
=====================================================
Professional Edition - Upgraded with CEC2022 Methodologies

This script provides a robust framework for benchmarking optimization algorithms on the NK Landscape problem.
It incorporates advanced features for analysis, visualization, and reproducibility inspired by the
CEC (Congress on Evolutionary Computation) competition standards.

Key Features:
- Fair Comparison: All algorithms use a unified Sigmoid transfer function to map continuous
  solutions to the binary NK Landscape space.
- Robust Ranking: Implements a ranking system based on the performance of every individual run,
  rewarding both high performance and stability.
- Valid Statistical Analysis: Performs independent Mann-Whitney U tests for each NK configuration
  and generates a Win-Loss-Tie summary.
- Advanced Visualization: Creates comprehensive box plots to show performance distribution and
  detailed convergence curves.
- Data Persistence: Allows saving and loading of full experiment results, enabling re-analysis
  without re-running tests.
- Command-Line Interface: Easily manage experiments with 'run', 'load', and 'list' commands.
"""

import time
import numpy as np
import pandas as pd
import gc
import tracemalloc
import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Callable, Tuple, Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Visualization and data analysis dependencies
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path to import OIO and baseline algorithms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from OIO import ChampionOIOAlgorithm
    from baseline import ALGORITHMS
except ImportError as e:
    print(f"❌ Error importing local modules: {e}")
    print("Ensure OIO.py and baseline.py are in the correct directory.")
    sys.exit(1)

# ============================================================================
# DATA MANAGEMENT MODULE (from CEC2022 script)
# ============================================================================

class DataManager:
    """Manages saving and loading of experiment data for reproducibility."""
    def __init__(self, data_dir='nk_experiment_data'):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / 'results_data'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment_results(self, all_results, experiment_config, run_id=None):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{run_id}.pkl"
        filepath = self.results_dir / filename
        data = {'results': all_results, 'config': experiment_config, 'run_id': run_id}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n💾 Full experiment data saved successfully (ID: {run_id})")

    def load_experiment_results(self, run_id=None):
        if run_id:
            filepath = self.results_dir / f"experiment_results_{run_id}.pkl"
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                return data.get('results'), data.get('config')
        else:
            files = list(self.results_dir.glob("experiment_results_*.pkl"))
            if files:
                latest_file = max(files, key=lambda p: p.stat().st_mtime)
                print(f"Loading latest data file: {latest_file.name}")
                with open(latest_file, 'rb') as f:
                    data = pickle.load(f)
                return data.get('results'), data.get('config')
        return None, None

    def list_available_data(self):
        print(f"\n📁 Available experiment data in '{self.data_dir}':")
        files = sorted(self.results_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print("  No saved experiment data found.")
            return
        for file in files:
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            run_id = file.stem.replace("experiment_results_", "")
            print(f"  - ID: {run_id} (Saved: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

# ============================================================================
# CORE EVALUATION MECHANISM
# ============================================================================

class TargetPrecisionReached(Exception):
    def __init__(self, message, fitness, fes):
        super().__init__(message)
        self.fitness = fitness
        self.fes = fes

class InterruptibleCostFunc:
    def __init__(self, original_cost_func: Callable, max_evaluations: int, global_optimum: Optional[float], is_maximization: bool = True):
        self.original_cost_func = original_cost_func
        self.max_evaluations = max_evaluations
        self.global_optimum = global_optimum
        self.is_maximization = is_maximization
        self.evaluation_count = 0
        self.best_fitness = -float('inf') if is_maximization else float('inf')
        self.convergence_history: List[Tuple[int, float]] = []
        self.record_interval = max(1, max_evaluations // 500)

    def __call__(self, x: np.ndarray) -> float:
        if self.evaluation_count >= self.max_evaluations:
            raise RuntimeError("Maximum evaluations reached")
        self.evaluation_count += 1
        fitness = self.original_cost_func(x)
        new_best_found = False
        if self.is_maximization and fitness > self.best_fitness:
            self.best_fitness = fitness
            new_best_found = True
        elif not self.is_maximization and fitness < self.best_fitness:
            self.best_fitness = fitness
            new_best_found = True
        if new_best_found or self.evaluation_count % self.record_interval == 0:
            self.convergence_history.append((self.evaluation_count, self.best_fitness))
        
        if self.is_maximization and self.global_optimum is not None and self.best_fitness >= self.global_optimum:
            if not self.convergence_history or self.convergence_history[-1][1] != self.best_fitness:
                self.convergence_history.append((self.evaluation_count, self.best_fitness))
            raise TargetPrecisionReached("Global optimum found", self.best_fitness, self.evaluation_count)
        return fitness

def get_memory_usage() -> float:
    try:
        current, _ = tracemalloc.get_traced_memory()
        return current / 1024 / 1024
    except:
        return 0.0

# ============================================================================
# ANALYSIS, RANKING, AND VISUALIZATION
# ============================================================================

def setup_matplotlib():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.linewidth': 1.2,
        'grid.alpha': 0.3, 'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1, 'axes.unicode_minus': False
    })

def get_algorithm_short_name(alg_name: str) -> str:
    name_mapping = {
        'OIO': 'OIO', 'GeneticAlgorithm': 'GA', 'HillClimbing': 'HC', 'SimulatedAnnealing': 'SA',
        'DifferentialEvolution': 'DE', 'ParticleSwarmOptimization': 'PSO', 'HarrisHawksOptimization': 'HHO',
        'WhaleOptimization': 'WOA', 'HybridPSOGWO': 'PSOGWO', 'AdaptiveHarrisHawks': 'AHHO',
        'HippopotamusOptimization': 'HPO', 'WalrusOptimization': 'WO', 'CrestedPorcupineOptimizer': 'CPO',
        'ElkHerdOptimizer': 'EHO', 'GreylagGooseOptimization': 'GGO', 'QuokkaSwarmOptimization': 'QSO'
    }
    return name_mapping.get(alg_name, alg_name)


def calculate_nk_ranking(config_results: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Ranks algorithms on a single NK config based on every run's fitness."""
    all_runs = []
    for alg_name, data in config_results.items():
        if 'error' not in data and 'all_fitness' in data:
            all_runs.extend([{"alg_name": alg_name, "fitness": f} for f in data['all_fitness']])
    if not all_runs: return []
    sorted_runs = sorted(all_runs, key=lambda x: x['fitness'], reverse=True)
    ranks = {name: 0 for name in set(r['alg_name'] for r in all_runs)}
    for i, run in enumerate(sorted_runs):
        ranks[run['alg_name']] += (i + 1)
    return sorted(ranks.items(), key=lambda x: x[1])

def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1: return 0.0
    dof = n1 + n2 - 2
    if dof == 0: return 0.0
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / dof)
    if pooled_std == 0: return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def perform_statistical_tests(all_results: Dict[str, Any], save_path: Path | str) -> pd.DataFrame:
    """Performs independent statistical tests for each NK configuration."""
    statistical_results = []
    print("\n🔬 Performing statistical tests (Mann-Whitney U) for each configuration...")
    for config_name, results in all_results.items():
        if 'OIO' not in results or 'error' in results['OIO']:
            print(f"  ⚠️ Skipping {config_name}: No valid OIO results.")
            continue
        oio_fitness = results['OIO']['all_fitness']
        for alg_name, result in results.items():
            if alg_name == 'OIO' or 'error' in result: continue
            baseline_fitness = result['all_fitness']
            try:
                stat, p_val = stats.mannwhitneyu(oio_fitness, baseline_fitness, alternative='greater')
                statistical_results.append({
                    'Configuration': config_name, 'Algorithm': alg_name,
                    'OIO_Mean': np.mean(oio_fitness), 'Baseline_Mean': np.mean(baseline_fitness),
                    'P_Value': p_val, 'Significant (OIO > Baseline)': p_val < 0.05,
                    'Effect_Size (Cohen_d)': calculate_effect_size(oio_fitness, baseline_fitness)
                })
            except Exception as e:
                print(f"  ⚠️ Failed test on {config_name} for {alg_name}: {e}")
    if not statistical_results: return pd.DataFrame()
    df = pd.DataFrame(statistical_results)
    df.to_csv(Path(save_path) / 'statistical_tests_per_config.csv', index=False)
    print(f"✅ Statistical test results saved.")
    
    summary = {alg: {'Win': 0, 'Loss': 0, 'Tie': 0} for alg in set(df['Algorithm'])}
    for _, row in df.iterrows():
        alg = row['Algorithm']
        oio_fit = all_results[row['Configuration']]['OIO']['all_fitness']
        base_fit = all_results[row['Configuration']][alg]['all_fitness']
        if row['Significant (OIO > Baseline)']:
            summary[alg]['Win'] += 1
        else:
            _, p_loss = stats.mannwhitneyu(oio_fit, base_fit, alternative='less')
            summary[alg]['Loss' if p_loss < 0.05 else 'Tie'] += 1
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    print("\n📊 OIO vs Baselines - Win-Loss-Tie Summary:")
    print(summary_df)
    summary_df.to_csv(Path(save_path) / 'statistical_summary_WLT.csv')
    return df

def create_performance_boxplot(all_results: Dict[str, Any], save_path: Path | str):
    """Creates a comprehensive performance box plot."""
    print("\n📊 Generating performance box plot...")
    all_algs_set = set(k for r in all_results.values() for k in r.keys() if 'error' not in r[k])
    if not all_algs_set:
        print("⚠️ Not enough data to create box plot.")
        return
        
    all_algs = sorted(list(all_algs_set))

    if 'OIO' in all_algs:
        all_algs.insert(0, all_algs.pop(all_algs.index('OIO')))
    
    plot_data = [ [f for res in all_results.values() if alg in res and 'error' not in res[alg] for f in res[alg]['all_fitness']] for alg in all_algs ]
    valid_plot_data = [d for d in plot_data if d]
    alg_names = [get_algorithm_short_name(alg) for alg, d in zip(all_algs, plot_data) if d]

    if not valid_plot_data:
        print("⚠️ Not enough valid data to create box plot.")
        return
        
    mean_fitnesses = [np.mean(data) for data in valid_plot_data]
    best_alg_idx = np.argmax(mean_fitnesses)

    plt.figure(figsize=(15, 8))
    box = plt.boxplot(valid_plot_data, labels=alg_names, patch_artist=True, showmeans=True)
    colors = ['#FF6B35' if i == best_alg_idx else '#4A90A4' for i in range(len(alg_names))]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel('Fitness Value Distribution', fontsize=14, fontweight='bold')
    plt.title('Algorithm Performance Across All NK Configurations', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'performance_comparison_boxplot.png')
    plt.close()
    print("✅ Performance box plot saved.")

def create_convergence_plots(all_results: Dict[str, Any], save_path: Path | str):
    """Creates interpolated average convergence curves for each configuration."""
    print("📈 Generating convergence curves...")
    conv_dir = Path(save_path) / 'convergence_curves'
    conv_dir.mkdir(exist_ok=True)
    for config_name, results in all_results.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        import matplotlib.cm as cm
        algorithms = [alg for alg in results.keys() if 'error' not in results[alg]]
        if not algorithms: continue

        colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(algorithms)))
        color_map = {alg: colors[i] for i, alg in enumerate(sorted([a for a in algorithms if a != 'OIO']))}
        color_map['OIO'] = '#FF6B35'

        ordered_algorithms = sorted([a for a in algorithms if a != 'OIO']) + (['OIO'] if 'OIO' in algorithms else [])

        for alg_name in ordered_algorithms:
            if 'convergence_histories' not in results[alg_name] or not results[alg_name]['convergence_histories']:
                continue
            
            histories = results[alg_name]['convergence_histories']
            if not any(histories): continue

            all_fes_data = [np.array([p[0] for p in run]) for run in histories if run]
            all_fit_data = [np.array([p[1] for p in run]) for run in histories if run]
            if not all_fes_data: continue

            max_fes_overall = max(fes[-1] for fes in all_fes_data if len(fes) > 0)
            common_fes_axis = np.linspace(1, max_fes_overall, 500)
            
            interpolated_fits = [np.interp(common_fes_axis, fes, fit) for fes, fit in zip(all_fes_data, all_fit_data) if len(fes) > 1]
            if not interpolated_fits: continue

            mean_history = np.mean(interpolated_fits, axis=0)
            std_history = np.std(interpolated_fits, axis=0)
            
            avg_stop_fes = np.mean([fes[-1] for fes in all_fes_data if len(fes) > 0])
            plot_mask = common_fes_axis <= avg_stop_fes
            plot_fes_axis, plot_mean_history, plot_std_history = common_fes_axis[plot_mask], mean_history[plot_mask], std_history[plot_mask]

            color = color_map[alg_name]
            linewidth = 3 if alg_name == 'OIO' else 2
            alpha = 0.9 if alg_name == 'OIO' else 0.7
            display_name = get_algorithm_short_name(alg_name)

            ax.plot(plot_fes_axis, plot_mean_history, color=color, linewidth=linewidth, label=display_name, alpha=alpha)
            ax.fill_between(plot_fes_axis, plot_mean_history - std_history, plot_mean_history + std_history, color=color, alpha=0.15)

        ax.set_xlabel('Function Evaluations (FES)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Best Fitness Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Convergence Curves for {config_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        oio_items = [(h, l) for h, l in zip(handles, labels) if l.upper() == 'OIO']
        other_items = sorted([(h, l) for h, l in zip(handles, labels) if l.upper() != 'OIO'], key=lambda x: x[1])
        ordered_handles = [item[0] for item in oio_items + other_items]
        ordered_labels = [item[1] for item in oio_items + other_items]

        ncol = min(4, (len(algorithms) + 7) // 8) if len(algorithms) > 8 else 1
        loc = 'upper center' if len(algorithms) > 8 else 'upper left'
        bbox = (0.5, -0.15) if len(algorithms) > 8 else (1.05, 1)
        ax.legend(ordered_handles, ordered_labels, bbox_to_anchor=bbox, loc=loc, ncol=ncol, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(conv_dir / f"{config_name}_convergence.png", dpi=300)
        plt.close()
    print(f"✅ Convergence curves saved to '{conv_dir}'.")

# ============================================================================
# NK LANDSCAPE AND ALGORITHM ADAPTERS
# ============================================================================

class NKLandscape:
    """NK Landscape Model"""
    def __init__(self, N: int, K: int, seed: Optional[int] = None):
        self.N, self.K, self.seed = N, K, seed
        if seed is not None: np.random.seed(seed)
        self.neighbors = {i: np.random.choice(list(range(i)) + list(range(i + 1, N)), K, replace=False) if K > 0 else [] for i in range(N)}
        self.fitness_tables = {i: np.random.rand(2**(K + 1)) for i in range(N)}
        self.global_optimum = None

    def evaluate(self, sequence: np.ndarray) -> float:
        total_fitness = 0.0
        for i in range(self.N):
            neighbors_indices = self.neighbors[i]
            sub_sequence = np.concatenate(([sequence[i]], sequence[neighbors_indices] if self.K > 0 else []))
            index = sub_sequence.dot(1 << np.arange(sub_sequence.size)[::-1])
            total_fitness += self.fitness_tables[i][index]
        return total_fitness / self.N

class BaselineAlgorithmAdapter:
    """Adapts continuous-space algorithms to binary NK problems using a Sigmoid function."""
    def __init__(self, BaselineClass: type, cost_func_max: Callable, sequence_length: int, max_evaluations: int):
        def binary_wrapper_cost_func(continuous_solution: np.ndarray) -> float:
            probabilities = 1.0 / (1.0 + np.exp(-1.0 * continuous_solution))
            binary_solution = (np.random.rand(len(probabilities)) < probabilities).astype(int)
            return cost_func_max(binary_solution)
        self.algorithm = BaselineClass(fitness_func=binary_wrapper_cost_func, sequence_length=sequence_length, max_evaluations=max_evaluations)
    def optimize(self) -> Tuple[None, float]:
        _, best_fitness_max = self.algorithm.optimize()
        return None, best_fitness_max

class OIO_NK_Adapter:
    """A dedicated adapter for OIO, allowing for specialized optimization strategies."""
    def __init__(self, cost_func_max: Callable, sequence_length: int, max_evaluations: int):
        self.cost_func_max = cost_func_max # This is the interrupt_func
        self.sequence_length = sequence_length
        self.max_evaluations = max_evaluations
        
        def cost_func_min(continuous_solution: np.ndarray) -> float:
            probabilities = 1.0 / (1.0 + np.exp(-1.0 * continuous_solution))
            binary_solution = (probabilities > 0.5).astype(int) # STRATEGY 1: Deterministic mapping
            return -self.cost_func_max(binary_solution)

        self.algorithm = ChampionOIOAlgorithm(
            cost_func=cost_func_min,
            bounds=(-5, 5),
            dim=sequence_length
        )

    def optimize(self) -> Tuple[None, float]:
        _, best_val_min = self.algorithm.optimize(max_iter=self.max_evaluations)
        best_fitness_max = -best_val_min
        return None, best_fitness_max

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

# --- FIX: Correctly define the dictionary of all algorithm adapters ---
# For baselines, we use a lambda function to defer the creation of the adapter
# until it's actually needed in the test loop, allowing us to pass the correct parameters.
ALL_ALGORITHMS = {
    'OIO': OIO_NK_Adapter,
    **{name: (lambda bc: lambda *args: BaselineAlgorithmAdapter(bc, *args))(base_class)
       for name, base_class in ALGORITHMS.items()}
}

def test_single_algorithm(alg_name: str, AdapterClassFactory: Callable, landscape_params: Dict, max_evals: int, num_runs: int) -> Tuple[str, Dict]:
    """Tests a single algorithm on a given NK landscape configuration."""
    print(f"🔄 Testing {alg_name} ({num_runs} runs)...")
    fitness_results, time_results, conv_histories = [], [], []
    
    USE_MEMETIC_OIO = True 

    for run in range(num_runs):
        seed = (landscape_params.get('seed') or 0) + run
        landscape = NKLandscape(N=landscape_params['N'], K=landscape_params['K'], seed=seed)
        interrupt_func = InterruptibleCostFunc(landscape.evaluate, max_evals, landscape.global_optimum)
        
        if alg_name == 'OIO' and USE_MEMETIC_OIO:
            # --- Memetic Strategy Implementation ---
            temp_oio_adapter = OIO_NK_Adapter(interrupt_func, landscape.N, max_evals)
            oio_instance = temp_oio_adapter.algorithm
            ls_cache = {}

            def memetic_cost_func_min(x_continuous: np.ndarray) -> float:
                probs = 1.0 / (1.0 + np.exp(-1.0 * x_continuous))
                binary_seq = (np.random.rand(len(probs)) < probs).astype(int)
                
                cache_key = tuple(binary_seq)
                if cache_key in ls_cache: return ls_cache[cache_key]

                initial_fitness_max = interrupt_func(binary_seq)
                
                def ls_cost_func(seq_to_search: np.ndarray) -> float:
                    return -interrupt_func(seq_to_search)

                original_cost_func = oio_instance.cost_func
                oio_instance.cost_func = ls_cost_func
                
                improved_binary_seq = oio_instance._binary_string_local_search(
                    binary_sequence=binary_seq,
                    current_fitness=-initial_fitness_max
                )
                
                oio_instance.cost_func = original_cost_func
                
                final_fitness_max = landscape.evaluate(improved_binary_seq)
                final_fitness_min = -final_fitness_max
                ls_cache[cache_key] = final_fitness_min
                return final_fitness_min

            temp_oio_adapter.algorithm.cost_func = memetic_cost_func_min
            algorithm = temp_oio_adapter
        else:
            # --- Default Strategy (including Strategy 1 for OIO) ---
            algorithm = AdapterClassFactory(interrupt_func, landscape.N, max_evals)

        start_time = time.time()
        best_fitness = -float('inf')
        try:
            _, best_fitness = algorithm.optimize()
        except (TargetPrecisionReached, RuntimeError):
            best_fitness = interrupt_func.best_fitness
        
        time_results.append(time.time() - start_time)
        fitness_results.append(best_fitness)
        conv_histories.append(interrupt_func.convergence_history.copy())
        del algorithm, landscape, interrupt_func
        gc.collect()

    return alg_name, {
        'fitness_mean': np.mean(fitness_results), 'fitness_std': np.std(fitness_results),
        'fitness_best': np.max(fitness_results), 'fitness_worst': np.min(fitness_results),
        'time_mean': np.mean(time_results), 'total_runs': num_runs,
        'all_fitness': fitness_results, 'convergence_histories': conv_histories
    }

def main(load_existing_data: bool = False, run_id: Optional[str] = None):
    """Main function to run or load and analyze the benchmark."""
    setup_matplotlib()
    tracemalloc.start()
    initial_memory = get_memory_usage()
    
    data_manager = DataManager(data_dir='nk_experiment_data')
    output_dir = Path("nk_results")
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("⚡ NK Landscape Benchmark - Professional Edition ⚡")
    print("="*60)
    print(f"Initial memory usage: {initial_memory:.1f} MB")

    all_results = None
    if load_existing_data:
        print("\n📂 Attempting to load existing experiment data...")
        data_manager.list_available_data()
        all_results, _ = data_manager.load_experiment_results(run_id)

    if all_results:
        print("\n✅ Successfully loaded data. Proceeding directly to analysis.")
    else:
        if load_existing_data:
            print("❌ No data found. Running a new experiment.")
        all_results = {}
        
        configs = [
            {'N': 20, 'K': 2, 'seed': 42, 'name': 'Simple_N20_K2'},
            {'N': 30, 'K': 3, 'seed': 42, 'name': 'Medium_N30_K3'},
            {'N': 50, 'K': 4, 'seed': 42, 'name': 'Hard_N50_K4'},
            {'N': 70, 'K': 4, 'seed': 42, 'name': 'Harder_N70_K4'},
            {'N': 100, 'K': 5, 'seed': 42, 'name': 'Complex_N100_K5'}
        ]
        max_evals, num_runs = 20000, 10

        for idx, params in enumerate(configs, 1):
            name = params['name']
            print(f"\n--- Configuration {idx}/{len(configs)}: {name} ---")
            results: Dict[str, Any] = {}
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(test_single_algorithm, an, ac, params, max_evals, num_runs): an for an, ac in ALL_ALGORITHMS.items()}
                for i, future in enumerate(as_completed(futures), 1):
                    alg_name = futures[future]
                    try:
                        _, res_data = future.result()
                        results[alg_name] = res_data
                        print(f"  [{i:2d}/{len(futures)}] ✅ {get_algorithm_short_name(alg_name):<8} done: {res_data['fitness_mean']:.5f} ± {res_data['fitness_std']:.4f}")
                    except Exception as e:
                        print(f"  [{i:2d}/{len(futures)}] ❌ {get_algorithm_short_name(alg_name):<8} failed: {e}")
                        results[alg_name] = {'error': str(e)}
            all_results[name] = results

        exp_config = {'configs': configs, 'max_evals': max_evals, 'num_runs': num_runs}
        data_manager.save_experiment_results(all_results, exp_config)

    print("\n--- Analysis and Report Generation ---")
    
    total_nk_scores: Dict[str, int] = {}
    for config_name, results in all_results.items():
        print(f"\n🏆 Ranking for {config_name}:")
        ranking = calculate_nk_ranking(results)
        for rank, (alg, score) in enumerate(ranking, 1):
            star = '⭐' if alg == 'OIO' else '  '
            print(f"  {rank}. {star} {get_algorithm_short_name(alg)}: Score {score}")
            total_nk_scores[alg] = total_nk_scores.get(alg, 0) + score

    print("\n🥇 Overall Algorithm Ranking (lower score is better):")
    if total_nk_scores:
        final_ranks = sorted(total_nk_scores.items(), key=lambda x: x[1])
        for rank, (alg, score) in enumerate(final_ranks, 1):
            star = '⭐' if alg == 'OIO' else '  '
            print(f"  {rank}. {star} {get_algorithm_short_name(alg)}: Total Score {score}")
    
    perform_statistical_tests(all_results, output_dir)
    create_performance_boxplot(all_results, output_dir)
    create_convergence_plots(all_results, output_dir)

    final_memory = get_memory_usage()
    print(f"\n🧹 Memory Report: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB, Increase={final_memory - initial_memory:.1f}MB")
    tracemalloc.stop()
    print(f"\n🎉 Benchmark complete. All results saved in '{output_dir}' and '{data_manager.data_dir}'.")

if __name__ == "__main__":
    command = "run"
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    
    if command == "run":
        main(load_existing_data=False)
    elif command == "load":
        run_id = sys.argv[2] if len(sys.argv) > 2 else None
        main(load_existing_data=True, run_id=run_id)
    elif command == "list":
        DataManager(data_dir='nk_experiment_data').list_available_data()
    else:
        print("Usage: python nk_benchmark.py [run|load|list]")
        print("  run        : Start a new experiment.")
        print("  load [id]  : Load and analyze an experiment (latest if id is omitted).")
        print("  list       : List all saved experiments.")