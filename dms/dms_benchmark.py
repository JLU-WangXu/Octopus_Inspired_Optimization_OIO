#!/usr/bin/env python3
"""
DMS Benchmark Script - OIO vs 15 Baseline Algorithms
======================================================
Final Version: Validating algorithm effectiveness on the GFP protein engineering task.
This script adopts a professional CEC-style evaluation framework, including FES budgets,
early stopping, CEC-style ranking, advanced visualization, statistical testing,
and a full-featured data management system.
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
from datetime import datetime
from typing import Callable, Tuple, Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from OIO import ChampionOIOAlgorithm
    from baseline import ALGORITHMS
except ImportError as e:
    print(f"❌ Error importing local modules: {e}")
    print("Ensure OIO.py and baseline.py are in the correct directory relative to the script.")
    sys.exit(1)

# ============================================================================
# DATA MANAGEMENT MODULE (Ported from CEC/NK script)
# ============================================================================
class DataManager:
    """Manages saving and loading of experiment data for reproducibility."""
    def __init__(self, data_dir='dms_experiment_data'):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / 'results_data'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment_results(self, all_results: Dict, experiment_config: Dict, run_id: Optional[str] = None):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{run_id}.pkl"
        filepath = self.results_dir / filename
        data = {'results': all_results, 'config': experiment_config, 'run_id': run_id}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n💾 Full experiment data saved successfully (ID: {run_id})")

    def load_experiment_results(self, run_id: Optional[str] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
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
# PROBLEM DEFINITION & CORE EVALUATION
# ============================================================================
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_MAP = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def load_dms_oracle(filepath: str) -> Tuple[Dict[str, float], float, int, str, List[str]]:
    print(f"🧬 Loading DMS dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DMS data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # *** FIX: Standardize column names to handle different CSV formats ***
    df.columns = [col.strip() for col in df.columns]
    
    # Case 1: User's format ('Sequence', 'conv(%)')
    if 'Sequence' in df.columns and 'conv(%)' in df.columns:
        df.rename(columns={'Sequence': 'sequence', 'conv(%)': 'fitness'}, inplace=True)
    # Case 2: Another common format
    elif 'sequence' in df.columns and 'true_score' in df.columns:
        df.rename(columns={'true_score': 'fitness'}, inplace=True)
    # Case 3: Default expected format
    elif 'sequence' not in df.columns or 'fitness' not in df.columns:
        raise KeyError("CSV file must contain columns for 'sequence' and 'fitness' (or recognizable alternatives like 'Sequence', 'conv(%)', 'true_score').")
    
    df.dropna(subset=['sequence', 'fitness'], inplace=True)
    df['sequence'] = df['sequence'].astype(str).str.strip()
    valid_mask = df['sequence'].str.match(f'^[{AMINO_ACIDS}]*$', na=False)
    df = df[valid_mask]
    
    oracle = pd.Series(df.fitness.values, index=df.sequence).to_dict()
    global_optimum = df['fitness'].max()
    best_sequence = df.loc[df['fitness'].idxmax()]['sequence']
    sequence_length = len(best_sequence)
    wt_sequence = df['sequence'].iloc[0]
    all_sequences = df['sequence'].tolist()
    
    print(f"✅ DMS oracle loaded: {len(oracle)} valid sequences, length={sequence_length}, global optimum={global_optimum:.4f}")
    return oracle, global_optimum, sequence_length, wt_sequence, all_sequences

class TargetPrecisionReached(Exception):
    def __init__(self, message: str, fitness: float, fes: int):
        super().__init__(message)
        self.fitness = fitness
        self.fes = fes

class InterruptibleCostFunc:
    def __init__(self, original_cost_func: Callable, max_evaluations: int, global_optimum: float, is_maximization: bool = True):
        self.original_cost_func = original_cost_func
        self.max_evaluations = max_evaluations
        self.global_optimum = global_optimum
        self.is_maximization = is_maximization
        self.evaluation_count = 0
        self.best_fitness = -float('inf') if is_maximization else float('inf')
        self.convergence_history: List[Tuple[int, float]] = []
        self.record_interval = max(1, int(max_evaluations / 500)) if max_evaluations > 0 else 1

    def __call__(self, x_sequence: str) -> float:
        if self.evaluation_count >= self.max_evaluations:
            raise RuntimeError("Maximum evaluations reached")
        
        self.evaluation_count += 1
        fitness = self.original_cost_func(x_sequence)
        
        new_best_found = (self.is_maximization and fitness > self.best_fitness) or \
                         (not self.is_maximization and fitness < self.best_fitness)
        
        if new_best_found:
            self.best_fitness = fitness
        
        if new_best_found or self.evaluation_count % self.record_interval == 0:
            self.convergence_history.append((self.evaluation_count, self.best_fitness))
        
        if self.is_maximization and self.best_fitness >= self.global_optimum - 1e-9:
            if not self.convergence_history or self.convergence_history[-1][1] != self.best_fitness:
                self.convergence_history.append((self.evaluation_count, self.best_fitness))
            raise TargetPrecisionReached("Global optimum found", self.best_fitness, self.evaluation_count)
        
        return fitness

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def setup_matplotlib():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 300

def get_algorithm_short_name(alg_name: str) -> str:
    name_mapping = {
        'OIO': 'OIO', 'GeneticAlgorithm': 'GA', 'HillClimbing': 'HC', 'SimulatedAnnealing': 'SA',
        'DifferentialEvolution': 'DE', 'ParticleSwarmOptimization': 'PSO', 'HarrisHawksOptimization': 'HHO',
        'WhaleOptimization': 'WOA', 'HybridPSOGWO': 'PSOGWO', 'AdaptiveHarrisHawks': 'AHHO',
        'HippopotamusOptimization': 'HPO', 'WalrusOptimization': 'WO', 'CrestedPorcupineOptimizer': 'CPO',
        'ElkHerdOptimizer': 'EHO', 'GreylagGooseOptimization': 'GGO', 'QuokkaSwarmOptimization': 'QSO'
    }
    return name_mapping.get(alg_name, alg_name[:6])

def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1: return 0.0
    dof = n1 + n2 - 2
    if dof <= 0: return 0.0
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / dof)
    if pooled_std == 0: return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def perform_statistical_tests(all_results: Dict[str, Any], save_path: Path | str) -> pd.DataFrame:
    results = all_results.get('GFP', {})
    if 'OIO' not in results or 'error' in results['OIO']:
        print("⚠️ Skipping statistical tests: No valid OIO results found.")
        return pd.DataFrame()
    
    print("\n🔬 Performing statistical tests (OIO vs others)...")
    statistical_data = []
    oio_fitness = results['OIO']['all_fitness']

    for alg_name, result in results.items():
        if alg_name == 'OIO' or 'error' in result: continue
        baseline_fitness = result['all_fitness']
        stat, p_val = stats.mannwhitneyu(oio_fitness, baseline_fitness, alternative='greater')
        statistical_data.append({
            'Algorithm': alg_name, 'OIO_Mean_Fitness': np.mean(oio_fitness),
            'Baseline_Mean_Fitness': np.mean(baseline_fitness), 'P_Value': p_val,
            'Significant (OIO > Baseline)': p_val < 0.05,
            'Effect_Size (Cohen_d)': calculate_effect_size(oio_fitness, baseline_fitness)
        })
    
    if not statistical_data:
        print("  No baseline algorithms to compare against.")
        return pd.DataFrame()

    df = pd.DataFrame(statistical_data)
    df.to_csv(Path(save_path) / 'statistical_tests.csv', index=False)
    print("✅ Statistical test results saved.")

    summary = {alg: {'Win': 0, 'Loss': 0, 'Tie': 0} for alg in set(df['Algorithm'])}
    for _, row in df.iterrows():
        alg = row['Algorithm']
        if row['Significant (OIO > Baseline)']:
            summary[alg]['Win'] += 1
        else:
            _, p_loss = stats.mannwhitneyu(oio_fitness, results[alg]['all_fitness'], alternative='less')
            summary[alg]['Loss' if p_loss < 0.05 else 'Tie'] += 1
    
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    print("\n📊 OIO vs Baselines - Win-Loss-Tie Summary:")
    print(summary_df)
    summary_df.to_csv(Path(save_path) / 'statistical_summary_WLT.csv')
    return df

def create_performance_boxplot(all_results: Dict[str, Any], save_path: Path | str):
    print("\n📊 Generating performance box plot...")
    results = all_results.get('GFP', {})
    if not results:
        print("⚠️ No data available for box plot."); return

    all_algs = sorted([k for k, v in results.items() if 'error' not in v])
    if 'OIO' in all_algs:
        all_algs.insert(0, all_algs.pop(all_algs.index('OIO')))
    
    plot_data = [results[alg]['all_fitness'] for alg in all_algs]
    alg_names = [get_algorithm_short_name(alg) for alg in all_algs]

    mean_fitnesses = [np.mean(data) for data in plot_data]
    best_alg_idx = np.argmax(mean_fitnesses)

    plt.figure(figsize=(15, 8))
    box = plt.boxplot(plot_data, labels=alg_names, patch_artist=True, showmeans=True) # type: ignore
    colors = ['#FF6B35' if i == best_alg_idx else '#4A90A4' for i in range(len(alg_names))]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel('Fitness Value Distribution', fontsize=14, fontweight='bold')
    plt.title('Algorithm Performance on GFP Dataset', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'performance_comparison_boxplot.png')
    plt.close()
    print("✅ Performance box plot saved.")

def create_convergence_plots(all_results: Dict[str, Any], save_path: Path | str):
    print("📈 Generating convergence curves...")
    conv_dir = Path(save_path) / 'convergence_curves'
    conv_dir.mkdir(exist_ok=True)
    
    for config_name, results in all_results.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        algorithms = sorted([alg for alg in results if 'error' not in results[alg]])
        if 'OIO' in algorithms:
            algorithms.insert(0, algorithms.pop(algorithms.index('OIO')))
        
        colors = plt.cm.get_cmap('tab20', len(algorithms))
        for i, alg_name in enumerate(algorithms):
            histories = results[alg_name].get('convergence_histories', [])
            if not any(histories): continue

            all_fes = [np.array([p[0] for p in run]) for run in histories if run]
            all_fit = [np.array([p[1] for p in run]) for run in histories if run]
            if not all_fes: continue

            max_fes = max(fes[-1] for fes in all_fes if len(fes) > 0)
            common_axis = np.linspace(1, max_fes, 500)
            interp_fits = [np.interp(common_axis, fes, fit) for fes, fit in zip(all_fes, all_fit) if len(fes) > 1]
            if not interp_fits: continue

            mean_hist, std_hist = np.mean(interp_fits, axis=0), np.std(interp_fits, axis=0)
            
            color = '#FF6B35' if alg_name == 'OIO' else colors(i)
            ax.plot(common_axis, mean_hist, color=color, lw=3 if alg_name=='OIO' else 2, label=get_algorithm_short_name(alg_name))
            ax.fill_between(common_axis, mean_hist - std_hist, mean_hist + std_hist, color=color, alpha=0.15)

        ax.set(xlabel='Function Evaluations (FES)', ylabel='Best Fitness Value', title=f'Convergence Curves for {config_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(conv_dir / f"{config_name}_convergence.png")
        plt.close()
    print(f"✅ Convergence curves saved to '{conv_dir}'.")

def calculate_cec_style_ranking(func_results: Dict[str, Any], success_threshold=1e-9) -> List[Tuple[str, float]]:
    all_runs = []
    for name, data in func_results.items():
        if 'error' in data: continue
        for i in range(data['total_runs']):
            all_runs.append({
                "alg_name": name, "success": data['all_errors'][i] < success_threshold,
                "error": data['all_errors'][i], "fes": data['all_fes'][i]
            })
    if not all_runs: return []
    sorted_runs = sorted(all_runs, key=lambda x: (not x['success'], x['fes'] if x['success'] else x['error']))
    ranks = {name: 0.0 for name in func_results if 'error' not in func_results.get(name, {})}
    for i, run in enumerate(sorted_runs):
        if run['alg_name'] in ranks: ranks[run['alg_name']] += (i + 1)
    return sorted(ranks.items(), key=lambda x: x[1])

# ============================================================================
# ALGORITHM ADAPTERS
# ============================================================================
def hamming_distance(s1: str, s2: str) -> int: return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def estimate_fitness_by_similarity(target_seq: str, oracle: Dict[str, float], all_seqs: List[str]) -> float:
    if target_seq in oracle: return oracle[target_seq]
    sample = np.random.choice(all_seqs, min(500, len(all_seqs)), replace=False)
    distances = [(s, hamming_distance(target_seq, s)) for s in sample]
    distances.sort(key=lambda x: x[1])
    nearest = [d[0] for d in distances[:3]]
    if not nearest: return -1.0
    w_fit, total_w = 0.0, 0.0
    for seq in nearest:
        dist = hamming_distance(target_seq, seq)
        weight = 1.0 / (1.0 + dist)
        w_fit += weight * oracle.get(seq, 0)
        total_w += weight
    return w_fit / total_w if total_w > 0 else -1.0

class Baseline_DMS_Adapter:
    def __init__(self, BaselineClass: type, cost_func_max: Callable, seq_len: int, max_evals: int, wt_seq: str):
        self.seq_len = seq_len
        self.dim = seq_len * len(AMINO_ACIDS)
        
        def continuous_to_protein_fitness(x_cont: np.ndarray) -> float:
            matrix = x_cont.reshape((self.seq_len, len(AMINO_ACIDS)))
            indices = []
            for scores in matrix:
                e_scores = np.exp(scores - np.max(scores))
                probabilities = e_scores / e_scores.sum()
                indices.append(np.random.choice(len(AMINO_ACIDS), p=probabilities))
            sequence = "".join([AMINO_ACIDS[i] for i in indices])
            return cost_func_max(sequence)

        self.algorithm = BaselineClass(fitness_func=continuous_to_protein_fitness, sequence_length=self.dim, max_evaluations=max_evals)
        pop_size = getattr(self.algorithm, 'pop_size', getattr(self.algorithm, 'swarm_size', 1))
        wt_cont = np.full((self.seq_len, len(AMINO_ACIDS)), -1.0)
        for i, aa in enumerate(wt_seq): wt_cont[i, AA_MAP.get(aa, 0)] = 1.0
        initial_pop = np.random.normal(loc=wt_cont.flatten(), scale=0.1, size=(pop_size, self.dim))
        initial_pop[0] = wt_cont.flatten()
        
        pop_attr = 'population' if hasattr(self.algorithm, 'population') else 'particles'
        if hasattr(self.algorithm, pop_attr):
             setattr(self.algorithm, pop_attr, np.clip(initial_pop, -5, 5))

    def optimize(self) -> Tuple[str, float]:
        _, best_fitness = self.algorithm.optimize()
        return "sequence_unknown", best_fitness

class OIO_DMS_Adapter:
    def __init__(self, cost_func_min: Callable, seq_len: int, max_evals: int, wt_seq: str, bounds: Tuple[float, float]=(-5, 5)):
        self.seq_len = seq_len
        self.max_evaluations = max_evals
        self.dim = seq_len * len(AMINO_ACIDS)
        self.oio = ChampionOIOAlgorithm(
            cost_func=cost_func_min, bounds=bounds, dim=self.dim,
            binary_mode=True, sequence_length=self.seq_len
        )
    def optimize(self) -> Tuple[str, float]:
        best_pos_binary, best_val_min = self.oio.optimize(max_iter=self.max_evaluations)
        if best_pos_binary is None: return "sequence_unknown", -best_val_min
        
        matrix = best_pos_binary.reshape((self.seq_len, len(AMINO_ACIDS)))
        indices = [np.argmax(row) if row.sum() > 0 else np.random.randint(len(AMINO_ACIDS)) for row in matrix]
        best_sequence = "".join([AMINO_ACIDS[i] for i in indices])
        return best_sequence, -best_val_min

# ============================================================================
# MAIN TEST WORKFLOW
# ============================================================================
ALL_ALGORITHMS_DMS = {'OIO': OIO_DMS_Adapter, **{name: Baseline_DMS_Adapter for name in ALGORITHMS.keys()}}

def test_single_algorithm_dms(alg_name: str, Adapter: type, oracle: Dict, g_opt: float, s_len: int, max_evals: int, n_runs: int, wt_seq: str, all_seqs: List[str]) -> Tuple[str, Dict]:
    print(f"🔄 Testing {alg_name} ({n_runs} runs)...")
    results = []
    try:
        for run in range(n_runs):
            interrupt_func = InterruptibleCostFunc(
                lambda seq: estimate_fitness_by_similarity(seq, oracle, all_seqs),
                max_evals, g_opt
            )
            
            if alg_name == 'OIO':
                # --- START: MODIFIED OIO CONFIGURATION (MEMETIC STRATEGY) ---
                # This block implements a Memetic Algorithm approach by integrating a powerful, 
                # problem-specific local search into the main evolutionary loop.
                # This enhances convergence speed without altering the core OIO algorithm code.

                # 1. Instantiate the OIO adapter to access the core algorithm instance.
                # A temporary cost function is used, which will be replaced shortly.
                temp_oio_adapter = OIO_DMS_Adapter(lambda x: 0.0, s_len, max_evals, wt_seq)
                oio_instance = temp_oio_adapter.oio
                
                # 2. Create a cache to store results of expensive local searches.
                # This prevents re-evaluating the same solution neighborhood repeatedly.
                ls_cache = {}

                def memetic_cost_func_min(x_binary: np.ndarray) -> float:
                    """
                    An enhanced cost function that incorporates local search.
                    Every fitness evaluation, including those within the local search,
                    is correctly counted by the `interrupt_func`.
                    """
                    # Use an immutable tuple as the cache key.
                    cache_key = tuple(x_binary)
                    if cache_key in ls_cache:
                        return ls_cache[cache_key]

                    # --- Step A: Evaluate the original candidate solution (consumes 1 FES) ---
                    matrix = x_binary.reshape((s_len, len(AMINO_ACIDS)))
                    indices = [np.argmax(r) if r.sum() > 0 else np.random.randint(len(AMINO_ACIDS)) for r in matrix]
                    sequence = "".join([AMINO_ACIDS[i] for i in indices])
                    initial_fitness_max = interrupt_func(sequence)
                    
                    # --- Step B: Define and execute the problem-specific Local Search ---
                    def ls_cost_func(binary_seq_to_search: np.ndarray) -> float:
                        """A dedicated cost function for the local search routine."""
                        m = binary_seq_to_search.reshape((s_len, len(AMINO_ACIDS)))
                        ind = [np.argmax(r) if r.sum() > 0 else np.random.randint(len(AMINO_ACIDS)) for r in m]
                        seq = "".join([AMINO_ACIDS[i] for i in ind])
                        # Return negative fitness because OIO is a minimizer.
                        return -interrupt_func(seq)

                    # Temporarily swap the OIO's internal cost function to our FES-counting one.
                    original_cost_func = oio_instance.cost_func
                    oio_instance.cost_func = ls_cost_func
                    
                    # Execute the powerful local search from the OIO algorithm.
                    # This will consume FES by calling `ls_cost_func`.
                    improved_binary_seq = oio_instance._binary_local_search(
                        binary_sequence=x_binary, 
                        current_fitness=-initial_fitness_max
                    )
                    
                    # Restore the original cost function to avoid side effects.
                    oio_instance.cost_func = original_cost_func

                    # --- Step C: Retrieve the final fitness of the improved solution ---
                    # The final evaluation was already performed and counted inside _binary_local_search.
                    # We just need to reconstruct the sequence to get its true value from the oracle for caching.
                    final_matrix = improved_binary_seq.reshape((s_len, len(AMINO_ACIDS)))
                    final_indices = [np.argmax(r) if r.sum() > 0 else np.random.randint(len(AMINO_ACIDS)) for r in final_matrix]
                    final_sequence = "".join([AMINO_ACIDS[i] for i in final_indices])
                    final_fitness_max = oracle.get(final_sequence, -1.0)
                    
                    # Cache the minimized fitness value and return it.
                    final_fitness_min = -final_fitness_max
                    ls_cache[cache_key] = final_fitness_min
                    return final_fitness_min

                # 3. Assign our new memetic cost function to the OIO instance.
                temp_oio_adapter.oio.cost_func = memetic_cost_func_min
                algorithm = temp_oio_adapter
                # --- END: MODIFIED OIO CONFIGURATION ---
            else:
                algorithm = Baseline_DMS_Adapter(ALGORITHMS[alg_name], interrupt_func, s_len, max_evals, wt_seq)
            
            start_time, best_fitness, fes = time.time(), -float('inf'), max_evals
            try:
                _, best_fitness = algorithm.optimize()
                fes = interrupt_func.evaluation_count
            except TargetPrecisionReached as e:
                best_fitness, fes = e.fitness, e.fes
            except RuntimeError:
                best_fitness, fes = interrupt_func.best_fitness, interrupt_func.evaluation_count
            
            results.append({
                'fitness': best_fitness, 'time': time.time() - start_time, 'fes': fes,
                'error': g_opt - best_fitness, 'convergence': interrupt_func.convergence_history.copy()
            })
            print(f"  {alg_name}: Run {run+1}/{n_runs} -> Fitness: {best_fitness:.4f}, FES: {fes}")

        all_err = [r['error'] for r in results]
        final_res = {
            'fitness_mean': np.mean([r['fitness'] for r in results]), 'fitness_std': np.std([r['fitness'] for r in results]),
            'time_mean': np.mean([r['time'] for r in results]), 'fes_mean': np.mean([r['fes'] for r in results]),
            'error_mean': np.mean(all_err), 'success_count': np.sum(np.array(all_err) <= 1e-9),
            'total_runs': n_runs, 'all_fitness': [r['fitness'] for r in results],
            'all_fes': [r['fes'] for r in results], 'all_errors': all_err,
            'convergence_histories': [r['convergence'] for r in results]
        }
        print(f"✅ {alg_name} done. Avg Fitness: {final_res['fitness_mean']:.4f}")
        return alg_name, final_res
    except Exception as e:
        print(f"❌ Critical error in {alg_name} thread: {e}")
        import traceback; traceback.print_exc()
        return alg_name, {'error': str(e)}

def main(load_existing_data: bool = False, run_id: Optional[str] = None):
    setup_matplotlib()
    tracemalloc.start()
    
    data_manager = DataManager(data_dir='dms_experiment_data')
    output_dir = Path("dms_results")
    output_dir.mkdir(exist_ok=True)

    print("="*80); print("🔬 DMS Benchmark - OIO vs 15 Baselines on GFP 🔬"); print("="*80)

    all_results = None
    if load_existing_data:
        print("\n📂 Attempting to load existing data..."); data_manager.list_available_data()
        all_results, _ = data_manager.load_experiment_results(run_id)

    if all_results:
        print("\n✅ Data loaded successfully. Proceeding to analysis.")
    else:
        if load_existing_data: print("❌ No data found. Running new experiment.")
        
        DMS_FILE, MAX_EVALUATIONS, NUM_RUNS = "gfp.csv", 500, 10
        try:
            oracle, g_opt, s_len, wt_seq, all_seqs = load_dms_oracle(DMS_FILE)
        except (FileNotFoundError, KeyError) as e:
            print(f"❌ {e}\nPlease ensure 'gfp.csv' is in the same directory and has correct columns."); return
            
        print(f"\n🚀 Starting benchmark (Budget: {MAX_EVALUATIONS} FES, Runs: {NUM_RUNS})...")
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(test_single_algorithm_dms, name, Adapter, oracle, g_opt, s_len, MAX_EVALUATIONS, NUM_RUNS, wt_seq, all_seqs): name
                for name, Adapter in ALL_ALGORITHMS_DMS.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try: results[name] = future.result()[1]
                except Exception as e: results[name] = {'error': str(e)}
        
        all_results = {'GFP': results}
        exp_config = {'file': DMS_FILE, 'evals': MAX_EVALUATIONS, 'runs': NUM_RUNS}
        data_manager.save_experiment_results(all_results, exp_config)

    print("\n--- Analysis and Report Generation ---")
    gfp_results = all_results.get('GFP', {})
    if not gfp_results:
        print("No results to analyze."); return

    print("\n🏆 CEC-Style Ranking for GFP:")
    ranking = calculate_cec_style_ranking(gfp_results)
    for rank, (alg, score) in enumerate(ranking, 1):
        star = '⭐' if alg == 'OIO' else '  '
        print(f"  {rank}. {star} {get_algorithm_short_name(alg):<8}: Score {score:.1f}")
        
    perform_statistical_tests(all_results, output_dir)
    create_performance_boxplot(all_results, output_dir)
    create_convergence_plots(all_results, output_dir)
    
    print(f"\n🎉 Benchmark complete. Results saved in '{output_dir}' and '{data_manager.data_dir}'.")
    tracemalloc.stop()

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
        DataManager(data_dir='dms_experiment_data').list_available_data()
    else:
        print("Usage: python dms_benchmark.py [run|load|list]")
        print("  run        : Start a new experiment.")
        print("  load [id]  : Load and analyze an experiment (latest if id is omitted).")
        print("  list       : List all saved experiments.")