#!/usr/bin/env python3
"""
CEC2022基准测试脚本 - OIO vs 15种基线算法
测试F1~F12函数，16线程并行，20000次评估，重复10次
包含可视化分析和统计检验
"""

import time
import numpy as np
import pandas as pd
import gc
import tracemalloc
from typing import Callable, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime

# 添加父目录到路径以导入OIO和基线算法
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from opfunu.cec_based.cec2022 import (
        F12022, F22022, F32022, F42022, F52022, F62022,
        F72022, F82022, F92022, F102022, F112022, F122022
    )
    print("✅ 成功导入opfunu CEC2022函数")
except ImportError:
    print("❌ 错误: 需要安装opfunu库")
    print("请运行: pip install opfunu")
    sys.exit(1)

from OIO import ChampionOIOAlgorithm
from baseline import ALGORITHMS

# ============================================================================
# 早停机制 - 异常中断模式
# ============================================================================

class TargetPrecisionReached(Exception):
    """用于在达到目标精度时中断算法的自定义异常"""
    def __init__(self, message, fitness, fes):
        super().__init__(message)
        self.fitness = fitness
        self.fes = fes

class InterruptibleCostFunc:
    """一个可中断的成本函数包装器，并记录全局收敛历史"""
    def __init__(self, original_cost_func, max_evaluations, success_threshold=1e-8):
        self.original_cost_func = original_cost_func
        self.max_evaluations = max_evaluations
        self.success_threshold = success_threshold
        self.evaluation_count = 0
        self.best_fitness = float('inf')

        # --- 修改：收敛历史记录 (FES, fitness) 元组 ---
        self.convergence_history = []  # 现在存储 (FES, fitness) 元组
        # 为了避免记录过多数据点，我们可以设置一个记录间隔
        # 每评估 record_interval 次，记录一次当前的最优值
        self.record_interval = max(1, max_evaluations // 500)  # 记录500个点
        # --- 修改结束 ---

    def __call__(self, x):
        # 检查1：是否已达到最大评估次数
        if self.evaluation_count >= self.max_evaluations:
            # 抛出异常，表示评估次数耗尽
            raise RuntimeError("Maximum evaluations reached")

        self.evaluation_count += 1
        fitness = self.original_cost_func(x)

        new_best_found = False
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            new_best_found = True

        # --- 修改：记录收敛历史 (FES, fitness) 元组 ---
        # 每隔一定间隔，或者找到了新的最优值时，记录一次
        if new_best_found or self.evaluation_count % self.record_interval == 0:
            # 记录当前的评估次数和对应的最优值
            self.convergence_history.append((self.evaluation_count, self.best_fitness))
        # --- 修改结束 ---

        # 检查2：是否达到目标精度
        # 注意：CEC2022的理论最优值是0
        error = fitness - 0.0
        if error < self.success_threshold:
            # --- 修改：在抛出异常前，确保最后的最优值被记录 ---
            if not self.convergence_history or self.convergence_history[-1][1] != self.best_fitness:
                self.convergence_history.append((self.evaluation_count, self.best_fitness))
            # --- 修改结束 ---
            # 抛出自定义异常，携带成功信息
            raise TargetPrecisionReached(
                "Target precision reached",
                fitness=self.best_fitness,
                fes=self.evaluation_count
            )

        return fitness

def get_memory_usage():
    """获取当前内存使用量(MB) - 使用tracemalloc"""
    try:
        current, _ = tracemalloc.get_traced_memory()
        return current / 1024 / 1024  # 转换为MB
    except:
        return 0.0

# ============================================================================
# 数据管理模块
# ============================================================================

class DataManager:
    """数据管理器 - 负责保存和加载实验数据"""

    def __init__(self, data_dir='experiment_data'):
        """初始化数据管理器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # 创建子目录
        self.convergence_dir = self.data_dir / 'convergence_data'
        self.results_dir = self.data_dir / 'results_data'
        self.plots_dir = self.data_dir / 'plot_data'

        for dir_path in [self.convergence_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)

    def save_convergence_data(self, func_name, algorithm_name, convergence_history, run_id=None):
        """保存收敛曲线数据

        Args:
            func_name: 函数名称 (如 'F1')
            algorithm_name: 算法名称 (如 'OIO')
            convergence_history: 收敛历史数据 (list)
            run_id: 运行ID，如果为None则自动生成时间戳
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{func_name}_{algorithm_name}_{run_id}.json"
        filepath = self.convergence_dir / filename

        data = {
            'function': func_name,
            'algorithm': algorithm_name,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'convergence_history': convergence_history
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_convergence_data(self, func_name, algorithm_name, run_id=None):
        """加载收敛曲线数据

        Args:
            func_name: 函数名称
            algorithm_name: 算法名称
            run_id: 运行ID，如果为None则加载最新的数据

        Returns:
            convergence_history: 收敛历史数据，如果未找到返回None
        """
        if run_id is not None:
            filename = f"{func_name}_{algorithm_name}_{run_id}.json"
            filepath = self.convergence_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data['convergence_history']
        else:
            # 查找最新的数据文件
            pattern = f"{func_name}_{algorithm_name}_*.json"
            files = list(self.convergence_dir.glob(pattern))
            if files:
                # 按修改时间排序，取最新的
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data['convergence_history']

        return None

    def save_experiment_results(self, all_results, experiment_config, run_id=None):
        """保存完整的实验结果

        Args:
            all_results: 所有实验结果
            experiment_config: 实验配置信息
            run_id: 运行ID
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"experiment_results_{run_id}.pkl"
        filepath = self.results_dir / filename

        data = {
            'results': all_results,
            'config': experiment_config,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        # 同时保存一个JSON版本（不包含复杂对象）
        json_filename = f"experiment_results_{run_id}.json"
        json_filepath = self.results_dir / json_filename

        # 转换结果为JSON可序列化格式
        json_results = self._convert_results_to_json(all_results)
        json_data = {
            'results': json_results,
            'config': experiment_config,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat()
        }

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def load_experiment_results(self, run_id=None):
        """加载实验结果

        Args:
            run_id: 运行ID，如果为None则加载最新的结果

        Returns:
            (all_results, experiment_config): 实验结果和配置
        """
        if run_id is not None:
            filename = f"experiment_results_{run_id}.pkl"
            filepath = self.results_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                return data['results'], data['config']
        else:
            # 查找最新的结果文件
            pattern = "experiment_results_*.pkl"
            files = list(self.results_dir.glob(pattern))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'rb') as f:
                    data = pickle.load(f)
                return data['results'], data['config']

        return None, None

    def _convert_results_to_json(self, all_results):
        """将结果转换为JSON可序列化格式"""
        json_results = {}
        for func_name, func_results in all_results.items():
            json_results[func_name] = {}
            for alg_name, alg_result in func_results.items():
                if 'error' in alg_result:
                    json_results[func_name][alg_name] = {'error': alg_result['error']}
                else:
                    json_results[func_name][alg_name] = {
                        'fitness_mean': float(alg_result.get('fitness_mean', 0)),
                        'fitness_std': float(alg_result.get('fitness_std', 0)),
                        'fitness_best': float(alg_result.get('fitness_best', 0)),
                        'fitness_worst': float(alg_result.get('fitness_worst', 0)),
                        'time_mean': float(alg_result.get('time_mean', 0)),
                        'success_count': int(alg_result.get('success_count', 0)),
                        'total_runs': int(alg_result.get('total_runs', 0)),
                        'all_fitness': [float(x) for x in alg_result.get('all_fitness', [])],
                        'all_times': [float(x) for x in alg_result.get('all_times', [])],
                        'convergence_histories': alg_result.get('convergence_histories', [])
                    }
        return json_results

    def list_available_data(self):
        """列出可用的数据文件"""
        print(f"\n📁 可用的实验数据 (存储在 {self.data_dir}):")

        # 列出实验结果文件
        result_files = list(self.results_dir.glob("experiment_results_*.pkl"))
        if result_files:
            print(f"\n🔬 实验结果文件 ({len(result_files)} 个):")
            for file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                print(f"  - {file.name} (修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

        # 列出收敛数据文件
        conv_files = list(self.convergence_dir.glob("*.json"))
        if conv_files:
            print(f"\n📈 收敛数据文件 ({len(conv_files)} 个):")
            # 按函数和算法分组
            by_func_alg = {}
            for file in conv_files:
                parts = file.stem.split('_')
                if len(parts) >= 3:
                    func_name = parts[0]
                    alg_name = parts[1]
                    key = f"{func_name}_{alg_name}"
                    if key not in by_func_alg:
                        by_func_alg[key] = []
                    by_func_alg[key].append(file)

            for key, files in sorted(by_func_alg.items()):
                print(f"  - {key}: {len(files)} 个运行记录")

    def save_plot_data(self, plot_type, data, filename=None):
        """保存绘图数据

        Args:
            plot_type: 绘图类型 ('histogram', 'convergence', 'statistical')
            data: 绘图数据
            filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{plot_type}_data_{timestamp}.json"

        filepath = self.plots_dir / filename

        plot_data = {
            'plot_type': plot_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, indent=2, ensure_ascii=False)

        return filepath

    def load_plot_data(self, plot_type, filename=None):
        """加载绘图数据

        Args:
            plot_type: 绘图类型
            filename: 文件名，如果为None则加载最新的

        Returns:
            绘图数据，如果未找到返回None
        """
        if filename is not None:
            filepath = self.plots_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data['data']
        else:
            # 查找最新的绘图数据
            pattern = f"{plot_type}_data_*.json"
            files = list(self.plots_dir.glob(pattern))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data['data']

        return None

# ============================================================================
# CEC2022函数适配器
# ============================================================================

class CEC2022FunctionAdapter:
    """CEC2022函数适配器，将连续优化函数适配为基线算法接口"""

    def __init__(self, cec_func, dim):
        self.cec_func = cec_func
        self.dim = dim
        self.bounds = cec_func.bounds
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

    def __call__(self, x):
        """评估函数值 - 适配基线算法接口"""
        # 基线算法期望最大化问题，CEC2022是最小化问题，所以取负值
        # 确保x在边界内
        x = np.clip(x, self.lb, self.ub)
        return -self.cec_func.evaluate(x)  # 取负值转换为最大化问题

    def evaluate(self, x):
        """为OIO算法提供evaluate方法接口"""
        # OIO算法期望最小化问题，直接返回原始函数值
        x = np.clip(x, self.lb, self.ub)
        return self.cec_func.evaluate(x)

# ============================================================================
# 算法包装器基类
# ============================================================================

from abc import ABC, abstractmethod

class AlgorithmWrapperBase(ABC):
    """所有算法包装器的抽象基类，统一接口"""

    def __init__(self, cost_func, dim, bounds, max_evaluations=20000):
        """
        统一的初始化接口

        Args:
            cost_func: 成本函数（InterruptibleCostFunc实例）
            dim: 问题维度
            bounds: 边界，可以是tuple (min_bound, max_bound) 或 array-like
            max_evaluations: 最大评估次数
        """
        self.cost_func = cost_func
        self.dim = dim
        self.max_evaluations = max_evaluations

        # 统一边界格式处理
        if isinstance(bounds, tuple) and len(bounds) == 2:
            # 如果是 (min_val, max_val) 格式
            self.bounds = bounds
        elif hasattr(bounds, '__len__') and len(bounds) == 2:
            # 如果是 [min_val, max_val] 格式
            self.bounds = (bounds[0], bounds[1])
        else:
            raise ValueError(f"Unsupported bounds format: {bounds}")

    @abstractmethod
    def optimize(self):
        """
        执行优化

        Returns:
            tuple: (best_position, best_value) 或 (None, best_value)
        """
        pass

# ============================================================================
# 基线算法适配器
# ============================================================================

class BaselineAlgorithmAdapter(AlgorithmWrapperBase):
    """基线算法适配器，将离散优化算法适配到连续优化问题，使用异常中断模式"""

    def __init__(self, BaselineClass, cost_func, fitness_func_adapter, sequence_length, max_evaluations=20000):
        # 获取边界信息
        if hasattr(fitness_func_adapter, 'lb') and hasattr(fitness_func_adapter, 'ub'):
            bounds = (fitness_func_adapter.lb[0], fitness_func_adapter.ub[0])  # 假设所有维度边界相同
            self.lb = np.array(fitness_func_adapter.lb)
            self.ub = np.array(fitness_func_adapter.ub)
        else:
            # 默认边界
            bounds = (-100.0, 100.0)
            self.lb = np.full(sequence_length, -100.0)
            self.ub = np.full(sequence_length, 100.0)

        # 调用父类初始化
        super().__init__(cost_func, sequence_length, bounds, max_evaluations)

        # 基线算法特有的属性
        self.BaselineClass = BaselineClass
        self.fitness_func_adapter = fitness_func_adapter
        self.sequence_length = sequence_length

    def _adapter_func_max(self, discrete_sequence):
        """将离散序列转换为连续值并评估，返回最大化值"""
        # 1. 转换：将[0,1]的离散序列映射到连续空间
        continuous_x = self.lb + discrete_sequence * (self.ub - self.lb)
        # 2. 调用可中断的函数（返回最小化值）
        min_fitness = self.cost_func(continuous_x)
        # 3. 返回最大化值（基线算法期望最大化问题）
        return -min_fitness

    def optimize(self):
        """使用基线算法优化，让异常自然地向上冒泡"""
        # 将 _adapter_func_max 传递给基线算法
        algorithm = self.BaselineClass(
            fitness_func=self._adapter_func_max,
            sequence_length=self.sequence_length,
            max_evaluations=self.max_evaluations
        )

        # 正常运行，让异常自然地向上冒泡
        _, best_fitness_max = algorithm.optimize()

        # 如果能运行到这里，说明没有中断
        true_fitness = -best_fitness_max

        return None, true_fitness

# ============================================================================
# OIO算法包装器
# ============================================================================

# ============================================================================
# OIO算法包装器
# ============================================================================

class OIOWrapper(AlgorithmWrapperBase):
    """OIO算法包装器，使用异常中断模式"""

    def __init__(self, cost_func, dim, bounds, max_evaluations=20000):
        # 调用父类初始化，统一接口
        super().__init__(cost_func, dim, bounds, max_evaluations)

    def optimize(self):
        """使用OIO算法优化，让异常自然地向上冒泡"""
        oio = ChampionOIOAlgorithm(
            cost_func=self.cost_func,
            bounds=self.bounds,
            dim=self.dim
        )

        # 计算迭代次数
        max_iter = max(10, self.max_evaluations // 200)  # 使用统一的最大评估次数

        # 正常运行，让异常自然地向上冒泡
        best_pos, best_val = oio.optimize(max_iter=max_iter)

        return best_pos, best_val

# ============================================================================
# 合并所有算法
# ============================================================================

# 合并OIO和所有基线算法
ALL_CEC_ALGORITHMS = {
    'OIO': OIOWrapper,
    **ALGORITHMS  # 包含所有15种基线算法
}

def test_single_algorithm_cec(alg_name, AlgorithmClass, func_adapter, max_evaluations=20000, num_runs=10, data_manager=None, func_name=None):
    """测试单个算法在CEC2022函数上的性能 - 在单个线程中运行多次，支持早停机制"""
    try:
        fitness_results = []
        time_results = []
        fes_results = []  # 新增：FES结果
        error_results = []  # 新增：误差结果
        convergence_histories = []
        success_threshold = 1e-8  # CEC2022成功阈值

        print(f"🔄 线程开始测试 {alg_name} ({num_runs}次运行)...")

        for run in range(num_runs):
            # 为每次运行创建独立的函数适配器副本
            run_func_adapter = CEC2022FunctionAdapter(func_adapter.cec_func, func_adapter.dim)

            # 创建可中断的成本函数实例
            interruptible_func = InterruptibleCostFunc(
                original_cost_func=run_func_adapter.evaluate,  # OIO和基线都用最小化接口
                max_evaluations=max_evaluations,
                success_threshold=success_threshold
            )

            # 创建算法实例，并传入可中断的函数
            if alg_name == 'OIO':
                algorithm = AlgorithmClass(
                    cost_func=interruptible_func,
                    dim=func_adapter.dim,
                    bounds=(run_func_adapter.lb[0], run_func_adapter.ub[0]),
                    max_evaluations=max_evaluations
                )
            else:
                algorithm = BaselineAlgorithmAdapter(
                    BaselineClass=AlgorithmClass,
                    cost_func=interruptible_func,  # 基线算法也使用
                    fitness_func_adapter=run_func_adapter,  # 仍然需要它来获取边界
                    sequence_length=func_adapter.dim,
                    max_evaluations=max_evaluations
                )

            start_time = time.time()
            best_fitness, fes_consumed = (float('inf'), max_evaluations)  # 默认值

            try:
                # 运行优化
                _, best_fitness = algorithm.optimize()  # 不再需要从这里返回FES
                # 如果正常结束，FES就是计数器的值
                fes_consumed = interruptible_func.evaluation_count
            except TargetPrecisionReached as e:
                # 如果成功中断
                print(f"  ⚡️ {alg_name} 在第 {run+1} 次运行提前终止于 FES={e.fes}")
                best_fitness = e.fitness
                fes_consumed = e.fes
            except RuntimeError as e:
                # 如果是评估次数耗尽
                best_fitness = interruptible_func.best_fitness
                fes_consumed = interruptible_func.evaluation_count
            except Exception:
                # 捕获其他算法内部错误
                raise

            end_time = time.time()

            run_time = end_time - start_time

            # 确保best_fitness是数值类型
            if isinstance(best_fitness, np.ndarray):
                best_fitness = float(best_fitness.item())
            elif not isinstance(best_fitness, (int, float)):
                best_fitness = float(best_fitness)

            # 计算误差 (CEC2022函数的理论最优值通常是0)
            optimal_value = 0.0
            error = best_fitness - optimal_value

            fitness_results.append(best_fitness)
            time_results.append(run_time)
            fes_results.append(fes_consumed)  # 新增
            error_results.append(error)  # 新增

            # --- 修改：从 interruptible_func 获取收敛历史 ---
            convergence_history = interruptible_func.convergence_history.copy()

            if convergence_history:
                print(f"  {alg_name}: 成功获取收敛历史，长度 = {len(convergence_history)}")
                convergence_histories.append(convergence_history)

                # 保存单次运行的收敛数据
                if data_manager is not None and func_name is not None:
                    run_id = f"run_{run+1}_{datetime.now().strftime('%H%M%S')}"
                    data_manager.save_convergence_data(func_name, alg_name, convergence_history, run_id)
            else:
                print(f"  {alg_name}: 未生成收敛历史")
            # --- 修改结束 ---

            # 每5次运行报告一次进度
            if (run + 1) % 5 == 0:
                print(f"  {alg_name}: 完成 {run + 1}/{num_runs} 次运行")

            # 清理
            del algorithm
            del run_func_adapter
            gc.collect()

        # 计算统计信息
        fitness_array = np.array(fitness_results)
        time_array = np.array(time_results)
        fes_array = np.array(fes_results)
        error_array = np.array(error_results)

        # 计算成功次数 (误差小于阈值)
        success_count = np.sum(error_array < success_threshold)

        result = {
            'fitness_mean': np.mean(fitness_array),
            'fitness_std': np.std(fitness_array),
            'fitness_best': np.min(fitness_array),
            'fitness_worst': np.max(fitness_array),
            'time_mean': np.mean(time_array),
            'success_count': success_count,  # 修改：实际成功次数
            'total_runs': num_runs,
            'all_fitness': fitness_results.copy(),
            'all_times': time_results.copy(),
            'all_fes': fes_results.copy(),  # 新增：所有FES记录
            'all_errors': error_results.copy(),  # 新增：所有误差记录
            'convergence_histories': convergence_histories.copy() if convergence_histories else []
        }

        print(f"✅ {alg_name} 线程完成: 最优={result['fitness_best']:.6e}, 平均={result['fitness_mean']:.6e}±{result['fitness_std']:.6e}")

        return alg_name, result

    except Exception as e:
        print(f"❌ {alg_name} 线程失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return alg_name, {'error': str(e)}

def calculate_cec_style_ranking(func_results, success_threshold=1e-8):
    """
    根据CEC竞赛规则对单个函数的结果进行排名

    Args:
        func_results: 单个函数的所有算法结果
        success_threshold: 成功阈值

    Returns:
        list: 排名结果 [(算法名, 总分), ...]，按总分升序排列
    """
    all_runs = []

    # 1. 汇集所有算法的所有运行结果
    for alg_name, result_data in func_results.items():
        if 'error' in result_data:
            continue

        if 'all_errors' in result_data and 'all_fes' in result_data:
            errors = result_data['all_errors']
            fes_list = result_data['all_fes']

            for i in range(len(errors)):
                error = errors[i]
                fes = fes_list[i]
                is_success = error < success_threshold

                all_runs.append({
                    "alg_name": alg_name,
                    "success": is_success,
                    "error": error,
                    "fes": fes
                })

    if not all_runs:
        return []

    # 2. 根据CEC规则排序
    # 规则1: 成功的排在失败的前面 (success=True排在前面)
    # 规则2: 如果都成功, FES少的排在前面
    # 规则3: 如果都失败, error小的排在前面
    sorted_runs = sorted(all_runs, key=lambda x: (
        not x['success'],  # 成功的排在前面
        x['fes'] if x['success'] else x['error']  # 成功看FES，失败看error
    ))

    # 3. 分配排名并计算总分
    algorithm_names = set(run['alg_name'] for run in all_runs)
    ranks = {alg_name: 0 for alg_name in algorithm_names}

    for i, run in enumerate(sorted_runs):
        ranks[run['alg_name']] += (i + 1)  # 排名从1开始

    # 4. 按总分排序 (分数越低越好)
    final_ranking = sorted(ranks.items(), key=lambda x: x[1])
    return final_ranking

def create_statistical_summary_table(statistical_results, output_dir):
    """创建一个 Win-Loss-Tie 总结表"""
    if not statistical_results:
        return

    df = pd.DataFrame(statistical_results)
    summary = {}

    for _, row in df.iterrows():
        alg = row['Baseline_Algorithm']
        if alg not in summary:
            summary[alg] = {'Win': 0, 'Loss': 0, 'Tie': 0}

        # OIO的p-value < 0.05 意味着OIO显著更好 (Win)
        if row['P_Value'] < 0.05 and row['OIO_Mean'] < row['Baseline_Mean']:
            summary[alg]['Win'] += 1
        # 如果OIO显著更差
        elif row['P_Value'] < 0.05 and row['OIO_Mean'] > row['Baseline_Mean']:
            summary[alg]['Loss'] += 1
        else:
            # 无法拒绝原假设，认为持平 (Tie)
            summary[alg]['Tie'] += 1

    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.index.name = 'Algorithm vs OIO'
    print("\n📊 OIO vs 基线算法统计总结 (Win-Loss-Tie):")
    print(summary_df.to_string())
    summary_df.to_csv(f'{output_dir}/statistical_summary.csv')

def setup_matplotlib():
    """设置matplotlib参数"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

def create_performance_boxplot(all_results, cec_functions, output_dir='results', data_manager=None):
    """创建性能对比箱线图"""
    Path(output_dir).mkdir(exist_ok=True)

    # 获取所有出现的算法名称
    all_algorithms = set()
    for func_results in all_results.values():
        all_algorithms.update(func_results.keys())

    # 收集每个算法在所有函数上的性能数据（用于箱线图）
    algorithm_all_data = {}  # 存储所有原始数据点
    algorithm_performance = {}  # 存储平均性能
    algorithm_std = {}  # 存储标准差

    for alg_name in all_algorithms:
        all_fitness_values = []  # 收集所有函数的所有运行结果
        all_means = []  # 收集所有函数的平均值

        for func_name in cec_functions.keys():
            if (alg_name in all_results[func_name] and
                'error' not in all_results[func_name][alg_name]):

                result = all_results[func_name][alg_name]

                # 收集该函数的平均值
                if 'fitness_mean' in result:
                    all_means.append(result['fitness_mean'])

                # 收集该函数的所有运行结果
                if 'all_fitness' in result and result['all_fitness']:
                    all_fitness_values.extend(result['all_fitness'])

        if all_means:
            algorithm_performance[alg_name] = np.mean(all_means)
            algorithm_std[alg_name] = np.std(all_means)
            algorithm_all_data[alg_name] = all_fitness_values

    # 保存箱线图数据
    boxplot_data = {
        'algorithm_performance': algorithm_performance,
        'algorithm_std': algorithm_std,
        'algorithm_all_data': algorithm_all_data,
        'functions': list(cec_functions.keys())
    }

    if data_manager is not None:
        data_manager.save_plot_data('boxplot', boxplot_data)

    # 找到最优算法（最小值）
    best_alg = min(algorithm_performance.keys(), key=lambda x: algorithm_performance[x])

    # 准备数据，OIO放在最左边
    algorithms = ['OIO'] + [alg for alg in sorted(algorithm_performance.keys()) if alg != 'OIO']

    # 准备箱线图数据
    boxplot_datasets = []
    for alg in algorithms:
        if alg in algorithm_all_data and algorithm_all_data[alg]:
            boxplot_datasets.append(algorithm_all_data[alg])
        else:
            # 如果没有原始数据，使用平均值创建单点数据
            boxplot_datasets.append([algorithm_performance[alg]])

    # 创建算法简称映射（用于显示）
    def get_algorithm_display_name(alg_name):
        """获取算法的显示名称（简称）"""
        # 如果算法名称已经是简称（长度<=4），直接使用
        if len(alg_name) <= 4:
            return alg_name

        # 常见算法简称映射
        name_mapping = {
            'GeneticAlgorithm': 'GA',
            'HillClimbing': 'HC',
            'SimulatedAnnealing': 'SA',
            'DifferentialEvolution': 'DE',
            'ParticleSwarmOptimization': 'PSO',
            'ParticleSwarm': 'PSO',
            'HarrisHawksOptimization': 'HHO',
            'WhaleOptimization': 'WOA',
            'HybridPSOGWO': 'PSO-GWO',
            'AdaptiveHarrisHawks': 'AHHO',
            'HippopotamusOptimization': 'HO',
            'WalrusOptimization': 'WO',
            'CrestedPorcupineOptimizer': 'CPO',
            'ElkHerdOptimizer': 'EHO',
            'GreylagGooseOptimization': 'GGO',
            'QuokkaSwarmOptimization': 'QSO',
            'AntColony': 'ACO',
            'TabuSearch': 'TS'
        }

        return name_mapping.get(alg_name, alg_name[:4].upper())

    # 获取显示用的算法名称
    display_algorithms = [get_algorithm_display_name(alg) for alg in algorithms]

    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))

    # 创建箱线图
    box_plot = ax.boxplot(boxplot_datasets,
                         patch_artist=True,  # 允许填充颜色
                         showmeans=True,     # 显示均值
                         meanline=True,      # 均值用线表示
                         notch=True,         # 显示置信区间
                         whis=1.5)          # 异常值检测范围

    # 设置颜色：最优算法用深橙色，其他算法用深青色
    best_color = '#FF6B35'    # 深橙色 - 突出显示最优算法
    other_color = '#4A90A4'   # 深青色 - 专业且易读

    # 为每个箱子设置颜色
    for i, (patch, alg) in enumerate(zip(box_plot['boxes'], algorithms)):
        if alg == best_alg:
            patch.set_facecolor(best_color)
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(other_color)
            patch.set_alpha(0.7)

    # 设置其他元素的颜色
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', alpha=0.8)

    # 设置均值线的颜色
    plt.setp(box_plot['means'], color='red', linewidth=2)

    # 设置标签和标题
    ax.set_xlabel('Algorithms', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness Value', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison Across All CEC2022 Functions (Box Plot)', fontsize=16, fontweight='bold')

    # 设置x轴标签
    ax.set_xticks(range(1, len(algorithms) + 1))
    ax.set_xticklabels(display_algorithms, rotation=45, ha='right')

    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')

    # 使用对数刻度（如果数据范围很大）
    ax.set_yscale('log')

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=best_color, alpha=0.7, label='Best Algorithm'),
        Patch(facecolor=other_color, alpha=0.7, label='Other Algorithms'),
        Line2D([0], [0], color='red', linewidth=2, label='Mean Value')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 性能对比箱线图已保存: {output_dir}/performance_comparison_boxplot.png")

def create_convergence_plots(all_results, cec_functions, output_dir='results'):
    """创建收敛曲线图，横坐标为FES，不填充早停曲线"""
    convergence_dir = Path(output_dir) / 'convergence_curves'
    convergence_dir.mkdir(parents=True, exist_ok=True)

    for func_name in cec_functions.keys():
        if func_name not in all_results:
            continue

        # 获取该函数的所有算法
        func_algorithms = list(all_results[func_name].keys())
        if not func_algorithms:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))

        # 为每个算法绘制收敛曲线
        import matplotlib.cm as cm

        # 为16种算法准备足够的颜色
        # 使用多个颜色映射组合，确保有足够的区分度
        def get_algorithm_colors(n_algorithms):
            """为n个算法生成足够区分的颜色"""
            if n_algorithms <= 20:
                # 使用tab20颜色映射，最多支持20种颜色
                return cm.get_cmap('tab20')(np.linspace(0, 1, n_algorithms))
            else:
                # 如果超过20种，组合多个颜色映射
                colors1 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
                colors2 = cm.get_cmap('Set3')(np.linspace(0, 1, n_algorithms - 20))
                return np.vstack([colors1, colors2])

        colors = get_algorithm_colors(len(func_algorithms))

        # 创建算法简称映射函数（与直方图中相同）
        def get_algorithm_display_name(alg_name):
            """获取算法的显示名称（简称）"""
            if len(alg_name) <= 4:
                return alg_name

            name_mapping = {
                'GeneticAlgorithm': 'GA',
                'HillClimbing': 'HC',
                'SimulatedAnnealing': 'SA',
                'DifferentialEvolution': 'DE',
                'ParticleSwarmOptimization': 'PSO',
                'ParticleSwarm': 'PSO',
                'HarrisHawksOptimization': 'HHO',
                'WhaleOptimization': 'WOA',
                'HybridPSOGWO': 'PSO-GWO',
                'AdaptiveHarrisHawks': 'AHHO',
                'HippopotamusOptimization': 'HO',
                'WalrusOptimization': 'WO',
                'CrestedPorcupineOptimizer': 'CPO',
                'ElkHerdOptimizer': 'EHO',
                'GreylagGooseOptimization': 'GGO',
                'QuokkaSwarmOptimization': 'QSO',
                'AntColony': 'ACO',
                'TabuSearch': 'TS'
            }

            return name_mapping.get(alg_name, alg_name[:4].upper())

        # 为算法分配颜色索引，确保OIO始终使用固定颜色
        color_indices = {}
        oio_color = '#FF6B35'  # OIO专用颜色
        other_color_idx = 0

        for alg_name in func_algorithms:
            if alg_name == 'OIO':
                color_indices[alg_name] = 'OIO'  # 特殊标记
            else:
                color_indices[alg_name] = other_color_idx
                other_color_idx += 1

        # 重新排序算法列表，确保OIO最后绘制（显示在最上层）
        other_algorithms = [alg for alg in func_algorithms if alg != 'OIO']
        ordered_algorithms = other_algorithms + (['OIO'] if 'OIO' in func_algorithms else [])

        for alg_name in ordered_algorithms:
            if (alg_name in all_results[func_name] and
                'error' not in all_results[func_name][alg_name] and
                all_results[func_name][alg_name].get('convergence_histories')):

                # histories 是一个列表，每个元素是一次运行的收敛历史
                # 例如: [[(fes, fit), (fes, fit), ...], [(fes, fit), ...]]
                histories = all_results[func_name][alg_name]['convergence_histories']

                if histories and histories[0]: # 确保历史记录不为空
                    # --- 核心修改开始 ---

                    # 1. 提取所有运行的FES和Fitness数据
                    all_fes_data = [np.array([point[0] for point in run]) for run in histories]
                    all_fit_data = [np.array([point[1] for point in run]) for run in histories]

                    # 2. 创建一个统一的FES插值轴
                    #    找到所有运行中FES的最大值，并创建一个公共的FES轴
                    max_fes_overall = max(fes_run[-1] for fes_run in all_fes_data if len(fes_run) > 0)
                    # 我们创建500个插值点
                    common_fes_axis = np.linspace(1, max_fes_overall, 500)

                    # 3. 对每次运行的收敛曲线进行插值
                    #    这样，所有曲线都在相同的FES点上有了对应的fitness值
                    interpolated_fits = []
                    for fes_run, fit_run in zip(all_fes_data, all_fit_data):
                        if len(fes_run) > 1:
                            # np.interp 要求x坐标单调递增
                            interp_fit = np.interp(common_fes_axis, fes_run, fit_run)
                            interpolated_fits.append(interp_fit)

                    if not interpolated_fits: continue # 如果没有可插值的数据，则跳过

                    # 4. 计算插值后曲线的平均值和标准差
                    mean_history = np.mean(interpolated_fits, axis=0)
                    std_history = np.std(interpolated_fits, axis=0)

                    # 5. 找到该算法实际停止的平均FES
                    #    我们只将曲线画到这个点为止
                    avg_stop_fes = np.mean([fes_run[-1] for fes_run in all_fes_data if len(fes_run) > 0])

                    # 6. 截断公共FES轴和平均曲线
                    plot_mask = common_fes_axis <= avg_stop_fes
                    plot_fes_axis = common_fes_axis[plot_mask]
                    plot_mean_history = mean_history[plot_mask]
                    plot_std_history = std_history[plot_mask]

                    # --- 核心修改结束 ---

                    # 分配颜色：OIO用固定颜色，其他算法用不同颜色
                    if alg_name == 'OIO':
                        color = oio_color
                        linewidth = 3
                        alpha = 0.9
                    else:
                        color_idx = color_indices[alg_name]
                        color = colors[color_idx] if color_idx < len(colors) else colors[color_idx % len(colors)]
                        linewidth = 2
                        alpha = 0.7

                    # 使用简称作为图例标签
                    display_name = get_algorithm_display_name(alg_name)

                    # 使用新的FES轴和截断后的数据进行绘图
                    ax.plot(plot_fes_axis, plot_mean_history, color=color, linewidth=linewidth,
                           label=display_name, alpha=alpha)

                    # 添加标准差阴影
                    ax.fill_between(plot_fes_axis,
                                   plot_mean_history - plot_std_history,
                                   plot_mean_history + plot_std_history,
                                   color=color, alpha=0.15)

        # --- 修改坐标轴标签 ---
        ax.set_xlabel('Function Evaluations (FES)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Best Fitness Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Convergence Curves for {func_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 优化图例显示，适应16种算法
        n_algorithms = len([alg for alg in ordered_algorithms
                           if (alg in all_results[func_name] and
                               'error' not in all_results[func_name][alg] and
                               all_results[func_name][alg].get('convergence_histories'))])

        # 获取图例句柄和标签，重新排序使OIO显示在最前面
        handles, labels = ax.get_legend_handles_labels()

        # 重新排序：OIO在前，其他算法按字母顺序
        oio_items = [(h, l) for h, l in zip(handles, labels) if l == 'OIO']
        other_items = [(h, l) for h, l in zip(handles, labels) if l != 'OIO']
        other_items.sort(key=lambda x: x[1])  # 按标签排序

        # 合并：OIO在前
        ordered_items = oio_items + other_items
        ordered_handles = [item[0] for item in ordered_items]
        ordered_labels = [item[1] for item in ordered_items]

        if n_algorithms <= 8:
            # 算法较少时，图例放在右侧
            ax.legend(ordered_handles, ordered_labels,
                     bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            # 算法较多时，图例放在下方，分多列显示
            ncol = min(4, (n_algorithms + 3) // 4)  # 最多4列
            ax.legend(ordered_handles, ordered_labels,
                     bbox_to_anchor=(0.5, -0.15), loc='upper center',
                     ncol=ncol, fontsize=9)

        ax.set_yscale('log')  # 使用对数刻度更好地显示收敛过程

        plt.tight_layout()
        plt.savefig(f'{convergence_dir}/{func_name}_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"📈 收敛曲线图已保存到: {convergence_dir}/ (横坐标为FES)")

def perform_statistical_tests(all_results, cec_functions, output_dir='results'):
    """执行统计检验"""
    Path(output_dir).mkdir(exist_ok=True)

    statistical_results = []

    print("\n🔬 执行统计检验 (Mann-Whitney U检验)...")

    for func_name in cec_functions.keys():
        if 'OIO' not in all_results[func_name] or 'error' in all_results[func_name]['OIO']:
            continue

        oio_data = all_results[func_name]['OIO']
        if 'all_fitness' not in oio_data:
            continue

        oio_results = oio_data['all_fitness']

        # 获取该函数的所有算法
        for alg_name in all_results[func_name].keys():
            if alg_name == 'OIO':
                continue

            if ('error' not in all_results[func_name][alg_name] and
                'all_fitness' in all_results[func_name][alg_name]):

                baseline_results = all_results[func_name][alg_name]['all_fitness']

                # 执行Mann-Whitney U检验（适用于独立样本）
                try:
                    statistic, p_value = mannwhitneyu(oio_results, baseline_results,
                                                     alternative='less')  # OIO是否显著更小

                    # 计算效应大小 (Cohen's d)
                    pooled_std = np.sqrt((np.var(oio_results) + np.var(baseline_results)) / 2)
                    cohens_d = (np.mean(oio_results) - np.mean(baseline_results)) / pooled_std

                    significance = 'Yes' if p_value < 0.05 else 'No'

                    statistical_results.append({
                        'Function': func_name,
                        'Baseline_Algorithm': alg_name,
                        'OIO_Mean': np.mean(oio_results),
                        'Baseline_Mean': np.mean(baseline_results),
                        'U_Statistic': statistic,
                        'P_Value': p_value,
                        'Significant': significance,
                        'Cohens_D': cohens_d,
                        'Effect_Size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
                    })

                except Exception as e:
                    print(f"  ⚠️ {func_name} vs {alg_name}: 统计检验失败 - {e}")

    # 保存统计检验结果
    if statistical_results:
        stats_df = pd.DataFrame(statistical_results)
        stats_file = f'{output_dir}/statistical_tests.csv'
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"📊 统计检验结果已保存: {stats_file}")

        # 显示显著性结果摘要
        significant_count = len(stats_df[stats_df['Significant'] == 'Yes'])
        total_count = len(stats_df)
        print(f"📈 显著性结果: {significant_count}/{total_count} 个比较中OIO显著优于基线算法")

    return statistical_results

def main(load_existing_data=False, run_id=None):
    """主函数

    Args:
        load_existing_data: 是否加载已有数据而不重新运行实验
        run_id: 指定要加载的实验ID，如果为None则加载最新的
    """
    # 设置matplotlib
    setup_matplotlib()

    # 初始化数据管理器
    data_manager = DataManager()

    # 启动内存监控
    tracemalloc.start()
    initial_memory = get_memory_usage()

    print("🌊 CEC2022基准测试 - OIO vs 基线算法")
    print("=" * 80)
    print(f"初始内存使用: {initial_memory:.1f} MB")

    # 测试配置
    dim = 10  # CEC2022标准维度
    num_runs = 10  # 每个算法运行次数
    max_evaluations = 20000  # 最大评估次数

    # CEC2022函数列表
    cec_functions = {
        'F1': F12022(ndim=dim),
        'F2': F22022(ndim=dim),
        'F3': F32022(ndim=dim),
        'F4': F42022(ndim=dim),
        'F5': F52022(ndim=dim),
        'F6': F62022(ndim=dim),
        'F7': F72022(ndim=dim),
        'F8': F82022(ndim=dim),
        'F9': F92022(ndim=dim),
        'F10': F102022(ndim=dim),
        'F11': F112022(ndim=dim),
        'F12': F122022(ndim=dim)
    }

    experiment_config = {
        'dim': dim,
        'num_runs': num_runs,
        'max_evaluations': max_evaluations,
        'functions': list(cec_functions.keys()),
        'algorithms': list(ALL_CEC_ALGORITHMS.keys())
    }

    print(f"测试配置:")
    print(f"  维度: {dim}D")
    print(f"  函数数量: {len(cec_functions)}")
    print(f"  算法数量: {len(ALL_CEC_ALGORITHMS)}")
    print(f"  每个算法运行次数: {num_runs}")
    print(f"  最大评估次数: {max_evaluations}")
    print(f"  总测试次数: {len(cec_functions) * len(ALL_CEC_ALGORITHMS) * num_runs}")

    # 检查是否加载已有数据
    if load_existing_data:
        print(f"\n📂 尝试加载已有实验数据...")
        data_manager.list_available_data()

        all_results, loaded_config = data_manager.load_experiment_results(run_id)
        if all_results is not None:
            print(f"✅ 成功加载实验数据!")
            if loaded_config:
                print(f"   加载的配置: {loaded_config}")

            # 使用加载的数据进行可视化和分析
            output_dir = 'results'
            Path(output_dir).mkdir(exist_ok=True)

            print(f"\n🎨 使用加载的数据生成可视化分析...")
            try:
                create_performance_boxplot(all_results, cec_functions, output_dir, data_manager)
                create_convergence_plots(all_results, cec_functions, output_dir)
                perform_statistical_tests(all_results, cec_functions, output_dir)
                print(f"✅ 可视化分析完成!")
            except Exception as e:
                print(f"⚠️ 可视化生成失败: {e}")

            return
        else:
            print(f"❌ 未找到可用的实验数据，将重新运行实验...")

    # 存储所有结果
    all_results = {}

    # 对每个函数进行测试
    for func_idx, (func_name, cec_func) in enumerate(cec_functions.items(), 1):
        print(f"\n{'='*60}")
        print(f"函数 {func_idx}/{len(cec_functions)}: {func_name} - {cec_func.__class__.__name__}")
        print(f"{'='*60}")

        # 创建函数适配器
        func_adapter = CEC2022FunctionAdapter(cec_func, dim)
        print(f"搜索范围: [{func_adapter.lb[0]:.1f}, {func_adapter.ub[0]:.1f}]")

        # 使用16线程并行测试所有算法
        func_results = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            # 提交所有任务
            future_to_alg = {
                executor.submit(test_single_algorithm_cec, alg_name, AlgorithmClass, func_adapter, max_evaluations, num_runs, data_manager, func_name): alg_name
                for alg_name, AlgorithmClass in ALL_CEC_ALGORITHMS.items()
            }

            # 收集结果
            completed = 0
            for future in as_completed(future_to_alg):
                alg_name = future_to_alg[future]
                completed += 1

                try:
                    result_name, result_data = future.result()
                    func_results[result_name] = result_data

                    if 'error' not in result_data:
                        fitness_mean = result_data['fitness_mean']
                        fitness_std = result_data['fitness_std']
                        time_mean = result_data['time_mean']
                        print(f"[{completed:2d}/{len(ALL_CEC_ALGORITHMS)}] ✅ {alg_name:<8} 完成: {fitness_mean:.6e}±{fitness_std:.4e} ({time_mean:.2f}s)")
                    else:
                        print(f"[{completed:2d}/{len(ALL_CEC_ALGORITHMS)}] ❌ {alg_name:<8} 错误: {result_data['error']}")

                except Exception as e:
                    print(f"[{completed:2d}/{len(ALL_CEC_ALGORITHMS)}] ❌ {alg_name:<8} 异常: {e}")
                    func_results[alg_name] = {'error': str(e)}

        all_results[func_name] = func_results

        # 显示该函数的CEC风格排名
        cec_ranking = calculate_cec_style_ranking(func_results)
        if cec_ranking:
            print(f"\n🏆 {func_name} CEC风格排名 (基于所有运行实例):")
            for rank, (alg_name, total_score) in enumerate(cec_ranking, 1):
                star = '⭐' if alg_name == 'OIO' else '  '
                print(f"  {rank}. {star} {alg_name}: 总分 {total_score}")

        # 同时显示传统的平均值排名作为对比
        valid_results = {k: v for k, v in func_results.items() if 'error' not in v}
        if valid_results:
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['fitness_mean'])
            print(f"\n📊 {func_name} 传统排名 (按平均值):")
            for rank, (alg_name, result) in enumerate(sorted_results, 1):
                star = '⭐' if alg_name == 'OIO' else '  '
                print(f"  {rank}. {star} {alg_name}: {result['fitness_mean']:.6e}")

    # 生成综合结果表格
    print(f"\n🏆 CEC2022综合基准测试结果")
    print("=" * 100)

    # 准备表格数据
    table_data = []
    for func_name in sorted(cec_functions.keys()):
        for alg_name in sorted(ALL_CEC_ALGORITHMS.keys()):
            result = all_results[func_name][alg_name]
            if 'error' not in result:
                table_data.append({
                    '函数': func_name,
                    '算法': alg_name,
                    '最优值': f"{result['fitness_best']:.6e}",
                    '平均值': f"{result['fitness_mean']:.6e}",
                    '标准差': f"{result['fitness_std']:.6e}",
                    '平均时间(s)': f"{result['time_mean']:.2f}"
                })
            else:
                table_data.append({
                    '函数': func_name,
                    '算法': alg_name,
                    '最优值': 'ERROR',
                    '平均值': 'ERROR',
                    '标准差': 'ERROR',
                    '平均时间(s)': 'ERROR'
                })

    # 创建DataFrame并打印
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

    # 创建结果目录
    output_dir = 'results'
    Path(output_dir).mkdir(exist_ok=True)

    # 保存详细结果
    detailed_file = f'{output_dir}/cec2022_detailed_results.csv'
    df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细结果已保存到: {detailed_file}")

    # 计算每个算法的总体排名 - CEC风格
    print(f"\n🥇 算法总体排名分析:")

    # CEC风格总排名：将每个函数的CEC排名分数相加
    cec_total_scores = {}
    traditional_ranks = {}

    for func_name in cec_functions.keys():
        # CEC风格排名
        cec_ranking = calculate_cec_style_ranking(all_results[func_name])
        for alg_name, score in cec_ranking:
            if alg_name not in cec_total_scores:
                cec_total_scores[alg_name] = 0
            cec_total_scores[alg_name] += score

        # 传统平均值排名
        valid_results = {k: v for k, v in all_results[func_name].items() if 'error' not in v}
        if valid_results:
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['fitness_mean'])
            for rank, (alg_name, _) in enumerate(sorted_results, 1):
                if alg_name not in traditional_ranks:
                    traditional_ranks[alg_name] = []
                traditional_ranks[alg_name].append(rank)

    # CEC风格最终排名 (总分越低越好)
    sorted_cec_ranks = sorted(cec_total_scores.items(), key=lambda x: x[1])

    print("🏆 CEC风格总排名 (基于所有运行实例的综合排名分数):")
    for rank, (alg_name, total_score) in enumerate(sorted_cec_ranks, 1):
        star = '⭐' if alg_name == 'OIO' else '  '
        print(f"  {rank}. {star} {alg_name}: 总分 {total_score}")

    # 传统平均排名
    avg_ranks = {}
    for alg_name, ranks in traditional_ranks.items():
        avg_ranks[alg_name] = np.mean(ranks)

    sorted_avg_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])

    print("\n📊 传统总排名 (按平均排名):")
    for rank, (alg_name, avg_rank) in enumerate(sorted_avg_ranks, 1):
        star = '⭐' if alg_name == 'OIO' else '  '
        print(f"  {rank}. {star} {alg_name}: 平均排名 {avg_rank:.2f}")

    # 保存排名结果 - 包含CEC风格和传统排名
    ranking_data = []

    # 创建算法到排名的映射
    cec_rank_map = {alg: rank for rank, (alg, _) in enumerate(sorted_cec_ranks, 1)}
    traditional_rank_map = {alg: rank for rank, (alg, _) in enumerate(sorted_avg_ranks, 1)}

    all_algorithms = set(cec_total_scores.keys()) | set(avg_ranks.keys())

    for alg_name in all_algorithms:
        ranking_data.append({
            'Algorithm': alg_name,
            'CEC_Rank': cec_rank_map.get(alg_name, 'N/A'),
            'CEC_Total_Score': cec_total_scores.get(alg_name, 'N/A'),
            'Traditional_Rank': traditional_rank_map.get(alg_name, 'N/A'),
            'Traditional_Avg_Rank': avg_ranks.get(alg_name, 'N/A'),
            'Is_OIO': 'Yes' if alg_name == 'OIO' else 'No'
        })

    ranking_df = pd.DataFrame(ranking_data)
    # 按CEC排名排序
    ranking_df = ranking_df.sort_values('CEC_Rank')

    ranking_file = f'{output_dir}/algorithm_rankings.csv'
    ranking_df.to_csv(ranking_file, index=False, encoding='utf-8-sig')
    print(f"📊 排名结果已保存到: {ranking_file}")

    # 保存完整的实验数据
    print(f"\n💾 保存实验数据...")
    current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_manager.save_experiment_results(all_results, experiment_config, current_run_id)
    print(f"✅ 实验数据已保存 (ID: {current_run_id})")

    # 执行可视化分析
    print(f"\n🎨 生成可视化分析...")
    try:
        create_performance_boxplot(all_results, cec_functions, output_dir, data_manager)
        create_convergence_plots(all_results, cec_functions, output_dir)
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")

    # 执行统计检验
    try:
        statistical_results = perform_statistical_tests(all_results, cec_functions, output_dir)
        # 创建统计总结表
        if statistical_results:
            create_statistical_summary_table(statistical_results, output_dir)
    except Exception as e:
        print(f"⚠️ 统计检验失败: {e}")

    # 内存清理
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory

    print(f"\n🧹 内存管理报告:")
    print(f"初始内存: {initial_memory:.1f} MB")
    print(f"最终内存: {final_memory:.1f} MB")
    print(f"内存增长: {memory_increase:.1f} MB")

    print(f"\n📁 所有结果已保存到:")
    print(f"  📊 CSV结果文件 ('{output_dir}' 目录):")
    print(f"    - {detailed_file}: 详细实验结果")
    print(f"    - {ranking_file}: 算法排名")
    print(f"    - statistical_tests.csv: 统计检验结果")
    print(f"  📈 可视化文件 ('{output_dir}' 目录):")
    print(f"    - performance_comparison_boxplot.png: 性能对比箱线图")
    print(f"    - convergence_curves/: 收敛曲线图")
    print(f"  💾 实验数据 ('{data_manager.data_dir}' 目录):")
    print(f"    - results_data/: 完整实验结果 (可重新加载)")
    print(f"    - convergence_data/: 收敛曲线原始数据")
    print(f"    - plot_data/: 绘图数据")

    tracemalloc.stop()
    print(f"\n🎯 CEC2022基准测试完成!")
    print(f"\n💡 提示: 下次可以使用 main(load_existing_data=True) 直接加载数据进行分析")

def load_and_plot(run_id=None):
    """仅加载数据并生成图表，不重新运行实验

    Args:
        run_id: 指定要加载的实验ID，如果为None则加载最新的
    """
    main(load_existing_data=True, run_id=run_id)

def run_new_experiment():
    """运行新的实验"""
    main(load_existing_data=False)

def list_available_experiments():
    """列出可用的实验数据"""
    data_manager = DataManager()
    data_manager.list_available_data()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "load":
            # 加载已有数据进行分析
            run_id = sys.argv[2] if len(sys.argv) > 2 else None
            print(f"🔄 加载已有数据进行分析...")
            load_and_plot(run_id)
        elif command == "list":
            # 列出可用的实验数据
            print(f"📋 列出可用的实验数据...")
            list_available_experiments()
        elif command == "run":
            # 运行新实验
            print(f"🚀 开始新的实验...")
            run_new_experiment()
        else:
            print(f"❌ 未知命令: {command}")
            print(f"可用命令:")
            print(f"  python cec2022_benchmark.py run    # 运行新实验")
            print(f"  python cec2022_benchmark.py load   # 加载最新数据进行分析")
            print(f"  python cec2022_benchmark.py load <run_id>  # 加载指定数据")
            print(f"  python cec2022_benchmark.py list   # 列出可用数据")
    else:
        # 默认运行新实验
        print(f"🚀 开始新的实验...")
        run_new_experiment()
