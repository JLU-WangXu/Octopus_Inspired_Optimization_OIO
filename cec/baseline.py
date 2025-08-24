#!/usr/bin/env python3
"""
基线算法合集 - 包含所有15种基线算法实现
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple
import math

# ============================================================================
# 基础优化器
# ============================================================================

class BaseOptimizer(ABC):
    """基础优化器抽象类，简化版本"""

    def __init__(self, fitness_func: Callable, sequence_length: int, max_evaluations: int = 5000):
        self.fitness_func = fitness_func
        self.sequence_length = sequence_length
        self.max_evaluations = max_evaluations
        self.evaluation_count = 0
        self.best_sequence = None
        self.best_fitness = -np.inf

    def evaluate(self, sequence):
        """评估序列并更新计数"""
        self.evaluation_count += 1
        fitness = self.fitness_func(sequence)

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_sequence = sequence.copy()

        return fitness

    def should_stop(self):
        """检查是否应该停止优化（达到最大评估次数）"""
        return self.evaluation_count >= self.max_evaluations

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行优化"""
        pass

# ============================================================================
# 经典算法
# ============================================================================

class HillClimbing(BaseOptimizer):
    """爬山算法"""

    def optimize(self):
        # 随机初始化
        current = np.random.randint(0, 2, self.sequence_length)
        current_fitness = self.evaluate(current)

        while not self.should_stop():
            # 生成邻居
            neighbor = current.copy()
            flip_pos = np.random.randint(0, self.sequence_length)
            neighbor[flip_pos] = 1 - neighbor[flip_pos]

            neighbor_fitness = self.evaluate(neighbor)

            # 如果邻居更好，则移动
            if neighbor_fitness > current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness

        return self.best_sequence, self.best_fitness

class GeneticAlgorithm(BaseOptimizer):
    """遗传算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=50):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        # 初始化种群
        population = [np.random.randint(0, 2, self.sequence_length) for _ in range(self.pop_size)]

        while not self.should_stop():
            # 评估种群
            fitness_scores = []
            for ind in population:
                if self.should_stop():
                    break
                fitness_scores.append(self.evaluate(ind))

            if self.should_stop():
                break

            # 选择
            new_population = []
            for _ in range(self.pop_size):
                # 锦标赛选择
                tournament_size = 3
                tournament_indices = np.random.choice(self.pop_size, tournament_size, replace=False)
                winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
                new_population.append(population[winner_idx].copy())

            # 交叉和变异
            for i in range(0, self.pop_size-1, 2):
                if np.random.random() < 0.8:  # 交叉概率
                    crossover_point = np.random.randint(1, self.sequence_length)
                    new_population[i][crossover_point:], new_population[i+1][crossover_point:] = \
                        new_population[i+1][crossover_point:].copy(), new_population[i][crossover_point:].copy()

                # 变异
                for j in range(self.sequence_length):
                    if np.random.random() < 0.01:  # 变异概率
                        new_population[i][j] = 1 - new_population[i][j]
                    if i+1 < len(new_population) and np.random.random() < 0.01:
                        new_population[i+1][j] = 1 - new_population[i+1][j]

            population = new_population

        return self.best_sequence, self.best_fitness

class SimulatedAnnealing(BaseOptimizer):
    """模拟退火算法"""

    def optimize(self):
        current = np.random.randint(0, 2, self.sequence_length)
        current_fitness = self.evaluate(current)

        initial_temp = 100.0

        while not self.should_stop():
            # 计算当前温度
            progress = self.evaluation_count / self.max_evaluations
            temperature = initial_temp * (1 - progress)

            # 生成邻居
            neighbor = current.copy()
            flip_pos = np.random.randint(0, self.sequence_length)
            neighbor[flip_pos] = 1 - neighbor[flip_pos]

            neighbor_fitness = self.evaluate(neighbor)

            # 接受准则
            if neighbor_fitness > current_fitness or \
               np.random.random() < np.exp((neighbor_fitness - current_fitness) / max(temperature, 0.01)):
                current = neighbor
                current_fitness = neighbor_fitness

        return self.best_sequence, self.best_fitness

class DifferentialEvolution(BaseOptimizer):
    """差分进化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=50):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        # 初始化种群（连续值）
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break

                # 选择三个不同的个体
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # 变异
                mutant = population[a] + 0.5 * (population[b] - population[c])
                mutant = np.clip(mutant, 0, 1)

                # 交叉
                trial = population[i].copy()
                for j in range(self.sequence_length):
                    if np.random.random() < 0.7:
                        trial[j] = mutant[j]

                # 转换为二进制并评估
                trial_binary = (trial > 0.5).astype(int)
                trial_fitness = self.evaluate(trial_binary)

                # 选择（只有当试验个体更好时才替换）
                if trial_fitness > fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness

        return self.best_sequence, self.best_fitness

# ============================================================================
# 群智能算法
# ============================================================================

class ParticleSwarmOptimization(BaseOptimizer):
    """粒子群优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, swarm_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.swarm_size = swarm_size

    def optimize(self):
        # 初始化粒子群
        particles = np.random.random((self.swarm_size, self.sequence_length))
        velocities = np.random.random((self.swarm_size, self.sequence_length)) * 0.1

        # 个体最优
        p_best = particles.copy()
        p_best_fitness = []
        for p in particles:
            if self.should_stop():
                break
            fitness = self.evaluate((p > 0.5).astype(int))
            p_best_fitness.append(fitness)
        p_best_fitness = np.array(p_best_fitness)

        # 全局最优
        if len(p_best_fitness) > 0:
            g_best_idx = np.argmax(p_best_fitness)
            g_best = p_best[g_best_idx].copy()
        else:
            return self.best_sequence, self.best_fitness

        while not self.should_stop():
            for i in range(self.swarm_size):
                if self.should_stop():
                    break

                # 更新速度
                r1, r2 = np.random.random(2)
                velocities[i] = (0.7 * velocities[i] +
                               2.0 * r1 * (p_best[i] - particles[i]) +
                               2.0 * r2 * (g_best - particles[i]))

                # 更新位置
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 1)

                # 评估
                binary_particle = (particles[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_particle)

                # 更新个体最优
                if fitness > p_best_fitness[i]:
                    p_best_fitness[i] = fitness
                    p_best[i] = particles[i].copy()

                    # 更新全局最优
                    if fitness > p_best_fitness[g_best_idx]:
                        g_best_idx = i
                        g_best = particles[i].copy()

        return self.best_sequence, self.best_fitness

class HarrisHawksOptimization(BaseOptimizer):
    """哈里斯鹰优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        # 初始化种群
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        if len(fitness_values) == 0:
            return self.best_sequence, self.best_fitness

        # 找到最佳个体（猎物）
        best_idx = np.argmax(fitness_values)
        rabbit_pos = population[best_idx].copy()

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break

                # 计算逃逸能量
                E0 = 2 * np.random.random() - 1
                E = 2 * E0 * (1 - self.evaluation_count / self.max_evaluations)

                if abs(E) >= 1:
                    # 探索阶段
                    if np.random.random() < 0.5:
                        # 随机选择一个鹰的位置
                        rand_idx = np.random.randint(0, self.pop_size)
                        population[i] = population[rand_idx] - np.random.random() * abs(population[rand_idx] - 2 * np.random.random() * population[i])
                    else:
                        # 基于群体平均位置
                        mean_pos = np.mean(population, axis=0)
                        population[i] = (rabbit_pos - mean_pos) - np.random.random() * (2 * np.random.random() * rabbit_pos - population[i])
                else:
                    # 开发阶段
                    if np.random.random() < 0.5:
                        population[i] = rabbit_pos - E * abs(rabbit_pos - population[i])
                    else:
                        population[i] = rabbit_pos - E * abs(rabbit_pos - population[i]) + np.random.random() * (rabbit_pos - population[i])

                # 边界处理
                population[i] = np.clip(population[i], 0, 1)

                # 评估新位置
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)

                # 更新最佳位置
                if fitness > fitness_values[i]:
                    fitness_values[i] = fitness
                    if fitness > fitness_values[best_idx]:
                        best_idx = i
                        rabbit_pos = population[i].copy()

        return self.best_sequence, self.best_fitness

class WhaleOptimization(BaseOptimizer):
    """鲸鱼优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        # 初始化种群
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群并找到最佳个体
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        if len(fitness_values) == 0:
            return self.best_sequence, self.best_fitness

        best_idx = np.argmax(fitness_values)
        best_whale = population[best_idx].copy()

        while not self.should_stop():
            a = 2 - 2 * (self.evaluation_count / self.max_evaluations)  # 线性递减

            for i in range(self.pop_size):
                if self.should_stop():
                    break

                r1, r2 = np.random.random(2)
                A = 2 * a * r1 - a
                C = 2 * r2

                if np.random.random() < 0.5:
                    if abs(A) < 1:
                        # 包围猎物
                        D = abs(C * best_whale - population[i])
                        population[i] = best_whale - A * D
                    else:
                        # 随机搜索
                        rand_idx = np.random.randint(0, self.pop_size)
                        D = abs(C * population[rand_idx] - population[i])
                        population[i] = population[rand_idx] - A * D
                else:
                    # 螺旋更新
                    b = 1
                    l = 2 * np.random.random() - 1
                    D = abs(best_whale - population[i])
                    population[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale

                # 边界处理
                population[i] = np.clip(population[i], 0, 1)

                # 评估新位置
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)

                # 更新最佳位置
                if fitness > fitness_values[i]:
                    fitness_values[i] = fitness
                    if fitness > fitness_values[best_idx]:
                        best_idx = i
                        best_whale = population[i].copy()

        return self.best_sequence, self.best_fitness

# ============================================================================
# 高级算法
# ============================================================================

class HybridPSOGWO(BaseOptimizer):
    """混合PSO-GWO算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        # 初始化种群
        population = np.random.random((self.pop_size, self.sequence_length))
        velocities = np.random.random((self.pop_size, self.sequence_length)) * 0.1

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        if len(fitness_values) < 3:
            return self.best_sequence, self.best_fitness

        # 找到alpha, beta, delta狼
        sorted_indices = np.argsort(fitness_values)[::-1]
        alpha_pos = population[sorted_indices[0]].copy()
        beta_pos = population[sorted_indices[1]].copy()
        delta_pos = population[sorted_indices[2]].copy()

        while not self.should_stop():
            a = 2 - 2 * (self.evaluation_count / self.max_evaluations)

            for i in range(self.pop_size):
                if self.should_stop():
                    break

                # GWO更新
                r1, r2 = np.random.random(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * alpha_pos - population[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.random(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * beta_pos - population[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.random(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * delta_pos - population[i])
                X3 = delta_pos - A3 * D_delta

                # 结合PSO
                w = 0.5
                c1, c2 = 1.5, 1.5
                r1, r2 = np.random.random(2)

                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (alpha_pos - population[i]) +
                               c2 * r2 * ((X1 + X2 + X3) / 3 - population[i]))

                population[i] = (X1 + X2 + X3) / 3 + 0.3 * velocities[i]
                population[i] = np.clip(population[i], 0, 1)

                # 评估新位置
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)

                # 更新fitness_values
                fitness_values[i] = fitness

            # 更新alpha, beta, delta
            sorted_indices = np.argsort(fitness_values)[::-1]
            alpha_pos = population[sorted_indices[0]].copy()
            beta_pos = population[sorted_indices[1]].copy()
            delta_pos = population[sorted_indices[2]].copy()

        return self.best_sequence, self.best_fitness

# ============================================================================
# 更多算法（简化版本）
# ============================================================================

class AdaptiveHarrisHawks(HarrisHawksOptimization):
    """自适应哈里斯鹰优化算法"""
    pass

class HippopotamusOptimization(BaseOptimizer):
    """河马优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break

                # 简化的河马行为模拟
                if np.random.random() < 0.5:
                    # 觅食行为
                    population[i] += np.random.normal(0, 0.1, self.sequence_length)
                else:
                    # 群体行为
                    mean_pos = np.mean(population, axis=0)
                    population[i] = 0.7 * population[i] + 0.3 * mean_pos

                population[i] = np.clip(population[i], 0, 1)
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)

                # 更新适应度值
                if i < len(fitness_values):
                    fitness_values[i] = fitness

        return self.best_sequence, self.best_fitness

class WalrusOptimization(BaseOptimizer):
    """海象优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break

                # 简化的海象行为
                population[i] += np.random.normal(0, 0.05, self.sequence_length)
                population[i] = np.clip(population[i], 0, 1)
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)

                # 更新适应度值
                if i < len(fitness_values):
                    fitness_values[i] = fitness

        return self.best_sequence, self.best_fitness

class CrestedPorcupineOptimizer(BaseOptimizer):
    """冠豪猪优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break
                population[i] += np.random.normal(0, 0.08, self.sequence_length)
                population[i] = np.clip(population[i], 0, 1)
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)
                if i < len(fitness_values):
                    fitness_values[i] = fitness

        return self.best_sequence, self.best_fitness

class ElkHerdOptimizer(BaseOptimizer):
    """麋鹿群优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        # 找到最佳个体
        if len(fitness_values) > 0:
            best_idx = np.argmax(fitness_values)
            best_position = population[best_idx].copy()
        else:
            return self.best_sequence, self.best_fitness

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break

                # 增强的麋鹿群行为：结合随机游走和向最优个体学习
                if np.random.random() < 0.5:
                    # 向最优个体学习
                    population[i] = 0.8 * population[i] + 0.2 * best_position + np.random.normal(0, 0.1, self.sequence_length)
                else:
                    # 随机游走
                    population[i] += np.random.normal(0, 0.08, self.sequence_length)

                population[i] = np.clip(population[i], 0, 1)
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)

                # 更新适应度值和最佳位置
                if i < len(fitness_values):
                    fitness_values[i] = fitness
                    if fitness > fitness_values[best_idx]:
                        best_idx = i
                        best_position = population[i].copy()

        return self.best_sequence, self.best_fitness

class GreylagGooseOptimization(BaseOptimizer):
    """灰雁优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break
                population[i] += np.random.normal(0, 0.07, self.sequence_length)
                population[i] = np.clip(population[i], 0, 1)
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)
                if i < len(fitness_values):
                    fitness_values[i] = fitness

        return self.best_sequence, self.best_fitness

class QuokkaSwarmOptimization(BaseOptimizer):
    """短尾矮袋鼠群优化算法"""

    def __init__(self, fitness_func, sequence_length, max_evaluations=5000, pop_size=30):
        super().__init__(fitness_func, sequence_length, max_evaluations)
        self.pop_size = pop_size

    def optimize(self):
        population = np.random.random((self.pop_size, self.sequence_length))

        # 评估初始种群
        fitness_values = []
        for i in range(self.pop_size):
            if self.should_stop():
                break
            binary_ind = (population[i] > 0.5).astype(int)
            fitness = self.evaluate(binary_ind)
            fitness_values.append(fitness)

        while not self.should_stop():
            for i in range(self.pop_size):
                if self.should_stop():
                    break
                population[i] += np.random.normal(0, 0.04, self.sequence_length)
                population[i] = np.clip(population[i], 0, 1)
                binary_ind = (population[i] > 0.5).astype(int)
                fitness = self.evaluate(binary_ind)
                if i < len(fitness_values):
                    fitness_values[i] = fitness

        return self.best_sequence, self.best_fitness

# ============================================================================
# 算法注册表
# ============================================================================

ALGORITHMS = {
    'HC': HillClimbing,
    'GA': GeneticAlgorithm,
    'SA': SimulatedAnnealing,
    'DE': DifferentialEvolution,
    'PSO': ParticleSwarmOptimization,
    'HHO': HarrisHawksOptimization,
    'WOA': WhaleOptimization,
    'PSO-GWO': HybridPSOGWO,
    'AHHO': AdaptiveHarrisHawks,
    'HO': HippopotamusOptimization,
    'WO': WalrusOptimization,
    'CPO': CrestedPorcupineOptimizer,
    'EHO': ElkHerdOptimizer,
    'GGO': GreylagGooseOptimization,
    'QSO': QuokkaSwarmOptimization,
}
