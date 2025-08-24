#!/usr/bin/env python3
"""
OIO算法 - 章鱼群优化算法 (NumPy向量化优化版本)
Ocean Intelligence Optimization Algorithm
简洁、高效、精准的仿生优化算法

"""

import numpy as np
import math

class ChampionTentacles:
    """超级冠军触手类 - 增强版本，支持二进制优化"""
    def __init__(self, cost_func, dim, global_bounds, swarm_size=30, max_iter=80, center=np.zeros(2), radius=5.0, tentacle_id=0, binary_mode=False):
        self.cost_func = cost_func
        self.dim = dim
        self.global_bounds = global_bounds  # (min_range, max_range)
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.center = center
        self.radius = radius
        self.bounds = (center - radius, center + radius)
        self.tentacle_id = tentacle_id
        self.g_best_pos = None
        self.g_best_val = np.inf

        self.shared_best_pos = None
        self.shared_best_val = np.inf
        self.stagnation_counter = 0
        self.diversity_threshold = 0.005  # 降低多样性阈值

        self.success_rates = np.zeros(swarm_size)
        self.elite_memory = []
        self.max_elite_size = 8  # 增加精英记忆大小

        # 添加自适应参数
        self.adaptive_factor = 1.0

        # 二进制优化相关参数
        self.binary_mode = binary_mode
        self.bit_flip_rate = 0.01  # 位翻转率
        self.crossover_rate = 0.8  # 交叉率

    def update_shared_info(self, shared_best_pos, shared_best_val):
        """更新共享信息"""
        self.shared_best_pos = shared_best_pos
        self.shared_best_val = shared_best_val

    def optimize(self, phase="exploration"):
        """超级冠军优化方法 - NumPy向量化优化版本"""
        swarm = self._smart_initialization()
        velocity = np.random.uniform(-0.05, 0.05, (self.swarm_size, self.dim))

        p_best_pos = swarm.copy()
        # 向量化适应度计算
        p_best_val = self._vectorized_fitness_evaluation(swarm)
        g_best_idx = np.argmin(p_best_val)
        self.g_best_pos = swarm[g_best_idx].copy()
        self.g_best_val = p_best_val[g_best_idx]

        self._update_elite_memory(self.g_best_pos, self.g_best_val)

        for t in range(self.max_iter):
            progress = t / self.max_iter

            E0 = 2 * np.random.random() - 1
            E = 2 * E0 * (1 - progress)
            exploration_prob = 0.5 * (1 + np.cos(np.pi * progress))

            # 自适应参数调整
            if phase == "exploration":
                w = (0.95 - 0.6 * progress) * self.adaptive_factor
                # --- MODIFIED FOR FASTER CONVERGENCE ---
                # 减弱个体学习，加强全局学习，以适应短时优化任务
                c1 = (1.5 - 1.0 * progress) * self.adaptive_factor
                c2 = (1.5 + 1.0 * progress) * self.adaptive_factor
                c3 = (0.8 * progress) * self.adaptive_factor
            else:
                w = (0.8 - 0.7 * progress) * self.adaptive_factor
                # --- MODIFIED FOR FASTER CONVERGENCE ---
                # 在开发阶段同样采用更“贪心”的策略
                c1 = (1.0 - 0.8 * progress) * self.adaptive_factor
                c2 = (2.0 + 1.0 * progress) * self.adaptive_factor
                c3 = (1.5 * progress) * self.adaptive_factor

            # 向量化的随机数生成
            random_vals = np.random.random(self.swarm_size)
            exploration_mask = (np.abs(E) >= 1) | (random_vals < exploration_prob)

            # 自适应策略选择
            strategy_rand = np.random.random(self.swarm_size)
            # 根据进度调整策略概率
            levy_prob = 0.3 + 0.2 * (1 - progress)  # 早期更多Levy飞行
            elite_prob = 0.2 + 0.3 * progress       # 后期更多精英引导

            levy_mask = strategy_rand < levy_prob
            elite_mask = (strategy_rand >= levy_prob) & (strategy_rand < levy_prob + elite_prob) & (self.success_rates > 0.3)
            pso_mask = ~(levy_mask | elite_mask)

            # 向量化的Levy飞行更新
            if np.any(exploration_mask & levy_mask):
                self._vectorized_levy_update(swarm, exploration_mask & levy_mask)

            # 向量化的精英引导更新
            if np.any(exploration_mask & elite_mask):
                self._vectorized_elite_update(swarm, exploration_mask & elite_mask)

            # 向量化的PSO更新
            if np.any(exploration_mask & pso_mask):
                self._vectorized_pso_update(swarm, velocity, p_best_pos,
                                          exploration_mask & pso_mask, w, c1, c2, c3)

            # 向量化的开发阶段更新
            exploitation_mask = ~exploration_mask
            if np.any(exploitation_mask):
                self._vectorized_exploitation_update(swarm, exploitation_mask, E)

            # 向量化的中心引导
            center_mask = np.random.random(self.swarm_size) < 0.15
            if np.any(center_mask):
                self._vectorized_center_guidance(swarm, center_mask, progress)

            # 向量化的边界约束
            swarm = np.clip(swarm, self.bounds[0], self.bounds[1])
            swarm = np.clip(swarm, self.global_bounds[0], self.global_bounds[1])

            # 向量化的适应度评估和更新
            new_costs = self._vectorized_fitness_evaluation(swarm)
            improvement_mask = new_costs < p_best_val

            # 更新个体最优
            p_best_val[improvement_mask] = new_costs[improvement_mask]
            p_best_pos[improvement_mask] = swarm[improvement_mask].copy()

            # 向量化的成功率更新
            self.success_rates[improvement_mask] = np.minimum(1.0, self.success_rates[improvement_mask] + 0.1)
            self.success_rates[~improvement_mask] = np.maximum(0.0, self.success_rates[~improvement_mask] - 0.05)

            # 更新全局最优
            best_idx = np.argmin(new_costs)
            if new_costs[best_idx] < self.g_best_val:
                old_best = self.g_best_val
                self.g_best_val = new_costs[best_idx]
                self.g_best_pos = swarm[best_idx].copy()
                self._update_elite_memory(swarm[best_idx], new_costs[best_idx])

                # 自适应因子调整 - 基于改善程度
                improvement = old_best - self.g_best_val
                if improvement > 0.01:
                    self.adaptive_factor = min(1.2, self.adaptive_factor * 1.05)
                else:
                    self.adaptive_factor = max(0.8, self.adaptive_factor * 0.95)

        if phase == "exploitation" and self.g_best_val > 1e-20:
            self._precision_local_search()

        return self.g_best_pos, self.g_best_val

    def _smart_initialization(self):
        """智能初始化 - 向量化优化版本"""
        swarm = np.zeros((self.swarm_size, self.dim))

        n1 = self.swarm_size // 2
        swarm[:n1] = np.random.uniform(self.bounds[0], self.bounds[1], (n1, self.dim))

        n2 = self.swarm_size // 4
        if self.shared_best_pos is not None:
            noise_scale = self.radius * 0.2
            # 向量化的噪声生成和位置更新
            noise = np.random.normal(0, noise_scale, (n2, self.dim))
            swarm[n1:n1+n2] = self.shared_best_pos + noise
        else:
            swarm[n1:n1+n2] = np.random.uniform(self.bounds[0], self.bounds[1], (n2, self.dim))

        n3 = self.swarm_size - n1 - n2
        swarm[n1+n2:] = np.random.uniform(self.bounds[0], self.bounds[1], (n3, self.dim))

        # 向量化的边界约束
        swarm = np.clip(swarm, self.bounds[0], self.bounds[1])
        swarm = np.clip(swarm, self.global_bounds[0], self.global_bounds[1])

        swarm[0] = np.clip(self.center, self.global_bounds[0], self.global_bounds[1])

        return swarm

    def _update_elite_memory(self, position, value):
        """更新精英记忆"""
        self.elite_memory.append((position.copy(), value))
        self.elite_memory.sort(key=lambda x: x[1])
        if len(self.elite_memory) > self.max_elite_size:
            self.elite_memory = self.elite_memory[:self.max_elite_size]

    def _vectorized_fitness_evaluation(self, swarm):
        """向量化适应度评估，异常会自然向上传播"""
        fitness_values = []
        for individual in swarm:
            fitness_values.append(self.cost_func(individual))
        return np.array(fitness_values)

    def _vectorized_levy_update(self, swarm, mask):
        """向量化Levy飞行更新"""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        rand_indices = np.random.randint(0, self.swarm_size, len(indices))
        levy_steps = self._vectorized_levy_flight(len(indices))
        random_factors = np.random.random((len(indices), self.dim))
        for i, idx in enumerate(indices):
            rand_idx = rand_indices[i]
            levy_step = levy_steps[i]
            new_pos = (swarm[rand_idx] + levy_step *
                      np.abs(swarm[rand_idx] - 2 * random_factors[i] * swarm[idx]))
            swarm[idx] = new_pos

    def _vectorized_elite_update(self, swarm, mask):
        """向量化精英引导更新"""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        for idx in indices:
            if len(self.elite_memory) > 0:
                elite_pos = self.elite_memory[np.random.randint(len(self.elite_memory))][0]
                swarm[idx] = elite_pos + np.random.normal(0, 0.1, self.dim)
            elif self.g_best_pos is not None:
                swarm[idx] = self.g_best_pos + np.random.normal(0, 0.1, self.dim)
            else:
                swarm[idx] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def _vectorized_pso_update(self, swarm, velocity, p_best_pos, mask, w, c1, c2, c3):
        """向量化PSO更新"""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        r_vals = np.random.random((len(indices), 3))
        for i, idx in enumerate(indices):
            r1, r2, r3 = r_vals[i]
            basic_velocity = (w * velocity[idx] +
                            c1 * r1 * (p_best_pos[idx] - swarm[idx]) +
                            c2 * r2 * (self.g_best_pos - swarm[idx]))

            if self.shared_best_pos is not None and self.shared_best_val < self.g_best_val:
                shared_guidance = c3 * r3 * (self.shared_best_pos - swarm[idx])
                basic_velocity += shared_guidance

            velocity[idx] = basic_velocity
            swarm[idx] += velocity[idx]

    def _levy_flight(self):
        """Levy飞行"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step

    def _vectorized_levy_flight(self, batch_size):
        """向量化Levy飞行"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma, (batch_size, self.dim))
        v = np.random.normal(0, 1, (batch_size, self.dim))
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step

    def _vectorized_exploitation_update(self, swarm, mask, E):
        """向量化开发阶段更新"""
        indices = np.where(mask)[0]
        if len(indices) == 0 or self.g_best_pos is None:
            return

        r_vals = np.random.random(len(indices))
        for i, idx in enumerate(indices):
            r = r_vals[i]
            if r < 0.5 and abs(E) < 0.5:
                delta_X = self.g_best_pos - swarm[idx]
                swarm[idx] = delta_X - E * np.abs(np.random.random() * self.g_best_pos - swarm[idx])
            elif r >= 0.5 and abs(E) < 0.5:
                swarm[idx] = self.g_best_pos - E * np.abs(self.g_best_pos - swarm[idx])
            else:
                jump_strength = 2 * (1 - np.random.random())
                swarm[idx] = self.g_best_pos - E * np.abs(jump_strength * self.g_best_pos - swarm[idx])

    def _vectorized_center_guidance(self, swarm, mask, progress):
        """向量化中心引导"""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        for idx in indices:
            center_candidates = [np.zeros(self.dim), self.center, self.g_best_pos]
            if self.shared_best_pos is not None:
                center_candidates.append(self.shared_best_pos)
            if len(self.elite_memory) > 0:
                center_candidates.append(self.elite_memory[0][0])

            selected_center = center_candidates[np.random.randint(len(center_candidates))]
            center_guidance = 0.1 * (1 - progress) * (selected_center - swarm[idx])
            swarm[idx] += center_guidance

    def _precision_local_search(self):
        """精密局部搜索 - 向量化优化版本"""
        if self.g_best_pos is None:
            return

        best_pos = self.g_best_pos.copy()
        best_val = self.g_best_val
        scales = [0.01, 0.001, 0.0001]
        for scale in scales:
            noise_scale = (self.global_bounds[1] - self.global_bounds[0]) * scale
            n_candidates_1 = 8
            candidates_1 = best_pos + np.random.normal(0, noise_scale, (n_candidates_1, self.dim))
            candidates_1 = np.clip(candidates_1, self.bounds[0], self.bounds[1])
            candidates_1 = np.clip(candidates_1, self.global_bounds[0], self.global_bounds[1])
            for candidate in candidates_1:
                candidate_val = self.cost_func(candidate)
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()
                    self.g_best_val = best_val
                    self.g_best_pos = best_pos.copy()
            n_candidates_2 = 3
            candidates_2 = np.random.normal(0, noise_scale, (n_candidates_2, self.dim))
            candidates_2 = np.clip(candidates_2, self.bounds[0], self.bounds[1])
            candidates_2 = np.clip(candidates_2, self.global_bounds[0], self.global_bounds[1])
            for candidate in candidates_2:
                candidate_val = self.cost_func(candidate)
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()
                    self.g_best_val = best_val
                    self.g_best_pos = best_pos.copy()

class ChampionOctopus:
    """超级冠军章鱼类 - 增强版本，支持二进制优化"""
    def __init__(self, cost_func, dim, global_bounds, binary_mode=False):
        self.cost_func = cost_func
        self.dim = dim
        self.global_bounds = global_bounds  # (min_range, max_range)
        self.best_position = None
        self.best_value = np.inf
        self.binary_mode = binary_mode

        self.tentacles = []
        tentacle_configs = [
            {'swarm_size': 40, 'max_iter': 100},
            {'swarm_size': 40, 'max_iter': 100},
            {'swarm_size': 40, 'max_iter': 100},
            {'swarm_size': 40, 'max_iter': 100},
            {'swarm_size': 40, 'max_iter': 100},
        ]

        for i, config in enumerate(tentacle_configs):
            tentacle = ChampionTentacles(
                cost_func, dim, global_bounds,
                swarm_size=config['swarm_size'],
                max_iter=config['max_iter'],
                tentacle_id=i,
                binary_mode=binary_mode
            )
            self.tentacles.append(tentacle)

    def search(self, center_position, search_radius, phase="exploration"):
        """章鱼搜索 - 增强版本"""
        offset_configs = [
            {'scale': 0.0, 'radius_factor': 0.8},
            {'scale': 0.3, 'radius_factor': 0.7},
            {'scale': 0.5, 'radius_factor': 0.6},
            {'scale': 0.8, 'radius_factor': 0.5},
            {'scale': 1.2, 'radius_factor': 0.4},
        ]

        for i, tentacle in enumerate(self.tentacles):
            config = offset_configs[min(i, len(offset_configs) - 1)]
            if config['scale'] == 0.0:
                tentacle.center = center_position.copy()
            else:
                if np.random.random() < 0.6:
                    offset = np.random.normal(0, search_radius * config['scale'], self.dim)
                else:
                    offset = np.random.uniform(-search_radius * config['scale'],
                                             search_radius * config['scale'], self.dim)
                tentacle.center = center_position + offset
                tentacle.center = np.clip(tentacle.center, self.global_bounds[0], self.global_bounds[1])

            base_radius = search_radius * config['radius_factor']
            radius_variation = 0.3 * np.random.random()
            tentacle.radius = base_radius * (1 + radius_variation)
            tentacle.bounds = (tentacle.center - tentacle.radius, tentacle.center + tentacle.radius)

        if phase == "exploitation" and self.best_position is not None:
            for tentacle in self.tentacles:
                tentacle.center = self.best_position.copy()
                tentacle.radius = search_radius * 0.3
                tentacle.bounds = (tentacle.center - tentacle.radius, tentacle.center + tentacle.radius)

        for tentacle in self.tentacles:
            best_pos, best_val = tentacle.optimize(phase)
            if best_val < self.best_value:
                self.best_value = best_val
                self.best_position = best_pos.copy()

        global_best_pos = self.best_position
        global_best_val = self.best_value
        for tentacle in self.tentacles:
            tentacle.update_shared_info(global_best_pos, global_best_val)
        return []

class ChampionOctopusSwarm:
    """冠军章鱼群类 - 增强版本"""
    def __init__(self, num_iteration, cost_func, dim, global_bounds):
        self.num_iteration = num_iteration
        self.cost_func = cost_func
        self.dim = dim
        self.global_bounds = global_bounds  # (min_range, max_range)
        self.best_position_all = None
        self.best_value_all = np.inf
        self.octopus = ChampionOctopus(cost_func, dim, global_bounds)
        self.stagnation_counter = 0
        self.last_best_value = np.inf
        self.diversity_threshold = 5

    def run_octopus_swarm(self):
        """运行章鱼群优化 - 增强版本"""
        best_starts = []
        for _ in range(3):
            start_pos = np.random.uniform(self.global_bounds[0], self.global_bounds[1], self.dim)
            start_val = self.cost_func(start_pos)
            best_starts.append((start_pos, start_val))

        best_starts.sort(key=lambda x: x[1])
        octopus_position = best_starts[0][0].copy()
        search_radius = (self.global_bounds[1] - self.global_bounds[0]) * 0.6

        for iteration in range(self.num_iteration):
            progress = iteration / self.num_iteration
            phase = "exploration" if progress < 0.7 else "exploitation"
            self.octopus.search(octopus_position, search_radius, phase)
            if self.octopus.best_value < self.best_value_all:
                self.best_value_all = self.octopus.best_value
                if self.octopus.best_position is not None:
                    self.best_position_all = self.octopus.best_position.copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            if self.best_position_all is not None and progress > 0.15:
                direction = self.best_position_all - octopus_position
                step_size = 0.4 * (1 - progress) * (1 + 0.1 * np.random.random())
                octopus_position += step_size * direction
                octopus_position = np.clip(octopus_position, self.global_bounds[0], self.global_bounds[1])
                if phase == "exploration":
                    search_radius *= (0.98 - 0.01 * np.random.random())
                else:
                    search_radius *= (0.90 - 0.05 * np.random.random())
            if self.stagnation_counter > self.diversity_threshold:
                if np.random.random() < 0.3:
                    restart_idx = min(2, len(best_starts) - 1)
                    octopus_position = best_starts[restart_idx][0] + np.random.normal(0, 0.1, self.dim)
                    octopus_position = np.clip(octopus_position, self.global_bounds[0], self.global_bounds[1])
                    search_radius = (self.global_bounds[1] - self.global_bounds[0]) * (0.4 + 0.2 * np.random.random())
                    self.stagnation_counter = 0
        if self.best_value_all > 1e-20:
            self._final_precision_search()

    def _final_precision_search(self):
        """最终精密搜索 - 向量化优化版本"""
        if self.best_position_all is None:
            return
        best_pos = self.best_position_all.copy()
        best_val = self.best_value_all
        scales = [0.001, 0.0001, 0.00001]
        for scale in scales:
            noise_scale = (self.global_bounds[1] - self.global_bounds[0]) * scale
            n_candidates_1 = 12
            candidates_1 = best_pos + np.random.normal(0, noise_scale, (n_candidates_1, self.dim))
            candidates_1 = np.clip(candidates_1, self.global_bounds[0], self.global_bounds[1])
            for candidate in candidates_1:
                candidate_val = self.cost_func(candidate)
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()
            n_candidates_2 = 5
            candidates_2 = np.random.normal(0, noise_scale, (n_candidates_2, self.dim))
            candidates_2 = np.clip(candidates_2, self.global_bounds[0], self.global_bounds[1])
            for candidate in candidates_2:
                candidate_val = self.cost_func(candidate)
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()
        self.best_value_all = best_val
        self.best_position_all = best_pos

class ChampionOIOAlgorithm:
    """OIO算法接口 - 针对CEC2022优化版本，增强二进制优化能力"""
    def __init__(self, cost_func, bounds, dim, binary_mode=False, sequence_length=None):
        self.cost_func = cost_func
        self.bounds = bounds
        self.dim = dim
        self.global_bounds = (bounds[0], bounds[1])
        self.binary_mode = binary_mode
        self.sequence_length = sequence_length
        self.bit_flip_probability = 0.01
        self.sigmoid_steepness = 1.0

    def sigmoid_transfer(self, x):
        return 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * x))

    def continuous_to_binary(self, x_continuous):
        if not self.binary_mode:
            return x_continuous
        probabilities = self.sigmoid_transfer(x_continuous)
        binary_sequence = (np.random.rand(len(probabilities)) < probabilities).astype(int)
        return binary_sequence

    def bit_flip_mutation(self, binary_sequence):
        if not self.binary_mode or binary_sequence is None:
            return binary_sequence
        mutated_sequence = binary_sequence.copy()
        flip_mask = np.random.rand(len(binary_sequence)) < self.bit_flip_probability
        mutated_sequence[flip_mask] = 1 - mutated_sequence[flip_mask]
        return mutated_sequence

    def optimize(self, max_iter=1000):
        if self.binary_mode:
            return self._optimize_binary(max_iter)
        else:
            return self._optimize_continuous(max_iter)

    def _optimize_continuous(self, max_iter):
        octopus = ChampionOctopus(self.cost_func, self.dim, self.global_bounds, binary_mode=False)
        start_pos = np.random.uniform(self.global_bounds[0], self.global_bounds[1], self.dim)
        search_radius = (self.global_bounds[1] - self.global_bounds[0]) * 0.6
        octopus.search(start_pos, search_radius, phase="exploration")
        best_pos_all = octopus.best_position
        best_val_all = octopus.best_value
        del octopus
        return best_pos_all, best_val_all

    def _optimize_binary(self, max_iter):
        def binary_cost_func(x_continuous):
            binary_seq = self.continuous_to_binary(x_continuous)
            if np.random.rand() < 0.1:
                binary_seq = self.bit_flip_mutation(binary_seq)
            return self.cost_func(binary_seq)
        
        octopus = ChampionOctopus(binary_cost_func, self.dim, self.global_bounds, binary_mode=True)
        start_pos = np.random.uniform(self.global_bounds[0], self.global_bounds[1], self.dim)
        search_radius = (self.global_bounds[1] - self.global_bounds[0]) * 0.6
        octopus.search(start_pos, search_radius, phase="exploration")
        best_pos_continuous = octopus.best_position
        best_val_all = octopus.best_value
        if best_pos_continuous is not None:
            best_pos_binary = self.continuous_to_binary(best_pos_continuous)
            best_pos_binary = self._binary_local_search(best_pos_binary, best_val_all)
        else:
            best_pos_binary = None
        del octopus
        return best_pos_binary, best_val_all

    def _binary_local_search(self, binary_sequence, current_fitness):
        """
        二进制局部搜索 - MODIFIED: 专为蛋白质序列问题优化
        邻域操作是“改变一个氨基酸”，而不是“翻转一个位”，这更符合问题本身。
        """
        if binary_sequence is None or not hasattr(self, 'sequence_length') or self.sequence_length is None:
            return binary_sequence

        best_sequence = binary_sequence.copy()
        best_fitness = current_fitness
        num_amino_acids = 20  # 标准氨基酸数量

        # 检查维度是否匹配，如果不匹配则退回至简单的位翻转，以保证鲁棒性
        if len(binary_sequence) != self.sequence_length * num_amino_acids:
            # print("Warning: Dimension mismatch in _binary_local_search. Falling back to bit-flip.")
            for i in range(len(binary_sequence)):
                neighbor = binary_sequence.copy()
                neighbor[i] = 1 - neighbor[i]
                try:
                    neighbor_fitness = self.cost_func(neighbor)
                    if neighbor_fitness < best_fitness:
                        best_sequence = neighbor.copy()
                        best_fitness = neighbor_fitness
                except:
                    continue
            return best_sequence

        # 将一维二进制向量重塑为 (序列长度, 氨基酸种类数) 的矩阵
        matrix = best_sequence.reshape((self.sequence_length, num_amino_acids))

        # 尝试对每个位置进行单点突变 (Hill Climbing策略)
        for i in range(self.sequence_length):
            # 找到当前位置的氨基酸索引
            active_indices = np.where(matrix[i] == 1)[0]
            original_aa_idx = active_indices[0] if len(active_indices) > 0 else -1
            
            # 尝试该位置的所有其他可能的氨基酸
            for new_aa_idx in range(num_amino_acids):
                if new_aa_idx == original_aa_idx:
                    continue

                neighbor_matrix = matrix.copy()
                if original_aa_idx != -1:
                    neighbor_matrix[i, original_aa_idx] = 0
                neighbor_matrix[i, new_aa_idx] = 1
                
                try:
                    neighbor_fitness = self.cost_func(neighbor_matrix.flatten())
                    if neighbor_fitness < best_fitness:
                        best_fitness = neighbor_fitness
                        best_sequence = neighbor_matrix.flatten().copy()
                        # 更新当前搜索的矩阵，以便在更好的解的基础上继续搜索
                        matrix = neighbor_matrix.copy()
                        original_aa_idx = new_aa_idx
                except:
                    # 如果评估失败，跳过这个邻居
                    continue
        
        return best_sequence

    def _enhanced_local_search(self, best_pos, best_val):
        if best_pos is None:
            return best_pos, best_val
        current_pos = best_pos.copy()
        current_val = best_val
        for scale in [0.1, 0.05, 0.01]:
            improved = True
            while improved:
                improved = False
                for _ in range(20):
                    noise = np.random.normal(0, scale, self.dim)
                    candidate = current_pos + noise
                    candidate = np.clip(candidate, self.global_bounds[0], self.global_bounds[1])
                    candidate_val = self.cost_func(candidate)
                    if candidate_val < current_val:
                        current_pos = candidate.copy()
                        current_val = candidate_val
                        improved = True
                        break
        return current_pos, current_val

    # --- START: NEW METHOD FOR STRATEGY 2 ---
    def _binary_string_local_search(self, binary_sequence, current_fitness):
        """
        A generic local search for binary strings using a 1-flip neighborhood (Hill Climbing).
        This method will consume FES as it calls self.cost_func.
        """
        if binary_sequence is None:
            return binary_sequence

        best_sequence = binary_sequence.copy()
        best_fitness = current_fitness
        N = len(binary_sequence)
        
        improved = True
        while improved:
            improved = False
            # Create a random order to check neighbors
            indices_to_check = np.random.permutation(N)
            
            for i in indices_to_check:
                # Create a neighbor by flipping one bit
                neighbor = best_sequence.copy()
                neighbor[i] = 1 - neighbor[i]
                
                try:
                    # Evaluate the neighbor (this consumes FES via the provided cost_func)
                    neighbor_fitness = self.cost_func(neighbor)
                    
                    if neighbor_fitness < best_fitness:
                        best_fitness = neighbor_fitness
                        best_sequence = neighbor.copy()
                        improved = True
                        # Found a better solution, restart the search from this new point
                        break 
                except:
                    # If evaluation fails for any reason, just skip this neighbor
                    continue
                    
        return best_sequence
    # --- END: NEW METHOD FOR STRATEGY 2 ---