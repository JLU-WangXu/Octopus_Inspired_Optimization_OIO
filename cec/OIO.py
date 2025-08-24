#!/usr/bin/env python3
"""
OIO算法 - 章鱼群优化算法 (NumPy向量化优化版本)
Ocean Intelligence Optimization Algorithm
简洁、高效、精准的仿生优化算法

"""

import numpy as np
import math

class ChampionTentacles:
    """超级冠军触手类 - 增强版本"""
    def __init__(self, cost_func, dim, global_bounds, swarm_size=30, max_iter=80, center=np.zeros(2), radius=5.0, tentacle_id=0):
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
                c1 = (2.5 - 1.5 * progress) * self.adaptive_factor  # 增强个体学习
                c2 = (0.8 + 2.0 * progress) * self.adaptive_factor  # 增强全局学习
                c3 = (0.8 * progress) * self.adaptive_factor
            else:
                w = (0.8 - 0.7 * progress) * self.adaptive_factor
                c1 = (2.0 - 1.5 * progress) * self.adaptive_factor
                c2 = (1.5 + 1.8 * progress) * self.adaptive_factor
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

            # 移除早停条件，让算法运行完整的迭代次数
            # if self.g_best_val < 1e-25:
            #     break

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
        # 简化版本，不需要try-except，因为我们的异常需要传播出去
        fitness_values = []
        for individual in swarm:
            # 这里的 self.cost_func 调用可能会抛出异常
            # 我们不在这里捕获它，让它自然地中断循环并向上传播
            fitness_values.append(self.cost_func(individual))
        return np.array(fitness_values)

    def _vectorized_levy_update(self, swarm, mask):
        """向量化Levy飞行更新"""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        # 向量化随机索引选择
        rand_indices = np.random.randint(0, self.swarm_size, len(indices))

        # 向量化Levy步长计算
        levy_steps = self._vectorized_levy_flight(len(indices))

        # 向量化位置更新
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
                # 如果没有最优位置，使用随机位置
                swarm[idx] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def _vectorized_pso_update(self, swarm, velocity, p_best_pos, mask, w, c1, c2, c3):
        """向量化PSO更新"""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        # 向量化随机数生成
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

            # 向量化生成候选解
            n_candidates_1 = 8
            candidates_1 = best_pos + np.random.normal(0, noise_scale, (n_candidates_1, self.dim))
            candidates_1 = np.clip(candidates_1, self.bounds[0], self.bounds[1])
            candidates_1 = np.clip(candidates_1, self.global_bounds[0], self.global_bounds[1])

            # 向量化评估候选解
            # 这里的 self.cost_func 调用可能会抛出异常，让它自然传播
            for candidate in candidates_1:
                candidate_val = self.cost_func(candidate)  # 可能会抛异常
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()
                    self.g_best_val = best_val
                    self.g_best_pos = best_pos.copy()

            # 第二批候选解
            n_candidates_2 = 3
            candidates_2 = np.random.normal(0, noise_scale, (n_candidates_2, self.dim))
            candidates_2 = np.clip(candidates_2, self.bounds[0], self.bounds[1])
            candidates_2 = np.clip(candidates_2, self.global_bounds[0], self.global_bounds[1])

            # 这里的 self.cost_func 调用也可能会抛出异常，让它自然传播
            for candidate in candidates_2:
                candidate_val = self.cost_func(candidate)  # 可能会抛异常
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()
                    self.g_best_val = best_val
                    self.g_best_pos = best_pos.copy()

            # 移除早停条件，让算法运行完整的搜索
            # if best_val < 1e-25:
            #     break

class ChampionOctopus:
    """超级冠军章鱼类 - 增强版本"""
    def __init__(self, cost_func, dim, global_bounds):
        self.cost_func = cost_func
        self.dim = dim
        self.global_bounds = global_bounds  # (min_range, max_range)
        self.best_position = None
        self.best_value = np.inf

        # 统一触手配置 - 确保公平比较
        # 每个触手: 40 × 100 = 4000次评估
        # 5个触手总计: 20000次评估
        self.tentacles = []
        tentacle_configs = [
            {'swarm_size': 40, 'max_iter': 100},  # 触手1
            {'swarm_size': 40, 'max_iter': 100},  # 触手2
            {'swarm_size': 40, 'max_iter': 100},  # 触手3
            {'swarm_size': 40, 'max_iter': 100},  # 触手4
            {'swarm_size': 40, 'max_iter': 100},  # 触手5
        ]

        for i, config in enumerate(tentacle_configs):
            tentacle = ChampionTentacles(
                cost_func, dim, global_bounds,
                swarm_size=config['swarm_size'],
                max_iter=config['max_iter'],
                tentacle_id=i
            )
            self.tentacles.append(tentacle)

    def search(self, center_position, search_radius, phase="exploration"):
        """章鱼搜索 - 增强版本"""
        # 更精细的触手配置
        offset_configs = [
            {'scale': 0.0, 'radius_factor': 0.8},    # 主触手 - 中心位置
            {'scale': 0.3, 'radius_factor': 0.7},    # 近距离触手
            {'scale': 0.5, 'radius_factor': 0.6},    # 中距离触手
            {'scale': 0.8, 'radius_factor': 0.5},    # 远距离触手
            {'scale': 1.2, 'radius_factor': 0.4},    # 探索触手
        ]

        for i, tentacle in enumerate(self.tentacles):
            config = offset_configs[min(i, len(offset_configs) - 1)]

            if config['scale'] == 0.0:
                tentacle.center = center_position.copy()
            else:
                # 使用混合策略生成偏移
                if np.random.random() < 0.6:
                    # 正态分布偏移
                    offset = np.random.normal(0, search_radius * config['scale'], self.dim)
                else:
                    # 均匀分布偏移
                    offset = np.random.uniform(-search_radius * config['scale'],
                                             search_radius * config['scale'], self.dim)

                tentacle.center = center_position + offset
                tentacle.center = np.clip(tentacle.center, self.global_bounds[0], self.global_bounds[1])

            # 自适应半径
            base_radius = search_radius * config['radius_factor']
            radius_variation = 0.3 * np.random.random()
            tentacle.radius = base_radius * (1 + radius_variation)
            tentacle.bounds = (tentacle.center - tentacle.radius, tentacle.center + tentacle.radius)

        # 如果是 exploitation 阶段，在优化前调整触手位置到全局最优附近
        if phase == "exploitation" and self.best_position is not None:
            for tentacle in self.tentacles:
                tentacle.center = self.best_position.copy()
                tentacle.radius = search_radius * 0.3
                tentacle.bounds = (tentacle.center - tentacle.radius, tentacle.center + tentacle.radius)

        # 只调用一次 optimize，让 tentacle 内部根据 phase 调整策略
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

        # 增加章鱼数量以提高搜索能力
        self.octopus = ChampionOctopus(cost_func, dim, global_bounds)

        # 添加多样性控制
        self.stagnation_counter = 0
        self.last_best_value = np.inf
        self.diversity_threshold = 5

    def run_octopus_swarm(self):
        """运行章鱼群优化 - 增强版本"""

        # 多起点策略
        best_starts = []
        # 这里的 for 循环可能会被 cost_func 抛出的异常中断
        # 这是我们期望的行为，所以不需要在这里添加 try-except
        for _ in range(3):
            start_pos = np.random.uniform(self.global_bounds[0], self.global_bounds[1], self.dim)
            start_val = self.cost_func(start_pos)  # 可能会抛异常
            best_starts.append((start_pos, start_val))



        # 选择最好的起点
        best_starts.sort(key=lambda x: x[1])
        octopus_position = best_starts[0][0].copy()
        search_radius = (self.global_bounds[1] - self.global_bounds[0]) * 0.6  # 增大初始搜索半径

        for iteration in range(self.num_iteration):
            progress = iteration / self.num_iteration

            # 动态阶段切换
            if progress < 0.7:  # 延长探索阶段
                phase = "exploration"
            else:
                phase = "exploitation"

            self.octopus.search(octopus_position, search_radius, phase)

            # 更新全局最优
            if self.octopus.best_value < self.best_value_all:
                self.best_value_all = self.octopus.best_value
                if self.octopus.best_position is not None:
                    self.best_position_all = self.octopus.best_position.copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1



            # 自适应位置更新
            if self.best_position_all is not None and progress > 0.15:
                direction = self.best_position_all - octopus_position
                # 自适应步长
                step_size = 0.4 * (1 - progress) * (1 + 0.1 * np.random.random())
                octopus_position += step_size * direction
                octopus_position = np.clip(octopus_position, self.global_bounds[0], self.global_bounds[1])

                # 自适应搜索半径
                if phase == "exploration":
                    search_radius *= (0.98 - 0.01 * np.random.random())
                else:
                    search_radius *= (0.90 - 0.05 * np.random.random())

            # 多样性维持策略
            if self.stagnation_counter > self.diversity_threshold:
                if np.random.random() < 0.3:  # 增加重启概率
                    # 选择一个好的起点重启
                    restart_idx = min(2, len(best_starts) - 1)
                    octopus_position = best_starts[restart_idx][0] + np.random.normal(0, 0.1, self.dim)
                    octopus_position = np.clip(octopus_position, self.global_bounds[0], self.global_bounds[1])
                    search_radius = (self.global_bounds[1] - self.global_bounds[0]) * (0.4 + 0.2 * np.random.random())
                    self.stagnation_counter = 0

            # 移除早停条件，让算法运行完整的迭代次数
            # if self.best_value_all < 1e-25:
            #     break

        # 增强的最终搜索
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

            # 向量化生成第一批候选解
            n_candidates_1 = 12
            candidates_1 = best_pos + np.random.normal(0, noise_scale, (n_candidates_1, self.dim))
            candidates_1 = np.clip(candidates_1, self.global_bounds[0], self.global_bounds[1])

            # 批量评估候选解
            # 这里的 self.cost_func 调用可能会抛出异常，让它自然传播
            for candidate in candidates_1:
                candidate_val = self.cost_func(candidate)  # 可能会抛异常
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()

            # 向量化生成第二批候选解
            n_candidates_2 = 5
            candidates_2 = np.random.normal(0, noise_scale, (n_candidates_2, self.dim))
            candidates_2 = np.clip(candidates_2, self.global_bounds[0], self.global_bounds[1])

            # 这里的 self.cost_func 调用也可能会抛出异常，让它自然传播
            for candidate in candidates_2:
                candidate_val = self.cost_func(candidate)  # 可能会抛异常
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_pos = candidate.copy()

            # 移除早停条件，让算法运行完整的搜索
            # if best_val < 1e-30:
            #     break

        self.best_value_all = best_val
        self.best_position_all = best_pos

class ChampionOIOAlgorithm:
    """OIO算法接口 - 针对CEC2022优化版本"""
    def __init__(self, cost_func, bounds, dim):
        self.cost_func = cost_func
        self.bounds = bounds
        self.dim = dim
        self.global_bounds = (bounds[0], bounds[1])  # (min_range, max_range)

    def optimize(self, max_iter=1000):
        """优化接口 - 修复FES超标问题的单次运行版本"""

        # 修复FES超标问题：移除外层循环，只调用一次search
        # 这是实现公平比较的核心修改
        octopus = ChampionOctopus(self.cost_func, self.dim, self.global_bounds)

        start_pos = np.random.uniform(self.global_bounds[0], self.global_bounds[1], self.dim)
        search_radius = (self.global_bounds[1] - self.global_bounds[0]) * 0.6

        # 让 search 方法在探索和开发之间自动切换，或者简化为只探索
        # 简化版：只进行一次完整的搜索过程
        octopus.search(start_pos, search_radius, phase="exploration")

        best_pos_all = octopus.best_position
        best_val_all = octopus.best_value

        del octopus
        return best_pos_all, best_val_all

    def _enhanced_local_search(self, best_pos, best_val):
        """增强局部搜索 - 针对二进制优化问题"""
        if best_pos is None:
            return best_pos, best_val

        current_pos = best_pos.copy()
        current_val = best_val

        # 多尺度邻域搜索
        for scale in [0.1, 0.05, 0.01]:
            improved = True
            while improved:
                improved = False

                # 生成多个候选解
                for _ in range(20):
                    # 在当前解附近生成候选解
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
