import sys
import numpy as np
import math


# 计算任意维度欧氏距离

def euclidean_distance(point1, point2):
    return math.sqrt(sum([(x2 - x1) ** 2 for x1, x2 in zip(point1, point2)]))


# 触手
class Tentacles:
    def __init__(self, cost_func, dim, swarm_size=50, max_iter=100, c1=2.0, c2=2.0, w=0.7, center=np.zeros(2),
                 radius=5.0):
        self.cost_func = cost_func
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.center = center
        self.radius = radius
        self.bounds = (center - radius, center + radius)
        self.g_best_pos = None
        self.g_best_val = np.inf

    def optimize(self):
        # 初始化粒子位置和速度
        swarm = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        swarm = np.clip(swarm, min_range, max_range)  ########初始化也要限制
        velocity = np.random.uniform(min_range * 0.001, max_range * 0.001, (self.swarm_size, self.dim))
        p_best_pos = swarm.copy()
        p_best_val = np.array([self.cost_func(p) for p in swarm])
        g_best_idx = np.argmin(p_best_val)
        self.g_best_pos = swarm[g_best_idx].copy()
        self.g_best_val = p_best_val[g_best_idx]

        for t in range(self.max_iter):

            for i in range(self.swarm_size):
                # 更新粒子速度
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (p_best_pos[i] - swarm[i]) + self.c2 * r2 * (
                            self.g_best_pos - swarm[i])

                # 更新粒子位置
                swarm[i] += velocity[i]
                # 边界处理
                swarm[i] = np.clip(swarm[i], self.bounds[0], self.bounds[1])
                swarm[i] = np.clip(swarm[i], min_range, max_range)

                # 更新个体最优
                if self.cost_func(swarm[i]) < p_best_val[i]:
                    p_best_val[i] = self.cost_func(swarm[i])
                    p_best_pos[i] = swarm[i].copy()

                    # 更新全局最优
                    if p_best_val[i] < self.g_best_val:
                        self.g_best_val = p_best_val[i]

                        self.g_best_pos = p_best_pos[i].copy()

        return self.g_best_pos, self.g_best_val


# 章鱼
class Octopus:
    def __init__(self, num_tentacles, cost_func, dim):
        # 初始化
        self.num_tentacles = num_tentacles
        self.cost_func = cost_func
        self.dim = dim
        # 参数矩阵，为触手结构的实例
        self.params_list = self.generate_tentacles()
        # 结果参数
        self.best_values = [float('inf')] * num_tentacles
        self.best_values_ratio = np.ones(num_tentacles)

        self.best_positions = np.zeros((num_tentacles, dim))

        self.global_best_value = np.inf
        self.global_best_position = None
        # 调整新位置
        self.center_list = np.zeros((num_tentacles, dim))
        self.radius_list = np.zeros(num_tentacles)
        self.reborn_flag = np.zeros(num_tentacles)

    # 触手的初始化，添加，减少
    def generate_tentacles(self):
        global max_range, min_range
        params_list = []

        step = (max_range - min_range) / self.num_tentacles
        for i in range(self.num_tentacles):
            swarm_size = 50  # min = 10,max=400
            max_iter = 200  # min = 20,max=400
            c1 = 1.8
            c2 = 1.8
            w = 0.6

            center = np.random.uniform(min_range + i * step, min_range + (i + 1) * step, self.dim)
            radius = (max_range - min_range) * np.random.uniform(0.1, 0.2)
            params_list.append((swarm_size, max_iter, c1, c2, w, center, radius))
        return params_list

    def add_tentacles(self, n):
        for _ in range(n):
            swarm_size = np.random.randint(20, 100)
            max_iter = np.random.randint(50, 200)
            c1 = np.random.uniform(1.0, 3.0)
            c2 = np.random.uniform(1.0, 3.0)
            w = np.random.uniform(0.5, 1.0)
            center = np.random.uniform(-5.0, 5.0, self.dim)
            radius = np.random.uniform(1.0, 10.0)
            self.params_list.append((swarm_size, max_iter, c1, c2, w, center, radius))

    def remove_tentacles(self, n):
        del self.params_list[n]

    # 根据参数运行，统计结果
    def run_tentacles(self):
        for i, params in enumerate(self.params_list):
            swarm_size, max_iter, c1, c2, w, center, radius = params

            tentacle = Tentacles(self.cost_func, self.dim, swarm_size, max_iter, c1, c2, w, center, radius)


            best_position, best_value = tentacle.optimize()

            if self.reborn_flag[i] == 1:
                self.best_values[i] = best_value
                self.best_positions[i] = best_position
                self.reborn_flag[i] = 0

            elif best_value < self.best_values[i]:
                self.best_values[i] = best_value
                self.best_positions[i] = best_position

            # 更新全局最优解和最优值
            if self.best_values[i] < self.global_best_value:
                self.global_best_value = self.best_values[i]
                self.global_best_position = self.best_positions[i]
            print("第", i, "个触手位置", tentacle.center, "size:", tentacle.swarm_size, "迭代次数", tentacle.max_iter, "找到的值",
                  self.best_values[i],"最佳位置",self.best_positions[i])

    # 根据运行结果，群体信息进行调整
    def adjust_tentacles(self, paradise, state):
        global max_range, min_range
        best_value_iteration = min(self.best_values)
        worst_value_iteration = max(self.best_values)
        total_range = max_range - min_range



        for i, params in enumerate(self.params_list):

            swarm_size, max_iter, c1, c2, w, center, radius = params
            maxratio = 1
            minratio = 0
            if state == 0:
                maxratio = 1.6
                minratio = 0.4
                center = self.best_positions[i]
            elif state == 1:
                maxratio = 1.2
                minratio = 0.8
                center = center + (self.best_positions[i] - center) * 3
            elif state == 2:
                maxratio = 2
                minratio = 0.9
                center = center + (self.best_positions[i] - center) * 0.7


            if best_value_iteration != worst_value_iteration:
                self.best_values_ratio[i] = (self.best_values[i] - worst_value_iteration) / (best_value_iteration - worst_value_iteration)  # 归一化（0,1）
                self.best_values_ratio[i] = self.best_values_ratio[i] * (maxratio - minratio) + minratio  # 投影到（0.2,1.1）
            swarm_size = np.round(swarm_size * self.best_values_ratio[i]).astype(int)
            max_iter = np.round(max_iter * self.best_values_ratio[i]).astype(int)
            radius = np.round(radius * self.best_values_ratio[i]).astype(int)
            swarm_size = np.clip(swarm_size, 10, 400)
            radius = np.clip(radius, total_range * 0.02, total_range * 0.08)
            max_iter = np.clip(max_iter, 20, 400).astype(int)

            if swarm_size < 12 or max_iter < 20:
                self.reborn_flag[i] = 1
                # print(i,"因为边角料重生了")

            elif i > 0:
                for j in range(i):
                    distance = euclidean_distance(self.center_list[j], center)
                    distance -= self.radius_list[j]

                    if distance < radius:
                        # print("第",i,j,"重了")
                        if self.best_values[i] > self.best_values[j] and self.reborn_flag[i] == 0:
                            self.reborn_flag[i] = 1
                            center = np.random.uniform(min_range, max_range, self.dim)
                        # print(i,"更大","第",i,"个触手变了")

                        elif self.reborn_flag[j] == 0:

                            temp_swarm_size, temp_max_iter, c1, c2, w, temp_center, temp_radius = self.params_list[j]
                            self.reborn_flag[j] = 1
                            temp_center = np.random.uniform(min_range, max_range, self.dim)
                            # print(j,"更大","第", j, "个触手变了")
                            for k in range(self.dim):
                                if np.random.randn() < 0.5:
                                    temp_center[k] = np.random.uniform(min_range, self.global_best_position[
                                        k] - total_range * 0.08)
                                else:
                                    temp_center[k] = np.random.uniform(
                                        self.global_best_position[k] + total_range * 0.08, max_range)
                                self.params_list[j] = (
                                20, 40, c1, c2, w, temp_center, total_range * np.random.uniform(0.06, 0.08))
                        # print("第",j,"个触手碰撞重生了")

            self.radius_list[i] = radius
            self.center_list[i] = center
            if self.reborn_flag[i] == 0:
                self.params_list[i] = (swarm_size, max_iter, c1, c2, w, center, radius)
            else:
                for k in range(self.dim):
                    if np.random.randn() < 0.5:
                        center[k] = np.random.uniform(min_range, self.global_best_position[k] - total_range * 0.08)
                    else:
                        center[k] = np.random.uniform(self.global_best_position[k] + total_range * 0.08, max_range)
                self.params_list[i] = (20, 40, c1, c2, w, center, total_range * np.random.uniform(0.06, 0.08))


class OctopusSwarm:
    def __init__(self, num_octopus, num_iteration, cost_func, dim):
        # 初始化
        self.num_octopus = num_octopus
        self.num_iteration = num_iteration
        self.cost_func = cost_func
        self.dim = dim
        # Octopus的实例
        self.num_tentacles = np.random.randint(10, 11, self.num_octopus)
        self.octopus_swarm = self.generate_octopus_swarm()
        # 结果参数
        self.best_value_all = np.inf
        self.best_position_all = None
        # 强化学习
        self.octopus_statues = np.ones(self.num_octopus)
        self.actions = np.ones(self.num_octopus)
        self.octopus_statues_list = self.generate_octopus_statues_list()
        self.actions_list = self.generate_actions_list()
        # SARSA 参数
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.1  # ε-greedy 探索率
        #  Q 值表
        self.q_values_list = self.generate_Q()



    def generate_octopus_swarm(self):
        octopus_swarm_list = []
        for i in range(self.num_octopus):
            octopus_swarm_list.append(Octopus(self.num_tentacles[i], self.cost_func, self.dim))
        return octopus_swarm_list

    def add_octopus(self, num_tent):
        self.octopus_swarm.append(Octopus(num_tent, self.cost_func, self.dim))

    # 删除第n个章鱼
    def remove_octopus(self, n):
        del self.octopus_swarm[n]



    def generate_octopus_statues_list(self):
        octopus_statues_list = []
        for i in range(self.num_octopus):
            octopus_statues_list.append([1, 2, 3])
        return octopus_statues_list


    def generate_actions_list(self):
        actions_list = []
        for i in range(self.num_octopus):
            actions_list.append([1, 2, 3])
        return actions_list

    def generate_Q(self):
        q_values_list = []
        for i in range(self.num_octopus):
            q_values = {}
            for state in [1, 2, 3]:
                for action in [1, 2, 3]:
                    q_values[(state, action)] = 0.0
            q_values_list.append(q_values)
        return q_values_list


    def choose_action(self, i, state):
        # ε-greedy 策略选择行动
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions_list[i])  # 随机选择行动
        else:
            # 选择具有最大 Q 值的行动
            max_q_value = max([self.q_values_list[i][(state, a)] for a in self.actions_list[i]])
            return np.random.choice([a for a in self.actions_list[i] if self.q_values_list[i][(state, a)] == max_q_value])

    def update_q_value(self, i, state, action, reward, next_state, next_action):
        # 根据 SARSA 更新公式更新 Q 值
        current_q_value = (self.q_values_list[i])[(state, action)]
        next_q_value = self.q_values_list[i][(next_state, next_action)]
        td_error = -reward + self.gamma * next_q_value - current_q_value
        self.q_values_list[i][(state, action)] += self.alpha * td_error




    def run_octopus_swarm(self):
        for j in range(self.num_iteration):

            for i in range(self.num_octopus):
                print("#############################第", i, "只章鱼##############################")
                state = self.octopus_statues[i]
                action = self.actions[i]

                self.octopus_swarm[i].run_tentacles()

                if self.octopus_swarm[i].global_best_value < self.best_value_all:
                    self.best_value_all = self.octopus_swarm[i].global_best_value
                    self.best_position_all = self.octopus_swarm[i].global_best_position

                reward = -self.best_value_all
                next_action = self.choose_action(i, state)
                next_state = next_action
                self.update_q_value(i, state, action, reward, next_state, next_action)
                self.octopus_statues[i] = next_state
                self.actions[i] = next_action


            for i in range(self.num_octopus):
                self.octopus_swarm[i].adjust_tentacles(self.best_position_all, self.octopus_statues_list[i])

            print("最佳值：",self.best_value_all,"     最佳位置：",self.best_position_all)


if __name__ == "__main__":
    def cost_func(x):  # rastrigin
        A = 10
        n = len(x)
        return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


    max_range = 32
    min_range = -32
    real_min_value = 0

    num_octopus = 1
    num_iteration = 50
    dim = 2

    # 创建章鱼对象
    octopus_swarm = OctopusSwarm(num_octopus, num_iteration, cost_func, dim)

    # 运行优化算法
    octopus_swarm.run_octopus_swarm()

    # 输出全局最优解和最优值
    print("章鱼个数:", num_octopus)
    print("每条章鱼触手个数:", octopus_swarm.num_tentacles)
    print("全局最优值:", octopus_swarm.best_value_all)
    print("全局最优位置:", octopus_swarm.best_position_all)