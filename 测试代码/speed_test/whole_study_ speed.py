import sys
import numpy as np
import math
import time
from functools import reduce
path1 = r"C:\Users\broth\Desktop\whole_study_speed.csv"
ImuDataTxt_OCT= open(path1,"w")




#
#标准版模型
#吸盘：pso算法,可以更换为1.任意群优化算法2.梯度优化算法
#吸盘和触手的交互满足协同进化模型(Cooperative Co-evolution Model)
#触手：群智能+个体强化学习实现的一部分
#触手和个体的交互满足主从协同模型 (Master-Slave Cooperative Model)
#个体：强化学习，用以决定更新的方式和状态
#








#计算任意维度欧氏距离
def euclidean_distance(point1, point2):
    return math.sqrt(sum([(x2 - x1) ** 2 for x1, x2 in zip(point1, point2)]))


#触手
class PSO:
    def __init__(self, cost_func, dim, swarm_size=50, max_iter=100, c1=2.0, c2=2.0, w=0.7, center=np.zeros(2), radius=5.0):
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
        global Found, max_range, min_range, real_min_value, diedai_count
        swarm = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        swarm = np.clip(swarm, min_range, max_range)  ########初始化也要限制
        velocity = np.zeros((self.swarm_size, self.dim))
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
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (p_best_pos[i] - swarm[i]) + self.c2 * r2 * (self.g_best_pos - swarm[i])

                # 更新粒子位置
                swarm[i] += velocity[i]

                # 边界处理
                swarm[i] = np.clip(swarm[i], self.bounds[0], self.bounds[1])

                # 更新个体最优
                if self.cost_func(swarm[i]) < p_best_val[i]:
                    p_best_val[i] = self.cost_func(swarm[i])
                    p_best_pos[i] = swarm[i].copy()

                    # 更新全局最优
                    if p_best_val[i] < self.g_best_val:
                        self.g_best_val = p_best_val[i]
                        self.g_best_pos = p_best_pos[i].copy()
                    if self.g_best_val - real_min_value<1e-9:
                        print("found")
                        Found = 1
                        break
            if Found == 1:
                break

        return self.g_best_pos, self.g_best_val


#章鱼
class PSOControl:
    def __init__(self, num_pso, cost_func, dim):
        #初始化
        self.num_pso = num_pso
        self.cost_func = cost_func
        self.dim = dim
        #PSO参数矩阵，为触手结构的实例
        self.params_list = self.generate_random_params()
        #结果参数
        self.best_values = np.zeros(num_pso)
        self.best_values_ratio = np.ones(num_pso)
        self.best_positions = np.zeros((num_pso, dim))
        self.global_best_value = np.inf
        self.global_best_position = None
        #状态参数
        self.state = 0


    #触手的初始化，添加，减少
    def generate_random_params(self):
        global max_range, min_range
        params_list = []
        for _ in range(self.num_pso):
            swarm_size = 20
            max_iter = 100
            c1 = 1.8
            c2 = 1.8
            w = 0.6  # 随范围改；章鱼数量；

            center = np.random.uniform(min_range, max_range, self.dim)  #####初始位置太限制了，要随函数变
            radius = np.random.uniform(1.0, 3.0) * 10 ** int(math.log(max_range, 10))
            params_list.append((swarm_size, max_iter, c1, c2, w, center, radius))
        return params_list

    def add_params(self,n):
        for _ in range(n):
            swarm_size = np.random.randint(20, 100)
            max_iter = np.random.randint(50, 200)
            c1 = np.random.uniform(1.0, 3.0)
            c2 = np.random.uniform(1.0, 3.0)
            w = np.random.uniform(0.5, 1.0)
            center = np.random.uniform(-5.0, 5.0, self.dim)
            radius = np.random.uniform(1.0, 10.0)
            self.params_list.append((swarm_size, max_iter, c1, c2, w, center, radius))

    def remove_params(self,n):
        del self.params_list[n]


    #根据参数运行，统计结果
    def run_pso(self):
        global Found
        for i, params in enumerate(self.params_list):
            swarm_size, max_iter, c1, c2, w, center, radius = params
            # 创建一个PSO对象
            pso = PSO(self.cost_func, self.dim, swarm_size, max_iter, c1, c2, w, center, radius)

            # 运行PSO优化算法
            best_position, best_value = pso.optimize()

            # 记录每个PSO的最优解和最优值
            self.best_values[i] = best_value
            self.best_positions[i] = best_position

            # 更新全局最优解和最优值
            if best_value < self.global_best_value:
                self.global_best_value = best_value
                self.global_best_position = best_position
            if Found == 1:
                break
            #print("第"+str(i)+"触手的结果为：")
            #print("最优值：", self.best_values[i])
            #print("最优位置：", self.best_positions[i])





    #根据运行结果，群体信息进行调整
    def adjust_pso(self,paradise,state):
        global max_range
        best_value_iteration = min(self.best_values)
        worst_value_iteration = max(self.best_values)


        for i, params in enumerate(self.params_list):
            if self.state == 1:#重复条件搜索
                a = 1

            elif self.state == 2:#扩张收缩搜索
                if best_value_iteration != worst_value_iteration:
                    self.best_values_ratio[i] = (self.best_values[i] - worst_value_iteration) / (best_value_iteration - worst_value_iteration)  # 归一化（0,1）
                    self.best_values_ratio[i] = self.best_values_ratio[i] * (1.3 - 0.7) + 0.7  # 投影到（0.7,1.3）

                swarm_size, max_iter, c1, c2, w, center, radius = params
                swarm_size = np.round(swarm_size * self.best_values_ratio[i]).astype(int)
                max_iter = np.round(max_iter * self.best_values_ratio[i]).astype(int)
                radius = np.round(radius * self.best_values_ratio[i]).astype(int)
                swarm_size = np.clip(swarm_size, 12, 1000)
                radius = np.clip(radius, max_range * 0.02, max_range * 0.08)
                if np.random.random() < 0.05:#ε=0.05
                    center = center + np.random.uniform(-1, 1, size=self.dim) * (center - self.best_positions[i])
                else:
                    # 以1-ε的概率进行利用，选择当前状态下评估值最高的动作
                    center = self.best_positions[i]

                if swarm_size > 0 and max_iter > 0 and radius > 0:
                    self.params_list[i] = (swarm_size, max_iter, c1, c2, w, center, radius)
                else :
                    self.params_list[i] = (10, 3, c1, c2, w, center,5)

            elif self.state == 3:#迁移搜索
                swarm_size, max_iter, c1, c2, w, center, radius = params
               #center = self.best_positions[i] + np.random.uniform(-0.02, 0.1, size=self.dim) * (self.best_positions[i] - paradise)
                center = paradise + np.random.uniform(-0.02, 0.1, size=self.dim) * (paradise - self.best_positions[i])
                self.params_list[i] = (swarm_size, max_iter, c1, c2, w, center, radius)


#章鱼群
class Controller:
    def __init__(self, num_control, num_iteration, cost_func, dim):
        #初始化
        self.num_control = num_control
        self.num_iteration = num_iteration
        self.cost_func = cost_func
        self.dim = dim
        #psocontrol的实例
        self.num_pso = np.random.randint(5, 6, self.num_control)
        self.PSOControls = self.generate_psocontrol()
        #结果参数
        self.best_value_all = np.inf
        self.best_position_all = None
        #强化学习
        self.octopus_statues = np.ones(self.num_control)
        self.actions = np.ones(self.num_control)
        self.octopus_statues_list =  self.generate_octopus_statues_list()
        self.actions_list =  self.generate_actions_list()
        ## SARSA 参数
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.1  # ε-greedy 探索率
        ##  Q 值表
        self.q_values_list = self.generate_Q()



    def generate_psocontrol(self):
        psocontrol_list = []
        for i in range(self.num_control):
            psocontrol_list.append(PSOControl(self.num_pso[i], self.cost_func, self.dim))
        return psocontrol_list

    # 这个功能暂时用不上，且适应新的内容还得改，先不用了
    # def add_psocontrol(self,num_pso):
    #     self.PSOControls.append(PSOControl(num_pso, self.cost_func, self.dim))
    #
    # #删除第n个章鱼
    # def remove_psocontrol(self,n):
    #     del self.PSOControls[n]



    def generate_octopus_statues_list(self):
        octopus_statues_list = []
        for i in range(self.num_control):
            octopus_statues_list.append([1, 2, 3])
        return octopus_statues_list


    def generate_actions_list(self):
        actions_list = []
        for i in range(self.num_control):
            actions_list.append([1, 2, 3])
        return actions_list

    def generate_Q(self):
        q_values_list = []
        for i in range(self.num_control):
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
        td_error = -reward + self.gamma * next_q_value - current_q_value#reward用哪个值？最小值还是和上一次结果的差值
        self.q_values_list[i][(state, action)] += self.alpha * td_error

    def run_psocontrol(self):
        global Found
        j = 0
        while (j < self.num_iteration) and (self.best_value_all > 0):
            j += 1
            print("第",j,"轮")
            for i in range(self.num_control):
                #print("##############################################################################")
                #print("第"+str(i)+"只章鱼的结果为")
                #强化学习
                state = self.octopus_statues[i]
                action = self.actions[i]

                #逻辑运行
                self.PSOControls[i].run_pso()
                if Found == 1:
                    break
                if self.PSOControls[i].global_best_value < self.best_value_all:
                    self.best_value_all = self.PSOControls[i].global_best_value
                    self.best_position_all = self.PSOControls[i].global_best_position

                reward = -self.best_value_all
                next_action = self.choose_action(i,state)
                next_state = next_action
                self.update_q_value(i, state, action, reward, next_state, next_action)
                self.octopus_statues[i] = next_state
                self.actions[i] = next_action
            if Found == 1:
                break


            for i in range(self.num_control):
                self.PSOControls[i].adjust_pso(self.best_position_all,self.octopus_statues_list[i])



if __name__ == "__main__":
    def rastrigin(x):#rastrigin (-5.12,5.12)
        A = 10
        n = len(x)

        return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])
    def sphere(x):#(-5.12,5.12)

        return sum([(xi ** 2 ) for xi in x])
    def Ackley(x):#(-32,32)
        a = 20
        b= 0.2
        c = 2*math.pi
        n = len(x)

        return -a*math.exp(-b*math.sqrt(1/n*sum([(xi**2)for xi in x]))) -math.exp(1/n*sum([(math.cos(c*xi))for xi in x]))+a+math.exp(1)


    def levy(x):  # Levy (-10, 10)
        A = 1
        w = 1 + (x - 1) / 4
        term1 = (np.sin(np.pi * w[0])) ** 2
        term2 = sum([(wi-1)**2*(1+10*math.pow(math.sin(math.pi*wi+1),2)) for wi in w])
        term3 = (w[len(w)-1]-1)**2*(1+math.sin(2*math.pi*w[len(w)-1])**2)

        return term1 + term2 + term3
    def Griewank(x):#(-600,600)
        n = len(x)
        sum_term = np.sum([xi ** 2 for xi in x])
        prod_term = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])

        return 1 / 4000 * sum_term - prod_term + 1
    def Schwefel(x):#(-500,500)
        term1 = 418.9829*len(x)
        term2 = sum([(xi*math.sin(math.sqrt(math.fabs(xi)))) for xi in x])

        return term1 - term2
    def Dixon_Price(x):#(-10,10)
        result = (x[0] - 1) ** 2
        for i in range(1, len(x)):
            result += i * (2 * x[i] ** 2 - x[i - 1]) ** 2

        return result
    def Styblinski_Tang(x):#(-5,5)

        return 1/2*sum([(math.pow(xi,4)-16*xi**2+5*xi)for xi in x])
    def Sum_square(x):#(-5.12,5.12)

        return sum([i * x[i] ** 2 for i in range(len(x))])
    def Trid(x):#(-d**2,d**2)
        term1 = sum([((xi-1)**2)for xi in x])
        term2 = 0

        for i in range(1,len(x)):
            term2 += x[i]*x[i-1]
        return term1-term2
    def Rosenbrock(x):#(-5,10)
        sum = 0
        for i in range(len(x) - 1):
            sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2

        return sum


    def Rotated_Hyper_Ellipsoid(x):#(-65.536,65.536)
        sum = 0
        for i in range(len(x)):
            for j in range(i):
                sum+=x[j]**2

        return sum


    def schaffer_function_n2(x):#(-100,100)
        return 0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    def Beale(x):#(-4.5,4.5)
        term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        term2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return term1 + term2 + term3


    def goldstein_price(x):#(-2,2)
        term1 = 1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
        term2 = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
        return term1 * term2


    def drop_wave(x):#(-5.12,5.12)
        term1 = 1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))
        term2 = 0.5 * (x[0] ** 2 + x[1] ** 2) + 2
        return -term1 / term2


    def easom(x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)

    ##################################
    num_control = 4
    num_iteration = 10
    dim = 2

#################################

    #for i in range(num_control):
        #for j in range(5):#5是触手数
            #ImuDataTxt_tentacles.write(str("octopus")+str(i)+",")
    #ImuDataTxt_tentacles.write('\n')


    octopus = 0#
    tentacles = 0#
    num_ex = 3
    COUNT = 0#
    '''
    for i in range(num_ex):
        begintime = time.time()
        controller = Controller(num_control, num_iteration, rastrigin, dim)# 运行所有PSO优化算法
        controller.run_psocontrol()
        end = time.time()
        print(end-begintime)
        Found = 0
    '''
    max_range = 0
    min_range = 0
    real_min_value = 0
    cost_fun = [Ackley,Griewank,levy,rastrigin,Schwefel,Dixon_Price,Styblinski_Tang,Sum_square,Trid,Rosenbrock,sphere,Rotated_Hyper_Ellipsoid,schaffer_function_n2,Beale,goldstein_price,drop_wave,easom]#
    max_range_list = [32, 600, 10, 5.12, 500, 10, 5, 10, dim**2, 10, 5.12, 65, 100, 4.5, 2, 5.12, 100]
    min_range_list = [-32, -600, -10, -5.12, -500, -10, -5, -10, -dim**2, -5, -5.12, -65, -100, -4.5, -2, -5.12, -100]
    real_min_value_list = [0,0,0,0,0,0, -39.16599*dim,0, -dim*(dim+4)*(dim-1)/6,0,0,0,0,0,3,-1,-1]

    fitness = np.full((num_ex, len(cost_fun)), -1.0)
    time_consume = np.full((num_ex, len(cost_fun)), -1.0)
    ImuDataTxt_OCT.write("fitness, ")
    ImuDataTxt_OCT.write("time, ")
    ImuDataTxt_OCT.write(str('\n'))

    for i in range(17):
        for j in range(num_ex):
            # 创建PSOControl对象
            #print("第",j,"次")
            COUNT=j
            Found = 0
            begin_time = time.time()
            max_range = max_range_list[i]
            min_range = min_range_list[i]
            real_min_value = real_min_value_list[i]


            controller = Controller(num_control, num_iteration, cost_fun[i], dim)
            # 运行所有PSO优化算法
            controller.run_psocontrol()
            end_time = time.time()
            time_cost = end_time - begin_time
            print("###################第", i, j, "次", "time:", time_cost, "s", )
            print(str(cost_fun[i]), max_range, min_range, real_min_value)
            fitness[j, i] = controller.best_value_all-real_min_value
            time_consume[j, i] = time_cost
            diedai_count = 0
            del controller

    for i in range(num_ex):
        for j in range(17):
            ImuDataTxt_OCT.write(str(fitness[i, j]) + ",")
            ImuDataTxt_OCT.write(str(time_consume[i, j]) + ",")
        ImuDataTxt_OCT.write('\n')
    print(time_consume)
    print('\n')
    print(fitness)
    fit_mean = np.mean(fitness)
    fit_std = np.std(fitness)
    time_mean = np.mean(time_consume)
    time_std = np.std(time_consume)
    print(fitness[0,0]-real_min_value, fit_std, time_mean, time_std)



