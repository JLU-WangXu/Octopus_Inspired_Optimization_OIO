import sys
import numpy as np
import math
import time
from functools import reduce
path1 = r"/home/wangxu/octopus/lixiang/10d/accuracy_try/output/whole_accuracy.csv"
ImuDataTxt_OCT= open(path1,"w")
path_record = r"/home/wangxu/octopus/lixiang/10d/accuracy_try/output/OAshoulian.csv"
Record_write = open(path_record,"w")

#计算任意维度欧氏距离

def euclidean_distance(point1, point2):
    return math.sqrt(sum([(x2 - x1) ** 2 for x1, x2 in zip(point1, point2)]))

Found = 0

#触手
class Tentacles:
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
        global Found,max_range,min_range,real_min_value,diedai_count,controller
        # 初始化粒子位置和速度
        swarm = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        swarm = np.clip(swarm, min_range, max_range)########初始化也要限制
        velocity = np.random.uniform(min_range*0.001,max_range*0.001,(self.swarm_size, self.dim))
        p_best_pos = swarm.copy()
        p_best_val = np.array([self.cost_func(p) for p in swarm])
        g_best_idx = np.argmin(p_best_val)
        self.g_best_pos = swarm[g_best_idx].copy()
        self.g_best_val = p_best_val[g_best_idx]

        for t in range(self.max_iter):

            Record_write.write(str(controller.best_value_all-real_min_value)+',')
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (p_best_pos[i] - swarm[i]) + self.c2 * r2 * (self.g_best_pos - swarm[i])

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


                    if math.fabs(self.g_best_val) ==real_min_value:
                        print("found")
                        Found = 1
                        break

            if Found == 1:
                break

        return self.g_best_pos, self.g_best_val


#章鱼
class TentaclesControl:
    def __init__(self, num_tentacle, cost_func, dim):
        #初始化
        self.num_tentacle = num_tentacle
        self.cost_func = cost_func
        self.dim = dim
        #Tentacles参数矩阵，为触手结构的实例
        self.params_list = self.generate_random_params()
        #结果参数
        self.best_values = np.zeros(num_tentacle)
        self.best_values_ratio = np.ones(num_tentacle)
        self.best_positions = np.zeros((num_tentacle, dim))
        self.global_best_value = np.inf
        self.global_best_position = None


    #触手的初始化，添加，减少
    def generate_random_params(self):
        global max_range, min_range
        params_list = []

        for _ in range(self.num_tentacle):

            swarm_size = 50
            max_iter = 200
            c1 =1.8
            c2 =1.8
            w = 0.6#随范围改；章鱼数量；

            center = np.random.uniform(min_range, max_range, self.dim)

            radius = np.random.uniform(1.0, 3.0)*10**int(math.log(max_range,10))
            #radius = (max_range-min_range)*np.random.uniform(0.03,0.05)
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
    def run_tentacle(self):
        global Found,tentacles,process_end,process_begin
        for i, params in enumerate(self.params_list):
            tentacles =i
            #print(octopus, tentacles)
            swarm_size, max_iter, c1, c2, w, center, radius = params
            # 创建一个Tentacles对象
            tentacle = Tentacles(self.cost_func, self.dim, swarm_size, max_iter, c1, c2, w, center, radius)

            # 运行Tentacles优化算法
            best_position, best_value = tentacle.optimize()
            print("第", i, "个触手位置", tentacle.g_best_pos,"size:",tentacle.swarm_size, "迭代次数",tentacle.max_iter,"找到的值", tentacle.g_best_val)
            # 记录每个Tentacles的最优解和最优值
            self.best_values[i] = best_value
            self.best_positions[i] = best_position


            # 更新全局最优解和最优值
            if best_value < self.global_best_value:
                self.global_best_value = best_value
                self.global_best_position = best_position
            if Found == 1:
                break
        process_end = time.time()
        print("本章鱼最优值：",self.global_best_value,"目前耗时：",process_end-process_begin)

    #根据运行结果，群体信息进行调整
    def adjust_tentacle(self,paradise):
        global max_range
        best_value_iteration = min(self.best_values)
        worst_value_iteration = max(self.best_values)


        for i, params in enumerate(self.params_list):

            if best_value_iteration != worst_value_iteration:
                self.best_values_ratio[i] = (self.best_values[i] - worst_value_iteration) / (
                            best_value_iteration - worst_value_iteration)  # 归一化（0,1）
                self.best_values_ratio[i] = self.best_values_ratio[i] * (1.3 - 0.7) + 0.7  # 投影到（0.7,1.3）


            swarm_size, max_iter, c1, c2, w, center, radius = params
            swarm_size = np.round(swarm_size * self.best_values_ratio[i]).astype(int)
            max_iter = np.round(max_iter * self.best_values_ratio[i]).astype(int)
            radius = np.round(radius * self.best_values_ratio[i]).astype(int)
            center = self.best_positions[i]
            swarm_size = np.clip(swarm_size, 12, 1000)
            radius = np.clip(radius,max_range*0.02,max_range*0.08)

            if swarm_size > 0 and max_iter != 0 and radius != 0:
                self.params_list[i] = (swarm_size, max_iter, c1, c2, w, center, radius)
            else :
                self.params_list[i] = (10, 3, c1, c2, w, center,5)


#章鱼群
process_begin = time.time()
process_end = 0
class Controller:
    def __init__(self, num_control, num_iteration, cost_func, dim):
        #初始化
        self.num_control = num_control
        self.num_iteration = num_iteration
        self.cost_func = cost_func
        self.dim = dim
        #tentaclecontrol的实例
        self.num_tentacle = np.random.randint(10,11,self.num_control)
        #self.search_range = search_range
        #结果参数
        self.best_value_all = np.inf
        self.best_position_all = None
        self.TentaclesControls = self.generate_tentaclecontrol()

    def generate_tentaclecontrol(self):
        tentaclecontrol_list = []
        for i in range(self.num_control):
            tentaclecontrol_list.append(TentaclesControl(self.num_tentacle[i], self.cost_func, self.dim))
        return tentaclecontrol_list


    def add_tentaclecontrol(self,num_tentacle):
        self.TentaclesControls.append(TentaclesControl(num_tentacle, self.cost_func, self.dim))

    #删除第n个章鱼
    def remove_tentaclecontrol(self,n):
        del self.TentaclesControls[n]


    def run_tentaclecontrol(self):
        global Found, octopus,COUNT
        for j in range(self.num_iteration):

            for i in range(self.num_control):
                print("#############################第", i, "只章鱼##############################")
                octopus = i
                self.TentaclesControls[i].run_tentacle()

                if self.TentaclesControls[i].global_best_value < self.best_value_all:
                    self.best_value_all = self.TentaclesControls[i].global_best_value
                    self.best_position_all = self.TentaclesControls[i].global_best_position
                if Found == 1:
                    break
            if Found == 1:
                break
            for i in range(self.num_control):
                self.TentaclesControls[i].adjust_tentacle(self.best_position_all)



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
        term1 = 418.98288727243369 * len(x)
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


    def Levy_N_13(x):  # 2维
        term1 = math.sin(3 * math.pi * x[0]) ** 2
        term2 = (x[0] - 1) ** 2 * (1 + math.sin(3 * math.pi * x[1]) ** 2)
        term3 = (x[1] - 1) ** 2 * (1 + math.sin(2 * math.pi * x[1]) ** 2)
        return term1 + term2 + term3


    def sum_of_different_powers(x):  # n维
        temp = 0
        for i in range(len(x)):
            temp += math.fabs(x[i]) ** (i + 1)
        return temp
    ##################################
    num_control = 4
    num_iteration = 5
    dim = 10

#################################

    #for i in range(num_control):
        #for j in range(5):#5是触手数
            #ImuDataTxt_tentacles.write(str("octopus")+str(i)+",")
    #ImuDataTxt_tentacles.write('\n')
    Found = 0

    octopus = 0#
    tentacles = 0#
    num_ex = 1
    COUNT = 0#
    '''
    for i in range(num_ex):
        begintime = time.time()
        controller = Controller(num_control, num_iteration, rastrigin, dim)# 运行所有Tentacles优化算法
        controller.run_psocontrol()
        end = time.time()
        print(end-begintime)
        Found = 0
    '''
    max_range = 0
    min_range = 0
    real_min_value = 0
    cost_fun = [Ackley, Griewank, levy, rastrigin, Dixon_Price, Sum_square, Trid, Rosenbrock,
                sphere, Rotated_Hyper_Ellipsoid, sum_of_different_powers]
    max_range_list = [32, 600, 10, 5.12, 10, 10, dim ** 2, 10, 5.12, 65, 1]
    min_range_list = [-32, -600, -10, -5.12, -10, -10, -dim ** 2, -5, -5.12, -65, -1]
    real_min_value_list = [0, 0, 0, 0, 0, 0, -dim * (dim + 4) * (dim - 1) / 6, 0, 0, 0, 0]
    fitness = np.full((num_ex, len(cost_fun)), -1.0)
    time_consume = np.full((num_ex, len(cost_fun)), -1.0)
    ImuDataTxt_OCT.write("fitness, ")
    ImuDataTxt_OCT.write("time, ")
    ImuDataTxt_OCT.write(str('\n'))

    for i in range(11):
        Record_write.write('\n')
        Record_write.write('\n')
        Record_write.write('\n')
        for j in range(num_ex):
            Record_write.write('\n')
            # 创建TentaclesControl对象
            #print("第",j,"次")
            COUNT=j
            Found = 0
            begin_time = time.time()
            max_range = max_range_list[i]
            min_range = min_range_list[i]
            real_min_value = real_min_value_list[i]


            controller = Controller(num_control, num_iteration, cost_fun[i], dim)
            # 运行所有Tentacles优化算法
            controller.run_tentaclecontrol()
            end_time = time.time()
            time_cost = end_time - begin_time
            print("###################第", i, j, "次", "time:", time_cost, "s", )
            print(str(cost_fun[i]), max_range, min_range, real_min_value)
            fitness[j, i] = controller.best_value_all-real_min_value
            time_consume[j, i] = time_cost
            diedai_count = 0
            del controller

    for i in range(num_ex):
        for j in range(11):
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

    # 输出全局最优解和最优值
    #print("章鱼个数:", num_control)
    #print("每条章鱼触手个数:", controller.num_tentacle)
    #print("全局最优值:", controller.best_value_all)
    #print("全局最优位置:", controller.best_position_all)


#1群体4章鱼40触手，群体迭代5次，，一触手50吸盘迭代200次： 5*200 = 1000
