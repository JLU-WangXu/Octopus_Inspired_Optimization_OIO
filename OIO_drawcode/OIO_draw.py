import sys
import numpy as np
import math
import time
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt




path1 = r"/OAAccuracy.csv"
ImuDataTxt_OCT= open(path1,"w")
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
        global row,column,fitness_record_array,num_ex,COUNT,Count_i,position_record_arr,particle_count,iter_count
        global END, num_iteration#

        swarm = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        swarm = np.clip(swarm, min_range, max_range)

        velocity = np.random.uniform(min_range*0.001,max_range*0.001,(self.swarm_size, self.dim))
        p_best_pos = swarm.copy()
        p_best_val = np.array([self.cost_func(p) for p in swarm])
        g_best_idx = np.argmin(p_best_val)
        self.g_best_pos = swarm[g_best_idx].copy()
        self.g_best_val = p_best_val[g_best_idx]
        record_fitness = float('inf')
        for t in range(self.max_iter):

            if COUNT == num_ex - 1 and t == self.max_iter - 1 and iter_count == num_iteration - 1:

            for i in range(self.swarm_size):

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (p_best_pos[i] - swarm[i]) + self.c2 * r2 * (self.g_best_pos - swarm[i])

                swarm[i] += velocity[i]
                # 边界处理
                swarm[i] = np.clip(swarm[i], self.bounds[0], self.bounds[1])
                swarm[i] = np.clip(swarm[i], min_range, max_range)
                #print(iter_count ,num_iteration-1)
                if COUNT == num_ex-1 and t == self.max_iter-1 and iter_count == num_iteration-1 :
                    position_record_arr[particle_count,Count_i,:] = swarm[i]
                    particle_count+=1
                    END = 1

                # 更新个体最优
                if self.cost_func(swarm[i]) < p_best_val[i]:
                    p_best_val[i] = self.cost_func(swarm[i])
                    p_best_pos[i] = swarm[i].copy()

                    # 更新全局最优
                    if p_best_val[i] < self.g_best_val:
                        self.g_best_val = p_best_val[i]

                        self.g_best_pos = p_best_pos[i].copy()
                    if self.g_best_val<record_fitness:
                        record_fitness = self.g_best_val


                    if math.fabs(self.g_best_val) == real_min_value:
                        pass
                      #  print("found")
                        #Found = 1
                        #break

            if COUNT == num_ex - 1:
                fitness_record_array[row][column] = math.fabs(record_fitness - real_min_value)
                row += 1
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
        self.best_values = [float('inf')] * num_tentacle
        self.best_values_ratio = np.ones(num_tentacle)

        self.best_positions = np.zeros((num_tentacle, dim))

        self.global_best_value = np.inf
        self.global_best_position = None
        #调整新位置
        self.center_list = np.zeros((num_tentacle, dim))
        self.radius_list = np.zeros(num_tentacle)
        self.reborn_flag = np.zeros(num_tentacle)

    #触手的初始化，添加，减少
    def generate_random_params(self):
        global max_range, min_range
        params_list = []


        step = (max_range-min_range)/self.num_tentacle
        for i in range(self.num_tentacle):

            swarm_size =50 #min = 10,max=400
            max_iter = 200 #min = 20,max=400
            c1 =1.8
            c2 =1.8
            w = 0.6

            center = np.random.uniform(min_range + i*step, min_range+(i+1)*step, self.dim)#####
            #center = np.random.uniform(min_range, max_range, self.dim)
            #radius = np.random.uniform(1.0, 3.0)*10**int(math.log(max_range,10))
            radius = (max_range-min_range)*np.random.uniform(0.1,0.2)
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
        global Found,tentacles,process_end,process_begin,END,tentacle_arr,Count_i,tentacle_count,radius_arr,octopus,num_tentacles
        for i, params in enumerate(self.params_list):
            tentacles =i
            #print(octopus, tentacles)
            swarm_size, max_iter, c1, c2, w, center, radius = params
            # 创建一个Tentacles对象
            tentacle = Tentacles(self.cost_func, self.dim, swarm_size, max_iter, c1, c2, w, center, radius)

            # 运行Tentacles优化算法
            best_position, best_value = tentacle.optimize()

            if self.reborn_flag[i] == 1:
                self.best_values[i] = best_value
                self.best_positions[i] = best_position
                self.reborn_flag[i] = 0
                # 记录每个Tentacles的最优解和最优值
            elif best_value < self.best_values[i]:
                self.best_values[i] = best_value
                self.best_positions[i] = best_position


            if END == 1:
                print("第",octopus, i, "个触手位置", tentacle.center, "半径", tentacle.radius,"size",tentacle.swarm_size)
                tentacle_arr[i+(octopus*num_tentacles), Count_i,:] = tentacle.center
                radius_arr[i+(octopus*num_tentacles),Count_i] = tentacle.radius
                Swarm_list.append(tentacle.swarm_size)


            # 更新全局最优解和最优值
            if self.best_values[i] < self.global_best_value:
                self.global_best_value = self.best_values[i]
                self.global_best_position = self.best_positions[i]
            if Found == 1:
                print('\n')
                break
        process_end = time.time()


    #根据运行结果，群体信息进行调整
    def adjust_tentacle(self,paradise):

        global max_range,min_range,COUNT
        global END,Swarm_list,Swarm_size_all

        best_value_iteration = min(self.best_values)
        worst_value_iteration = max(self.best_values)
        total_range = max_range-min_range
        swarm_size = 0
        for i, params in enumerate(self.params_list):

            if best_value_iteration != worst_value_iteration:
                self.best_values_ratio[i] = (self.best_values[i] - worst_value_iteration) / ( best_value_iteration - worst_value_iteration)  # 归一化
                self.best_values_ratio[i] = self.best_values_ratio[i] * (1.6 - 0.4) + 0.4  


            swarm_size, max_iter, c1, c2, w, center, radius = params
            swarm_size = np.round(swarm_size * self.best_values_ratio[i]).astype(int)
            max_iter = np.round(max_iter * self.best_values_ratio[i]).astype(int)
            radius = np.round(radius * self.best_values_ratio[i]).astype(int)
            center = self.best_positions[i]
            swarm_size = np.clip(swarm_size, 10, 400)
            radius = np.clip(radius,total_range*0.02,total_range*0.3)
            max_iter = np.clip(max_iter,20,400).astype(int)

            if swarm_size < 12 or max_iter < 30:
                self.reborn_flag[i] = 1

            elif i>0 :


                for j in range(i):
                    distance = euclidean_distance(self.center_list[j], center)
                    distance -= self.radius_list[j]

                    if distance < radius:
                        if self.best_values[i]>self.best_values[j] and self.reborn_flag[i] ==0 :
                            self.reborn_flag[i] = 1
                            center = np.random.uniform(min_range, max_range, self.dim)


                        elif self.reborn_flag[j] == 0:

                            temp_swarm_size, temp_max_iter, c1, c2, w, temp_center, temp_radius =self.params_list[j]
                            self.reborn_flag[j] = 1
                            temp_center = np.random.uniform(min_range, max_range, self.dim)
                            for k in range(self.dim):
                                if np.random.randn() < 0.5:
                                    temp_center[k] = np.random.uniform(min_range,self.global_best_position[k] - total_range * 0.08)
                                else:
                                    temp_center[k] = np.random.uniform(self.global_best_position[k] + total_range * 0.08,max_range)
                                self.params_list[j] = (20, 40, c1, c2, w, temp_center, total_range * np.random.uniform(0.06, 0.08))

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



#章鱼群
process_begin = time.time()
process_end = 0
class Controller:
    def __init__(self, num_control, num_iteration, cost_func, dim):
        global num_tentacles
        #初始化
        self.num_control = num_control
        self.num_iteration = num_iteration
        self.cost_func = cost_func
        self.dim = dim
        #tentaclecontrol的实例
        self.num_tentacle = np.random.randint(num_tentacles,num_tentacles+1,self.num_control)
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
        global Found, octopus,COUNT,iter_count
        for j in range(self.num_iteration):

            for i in range(self.num_control):
               # print("#############################第", i, "只章鱼##############################")
                octopus = i
                self.TentaclesControls[i].run_tentacle()

                if self.TentaclesControls[i].global_best_value < self.best_value_all:
                    self.best_value_all = self.TentaclesControls[i].global_best_value
                    self.best_position_all = self.TentaclesControls[i].global_best_position
                if Found == 1:
                    break
            if COUNT==num_ex-1 :
                iter_count+=1
            if Found == 1:
                break
            for i in range(self.num_control):
                self.TentaclesControls[i].adjust_tentacle(self.best_position_all)





def plot_contour(cost_fun, position_record_arr, max_range, min_range, tentacle_arr, radius_arr, x_target,y_target,particle_color=None,
                 tentacle_color=None, square_color=None):
    global octopus,num_tentacles
    if particle_color is None:
        #particle_color = (219/255,49/255,36/255)
        particle_color = 'red'
    if tentacle_color is None:
        #tentacle_color = (135/255, 187/255, 164/255)
        tentacle_color = 'green'
    if square_color is None:
        square_color = (135/255,206/255 , 235/255,1)
        #square_color = 'green'

    x = np.linspace(min_range, max_range, 100)
    y = np.linspace(min_range, max_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = cost_fun([X[i, j], Y[i, j]])
    plt.figure(figsize=(8, 6))
    levels = np.linspace(np.min(Z), np.max(Z), 20)
    contours = plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.jet,zorder=1)
    plt.colorbar(label='Z Value')


    for i in range(tentacle_arr.shape[0]):
        #square = plt.Rectangle((tentacle_arr[i, 0] - radius_arr[i], tentacle_arr[i, 1] - radius_arr[i]),
                             # width=2 * radius_arr[i], height=2 * radius_arr[i], angle=0, color=square_color,
                              # fill=True, alpha=0.7, zorder=1)

        #plt.gca().add_artist(square)
        fill_square = plt.Rectangle((tentacle_arr[i, 0] - radius_arr[i], tentacle_arr[i, 1] - radius_arr[i]),
                                    width=2 * radius_arr[i], height=2 * radius_arr[i], angle=0, color=square_color,
                                    fill=True, alpha=0.5, zorder=2)
        plt.gca().add_artist(fill_square)
        if i<num_tentacles:
            edge_square = plt.Rectangle((tentacle_arr[i, 0] - radius_arr[i], tentacle_arr[i, 1] - radius_arr[i]),
                                        width=2 * radius_arr[i], height=2 * radius_arr[i], angle=0, color='blue',
                                        fill=False, linewidth=2, zorder=3)
            plt.gca().add_artist(edge_square)

        if i>=num_tentacles and i <num_tentacles*2:
            edge_square = plt.Rectangle((tentacle_arr[i, 0] - radius_arr[i], tentacle_arr[i, 1] - radius_arr[i]),
                                        width=2 * radius_arr[i], height=2 * radius_arr[i], angle=0, color='yellow',
                                        fill=False, linewidth=2, zorder=3)
            plt.gca().add_artist(edge_square)
        plt.scatter(tentacle_arr[i, 0], tentacle_arr[i, 1], color=tentacle_color, s=50, marker='x',alpha=1,zorder=4)

    plt.scatter(x_target, y_target, color="white", s=100, marker='*')

    for k in range(position_record_arr.shape[0]):
        plt.scatter(position_record_arr[k, 0], position_record_arr[k, 1], color=particle_color, s=5,zorder=5)



    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot of ' + cost_fun.__name__)

    # 保存图像为文件
    save_path = r"/plot"
    plt.savefig(f"{save_path}\\plot_{i}_{cost_fun.__name__}.png")  # 指定路径和文件名保存图像
    #plt.show()
    plt.close()  # 关闭当前图




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
        x = np.array(x)  # Convert x to a NumPy array
        A = 1
        w = 1 + (x - 1) / 4
        term1 = (np.sin(np.pi * w[0])) ** 2
        term2 = sum([(wi - 1) ** 2 * (1 + 10 * math.pow(math.sin(math.pi * wi + 1), 2)) for wi in w])
        term3 = (w[len(w) - 1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[len(w) - 1]) ** 2)
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
    num_control = 1
    num_iteration = 50
    dim = 2

#################################

    #for i in range(num_control):
        #for j in range(5):
            #ImuDataTxt_tentacles.write(str("octopus")+str(i)+",")
    #ImuDataTxt_tentacles.write('\n')
    Found = 0

    octopus = 0#
    tentacles = 0#
    num_ex = 10


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
    #cost_fun = [Ackley, Griewank, levy, rastrigin, Dixon_Price, Sum_square, Trid, Rosenbrock,
    #            sphere, Rotated_Hyper_Ellipsoid, sum_of_different_powers]
    #max_range_list = [32, 600, 10, 5.12, 10, 10, dim ** 2, 10, 5.12, 65, 1]
    #min_range_list = [-32, -600, -10, -5.12, -10, -10, -dim ** 2, -5, -5.12, -65, -1]
    #real_min_value_list = [0, 0, 0, 0, 0, 0, -dim * (dim + 4) * (dim - 1) / 6, 0, 0, 0, 0]
    cost_fun = [Ackley, Griewank, levy, rastrigin, Schwefel, Dixon_Price, Sum_square, Trid, Rosenbrock,
                sphere, Rotated_Hyper_Ellipsoid, sum_of_different_powers, schaffer_function_n2, Beale, goldstein_price,
                drop_wave, Levy_N_13, easom]
    max_range_list = [32, 600, 10, 5.12, 500, 10, 10, dim ** 2, 10, 5.12, 65, 1, 100, 4.5, 2, 5.12, 10, 100]
    min_range_list = [-32, -600, -10, -5.12, -500, -10, -10, -dim ** 2, -5, -5.12, -65, -1, -100, -4.5, -2, -5.12, -10,
                      -100]
    real_min_value_list = [0, 0, 0, 0, 0, 0, 0, -dim * (dim + 4) * (dim - 1) / 6, 0, 0, 0, 0, 0, 0, 3, -1, 0, -1]

    fitness = np.full((num_ex, len(cost_fun)), -1.0)
    time_consume = np.full((num_ex, len(cost_fun)), -1.0)
    ImuDataTxt_OCT.write("fitness, ")
    ImuDataTxt_OCT.write("time, ")
    ImuDataTxt_OCT.write(str('\n'))

    num_tentacles = 5
    Swarm_list = []
    Swarm_size_all = np.zeros(18)
    END = 0
    Count_i = 0
    COUNT = 0  #
    particle_count = 0
    position_record_arr = np.empty((num_tentacles*num_control*1000, len(cost_fun), dim))  # (4*5*50)
    tentacle_arr = np.empty((num_tentacles*num_control, len(cost_fun), dim))
    tentacle_count = 0
    radius_arr = np.empty((num_tentacles*num_control, len(cost_fun)))
    iter_count = 0
    x_list = [0,0,1,0,420.9687,1,0,2,1,0,0,0,0,3,0,0,1,math.pi]
    y_list = [0,0,1,0,420.9687,2**(-1/2),0,2,1,0,0,0,0,0.5,-1,0,1,math.pi]
    row = 0##
    column = 0#
    fitness_record_array = np.full((20000, len(cost_fun)), -1.23456)#

    for i in range(18):
        print("###########################################################")
        Count_i = i
        for j in range(num_ex):


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
            print("###################第", i, j, "次", "time:", time_cost, "s","best_value:",controller.best_value_all,"functionName:", str(cost_fun[i]))

            fitness[j, i] = controller.best_value_all-real_min_value
            diedai_count = 0
            time_consume[j, i] = time_cost


            del controller
        Swarm_size_all[i] = int(sum(Swarm_list))


        #print("Swarm_size 544",int(Swarm_size_all[i]),'\n')
        position_record_arr = position_record_arr[:int(Swarm_size_all[i]), :, :]
        print(Swarm_list)
        print(radius_arr[:, i])

        plot_contour(cost_fun[i], position_record_arr[:, i, :], max_range_list[i], min_range_list[i],tentacle_arr[:,i,:],radius_arr[:,i],x_list[i],y_list[i])
        column += 1
        row = 0
        particle_count = 0
        Swarm_list=[]
        iter_count = 0
        END = 0
        position_record_arr = np.empty((num_tentacles * 1000, len(cost_fun), dim))

    for i in range(num_ex):
        for j in range(18):
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

    df = pd.DataFrame(fitness_record_array)
    file_path = r'/OAshoulian.csv'
    df.to_csv(file_path, index=False)


