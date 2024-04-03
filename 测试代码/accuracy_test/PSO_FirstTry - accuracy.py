import numpy as np
import matplotlib.pyplot as plt
import time
import math
path = r"C:\Users\broth\Desktop\PSO-accuracy.csv"
ImuDataTxt_PSO= open(path,"w")


class Particle:
    def __init__(self, dim):
        global max_range,min_range
        self.velocity = np.random.uniform(low=0, high=0, size=dim)  # 降低速度的范围
        self.position = np.random.uniform(low=min_range, high=max_range, size=dim)  # 缩小位置的范围
        self.best_score = float('inf')
        self.best_position = np.copy(self.position)

class PSO:
    def __init__(self, dim, scale, iteration):
        self.dim = dim
        self.scale = scale
        self.iteration = iteration
        self.particles = [Particle(dim) for _ in range(scale)]
        self.global_best_position = np.zeros(dim)
        self.global_best_score = float('inf')
        self.w = 0.6  # 惯性权重
        self.c1 = 1.8  # 个体认知因子
        self.c2 = 1.8  # 社会认知因子


    def velocity_update(self, particle):


        inertia_V = particle.velocity * self.w
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        individual = self.c1 * r1 * (particle.best_position - particle.position)
        society = self.c2 * r2 * (self.global_best_position - particle.position)
        return inertia_V + individual + society

    def optimize(self,cost_fun):
        global Found,min_range,max_range,real_min_value
        for i in range(self.iteration):
            for particle in self.particles:
                particle.position = np.clip(particle.position,min_range, max_range)

                score = cost_fun(particle.position)###############################################

                if score < particle.best_score:
                    particle.best_position = np.copy(particle.position)
                    particle.best_score = score
                if score < self.global_best_score:
                    self.global_best_position = np.copy(particle.position)
                    self.global_best_score = particle.best_score
                #print(self.global_best_score)
                if self.global_best_score == real_min_value:
                    print("已经找到")
                    Found = 1
                    break
                particle.velocity = self.velocity_update(particle)
                particle.position += particle.velocity

            if Found == 1:
                break



Found = 0



def rastrigin(x):  # rastrigin (-5.12,5.12)
    A = 10
    n = len(x)
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def sphere(x):  # (-5.12,5.12)
    return sum([(xi ** 2) for xi in x])


def Ackley(x):  # (-32,32)
    a = 20
    b = 0.2
    c = 2 * math.pi
    n = len(x)
    return -a * math.exp(-b * math.sqrt(1 / n * sum([(xi ** 2) for xi in x]))) - math.exp(
        1 / n * sum([(math.cos(c * xi)) for xi in x])) + a + math.exp(1)


def levy(x):  # Levy (-10, 10)
    A = 1
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0])) ** 2
    term2 = sum([(wi - 1) ** 2 * (1 + 10 * math.pow(math.sin(math.pi * wi + 1), 2)) for wi in w])
    term3 = (w[len(w) - 1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[len(w) - 1]) ** 2)
    return term1 + term2 + term3


def Griewank(x):  # (-600,600)
    n = len(x)
    sum_term = np.sum([xi ** 2 for xi in x])
    prod_term = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return 1 / 4000 * sum_term - prod_term + 1


def Schwefel(x):  # (-500,500)
    term1 = 418.9829 * len(x)
    term2 = sum([(xi * math.sin(math.sqrt(math.fabs(xi)))) for xi in x])
    return term1 - term2


def Dixon_Price(x):  # (-10,10)
    result = (x[0] - 1) ** 2
    for i in range(1, len(x)):
        result += i * (2 * x[i] ** 2 - x[i - 1]) ** 2
    return result


def Styblinski_Tang(x):  # (-5,5)
    return 1 / 2 * sum([(math.pow(xi, 4) - 16 * xi ** 2 + 5 * xi) for xi in x])


def Sum_square(x):  # (-5.12,5.12)
    return sum([i * x[i] ** 2 for i in range(len(x))])


def Trid(x):  # (-d**2,d**2)
    term1 = sum([((xi - 1) ** 2) for xi in x])
    term2 = 0
    for i in range(1, len(x)):
        term2 += x[i] * x[i - 1]
    return term1 - term2


def Rosenbrock(x):  # (-5,10)
    sum = 0
    for i in range(len(x) - 1):
        sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return sum


def Rotated_Hyper_Ellipsoid(x):  # (-65.536,65.536)
    sum = 0
    for i in range(len(x)):
        for j in range(i):
            sum += x[j] ** 2
    return sum


def schaffer_function_n2(x):  # (-100,100)
    return 0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2


def Beale(x):  # (-4.5,4.5)
    term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
    term2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
    term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    return term1 + term2 + term3


def goldstein_price( x):  # (-2,2)
    term1 = 1 + (x[0] + x[1] + 1) ** 2 * (
            19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
    term2 = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (
            18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
    return term1 * term2


def drop_wave(x):  # (-5.12,5.12)
    term1 = 1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))
    term2 = 0.5 * (x[0] ** 2 + x[1] ** 2) + 2
    return -term1 / term2


def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)


if __name__ == '__main__':
    dim = 2
    iteration = 1000
    scale = 400


    ImuDataTxt_PSO.write("Fitness"+",")
    ImuDataTxt_PSO.write("Time" + ",")
    ImuDataTxt_PSO.write('\n')
    #start_time = time.time()
    #Pso = PSO(dim, scale, iteration)
    #Pso.optimize()
    #end_time = time.time()
    #cost_time = end_time - start_time
    num_ex = 3
    max_range = 0
    min_range = 0
    real_min_value = 0
    cost_fun = [Ackley, Griewank, levy, rastrigin, Schwefel, Dixon_Price, Styblinski_Tang, Sum_square, Trid, Rosenbrock,
                sphere, Rotated_Hyper_Ellipsoid, schaffer_function_n2, Beale, goldstein_price, drop_wave, easom]  #
    max_range_list = [32, 600, 10, 5.12, 500, 10, 5, 10, dim ** 2, 10, 5.12, 65, 100, 4.5, 2, 5.12, 100]
    min_range_list = [-32, -600, -10, -5.12, -500, -10, -5, -10, -dim ** 2, -5, -5.12, -65, -100, -4.5, -2, -5.12, -100]
    real_min_value_list = [0, 0, 0, 0, 0, 0, -39.16599 * dim, 0, -dim * (dim + 4) * (dim - 1) / 6, 0, 0, 0, 0, 0, 3, -1,
                           -1]
    fitness = np.full((num_ex, len(cost_fun)), -1.0)
    time_consume = np.full((num_ex, len(cost_fun)), -1.0)
    for i in range(17):
        for j in range(num_ex):
            # 创建PSOControl对象
            # print("第",j,"次")
            COUNT = j
            Found = 0

            max_range = max_range_list[i]
            min_range = min_range_list[i]
            real_min_value = real_min_value_list[i]

            begin_time = time.time()
            Pso = PSO(dim, scale, iteration)
            Pso.optimize(cost_fun[i])
            end_time = time.time()

            time_cost = end_time - begin_time
            print("###################第",i, j, "次", "time:", time_cost, "s")

            fitness[j, i] =Pso.global_best_score - real_min_value
            time_consume[j, i] = time_cost

            del Pso

    for i in range(num_ex):
        for j in range(17):
            ImuDataTxt_PSO.write(str(fitness[i, j]) + ",")
            ImuDataTxt_PSO.write(str(time_consume[i, j]) + ",")
        ImuDataTxt_PSO.write('\n')


    print(fitness)
    print(time_consume)
    fit_mean = np.mean(fitness)
    fit_std = np.std(fitness)
    time_mean = np.mean(time_consume)
    time_std = np.std(time_consume)
    print(fit_mean-real_min_value, fit_std, time_mean, time_std)

