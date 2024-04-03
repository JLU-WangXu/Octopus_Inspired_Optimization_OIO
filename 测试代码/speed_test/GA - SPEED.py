import numpy as np
import matplotlib.pyplot as plt
import time
import math
path = r"C:\Users\broth\Desktop\GA-speed.csv"
ImuDataTxt_GA= open(path,"w")

def rastrigin(x):  # (-5.12,5.12)
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

# 生成随机种群
def generate_population(pop_size, dim):
    global min_range,max_range
    return np.random.uniform(min_range, max_range, size=(pop_size, dim))

# 计算种群适应度
def compute_fitness(population,cost_fun):

    return np.apply_along_axis(cost_fun, 1, population)



def selection(population, fitness):
    # 对适应度进行排序，并使用排序的索引（即排名）作为选择概率的基础
    ranks = np.argsort(fitness)
    # 生成一个与排名成正比的选择概率分布
    rank_prob = np.linspace(1, len(population), num=len(population))
    rank_prob = rank_prob / rank_prob.sum()
    # 根据排名概率选择个体
    idx = np.random.choice(len(population), size=len(population), p=rank_prob[ranks])
    return population[idx]
# 交叉操作
def crossover(population, crossover_rate):
    global min_range,max_range
    new_population = []
    for i in range(0, len(population), 2):
        parent1 = population[i]
        parent2 = population[i+1]
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            new_population.extend([child1, child2])
        else:
            new_population.extend([parent1, parent2])
    new_population = np.clip(new_population,min_range,max_range)
    return np.array(new_population)

# 变异操作
def mutation(population, mutation_rate):
    global min_range,max_range
    for i in range(len(population)):
        for j in range(len(population[i])):
            if np.random.rand() < mutation_rate:
                population[i][j] = np.random.uniform(min_range, max_range)
    return population

# 主函数
def genetic_algorithm(pop_size, dim, max_iter, crossover_rate, mutation_rate, cost_fun):
    global real_min_value
    population = generate_population(pop_size, dim)
    best_solution = None
    best_fitness = np.inf

    for _ in range(max_iter):
        fitness = compute_fitness(population, cost_fun)
        best_idx = np.argmin(fitness)
        current_best_fitness = fitness[best_idx]
        current_best_solution = population[best_idx]

        if current_best_fitness < best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness

        if np.any(fitness - real_min_value<1e-9):  # 检查是否有任何适应度等于最优值
            print("Found")
            break

        population = selection(population, fitness)
        population = crossover(population, crossover_rate)
        population = mutation(population, mutation_rate)

    return best_solution, best_fitness


# 设置算法参数




# 运行遗传算法


# 打印结果

if __name__ == '__main__':
    dim = 2
    max_iter = 1000
    pop_size = 400

    crossover_rate = 0.8
    mutation_rate = 0.01

    ImuDataTxt_GA.write("Fitness"+",")
    ImuDataTxt_GA.write("Time" + ",")
    ImuDataTxt_GA.write('\n')

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

            Found = 0

            max_range = max_range_list[i]
            min_range = min_range_list[i]
            real_min_value = real_min_value_list[i]

            begin_time = time.time()
            best_solution, best_fitness = genetic_algorithm(pop_size, dim, max_iter, crossover_rate, mutation_rate,cost_fun[i])
            end_time = time.time()

            time_cost = end_time - begin_time
            print("###################第",i, j, "次", "time:", time_cost, "s",)

            fitness[j, i] = best_fitness - real_min_value
            time_consume[j, i] = time_cost



    for i in range(num_ex):
        for j in range(17):
            ImuDataTxt_GA.write(str(fitness[i, j]) + ",")
            ImuDataTxt_GA.write(str(time_consume[i, j]) + ",")
        ImuDataTxt_GA.write('\n')


    print(fitness)
    print(time_consume)
    fit_mean = np.mean(fitness)
    fit_std = np.std(fitness)
    time_mean = np.mean(time_consume)
    time_std = np.std(time_consume)
    print(fit_mean-real_min_value, fit_std, time_mean, time_std)