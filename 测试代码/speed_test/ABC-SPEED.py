import sys
import numpy as np
import math
import time
from functools import reduce
path1 = r"C:\Users\broth\Desktop\ABC_speed.csv"
ImuDataTxt_ABC= open(path1,"w")


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


# 定义 ABC 算法函数
def abc_algorithm(dimension, max_iter, num_bees, func):
    global max_range,min_range,real_min_value,Found
    # 初始化蜜蜂群，随机生成 num_bees 个蜜蜂在搜索空间内
    bees = np.random.uniform(min_range, max_range, (num_bees, dimension))
    # 计算每只蜜蜂的适应度值（目标函数值）
    values = np.apply_along_axis(func, 1, bees)
    # 找到初始状态下的最佳蜜蜂和其对应的适应度值
    best_bee = bees[np.argmin(values)]
    best_value = np.min(values)

    # 开始迭代
    for i in range(max_iter):
        # 更新蜜源
        for j in range(num_bees):
            # 随机选择一个蜜蜂索引 k，保证不等于当前蜜蜂的索引 j
            k = np.random.randint(0, num_bees - 1)
            while k == j:
                k = np.random.randint(0, num_bees - 1)
            # 生成随机向量 phi，用于更新当前蜜蜂的位置
            phi = np.random.uniform(-1, 1, dimension)
            # 根据 ABC 算法更新当前蜜蜂的位置
            new_bee = bees[j] + phi * (bees[j] - bees[k])
            new_bee = np.clip(new_bee,min_range,max_range)#应该不用加
            # 计算新位置的适应度值
            new_value = func(new_bee)
            if new_value - real_min_value<1e-9:
                print("Found",new_value)
                Found = 1
                break
            # 若新位置的适应度值更好，则更新蜜蜂位置和适应度值
            if new_value < values[j]:
                bees[j] = new_bee
                values[j] = new_value
            else:
                # 否则随机选择一个蜜蜂，并与当前蜜蜂比较适应度值
                limit_bee = np.random.randint(0, num_bees - 1)
                while limit_bee == j:
                    limit_bee = np.random.randint(0, num_bees - 1)
                if values[j] < values[limit_bee]:
                    bees[j] = new_bee
                    values[j] = new_value


        # 选择蜜蜂中的最佳蜜蜂
        best_index = np.argmin(values)
        if values[best_index] < best_value:
            best_bee = bees[best_index]
            best_value = values[best_index]
        if Found == 1:
            break
    return best_bee, best_value



# 设置算法参数和搜索空间范围


# 使用 ABC 算法优化 Rastrigin 函数


# 打印找到的最佳解和对应的适应度值

if __name__ == '__main__':

    dim = 2
    max_iter = 1000
    num_bees = 400



    ImuDataTxt_ABC.write("Fitness"+",")
    ImuDataTxt_ABC.write("Time" + ",")
    ImuDataTxt_ABC.write('\n')

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
            best_bee, best_value = abc_algorithm(dim, max_iter, num_bees,cost_fun[i])
            end_time = time.time()

            time_cost = end_time - begin_time
            print("###################第",i, j, "次", "time:", time_cost, "s",)

            fitness[j, i] = best_value - real_min_value
            time_consume[j, i] = time_cost



    for i in range(num_ex):
        for j in range(17):
            ImuDataTxt_ABC.write(str(fitness[i, j]) + ",")
            ImuDataTxt_ABC.write(str(time_consume[i, j]) + ",")
        ImuDataTxt_ABC.write('\n')


    print(fitness)
    print(time_consume)
    fit_mean = np.mean(fitness)
    fit_std = np.std(fitness)
    time_mean = np.mean(time_consume)
    time_std = np.std(time_consume)
    print(fit_mean-real_min_value, fit_std, time_mean, time_std)