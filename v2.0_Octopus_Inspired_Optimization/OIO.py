import numpy as np
import math
import threading
from numba import jit

#Calculate Euclidean distance in any dimension

def euclidean_distance(point1, point2):
    return math.sqrt(sum([(x2 - x1) ** 2 for x1, x2 in zip(point1, point2)]))


#tentacle
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
        swarm = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        swarm = np.clip(swarm, min_range, max_range)
        velocity = np.random.uniform(min_range*0.001,max_range*0.001,(self.swarm_size, self.dim))
        p_best_pos = swarm.copy()
        p_best_val = np.array([self.cost_func(p) for p in swarm])
        g_best_idx = np.argmin(p_best_val)
        self.g_best_pos = swarm[g_best_idx].copy()
        self.g_best_val = p_best_val[g_best_idx]

        for t in range(self.max_iter):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocity = (self.w * velocity + 
                        self.c1 * r1 * (p_best_pos - swarm) + 
                        self.c2 * r2 * (self.g_best_pos - swarm))
        
            swarm += velocity
            swarm = np.clip(swarm, self.bounds[0], self.bounds[1])
            swarm = np.clip(swarm, min_range, max_range)
        
            new_costs = np.array([self.cost_func(p) for p in swarm])
            improved = new_costs < p_best_val
            p_best_val[improved] = new_costs[improved]
            p_best_pos[improved] = swarm[improved]
        
            g_best_idx = np.argmin(p_best_val)
            if p_best_val[g_best_idx] < self.g_best_val:
                self.g_best_val = p_best_val[g_best_idx]
                self.g_best_pos = p_best_pos[g_best_idx].copy()

        return self.g_best_pos, self.g_best_val


#octopus
class Octopus:
    def __init__(self, num_tentacles, cost_func, dim):
        #initialization
        self.num_tentacles = num_tentacles
        self.cost_func = cost_func
        self.dim = dim
        #Parameter matrix, an example of tentacle structure
        self.params_list = self.generate_tentacles()
        #Result parameters
        self.best_values = [float('inf')] * num_tentacles
        self.best_values_ratio = np.ones(num_tentacles)

        self.best_positions = np.zeros((num_tentacles, dim))

        self.global_best_value = np.inf
        self.global_best_position = None
        #Adjust to a new location
        self.center_list = np.zeros((num_tentacles, dim))
        self.radius_list = np.zeros(num_tentacles)
        self.reborn_flag = np.zeros(num_tentacles)

    #Initialization, addition, and reduction of tentacles
    def generate_tentacles(self):
        global max_range, min_range
        params_list = []

        step = (max_range-min_range)/self.num_tentacles
        for i in range(self.num_tentacles):

            swarm_size = 50#min = 10,max=400
            max_iter = 200#min = 20,max=400
            c1 =1.8
            c2 =1.8
            w = 0.6

            center = np.random.uniform(min_range + i*step, min_range+(i+1)*step, self.dim)
            radius = (max_range-min_range)*np.random.uniform(0.1,0.2)
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


    #Run according to the parameters and calculate the results
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

            # Update the global optimal solution and optimal value
            if self.best_values[i] < self.global_best_value:
                self.global_best_value = self.best_values[i]
                self.global_best_position = self.best_positions[i]
            print("The", i, "-th tentacle position", tentacle.center, "size:", tentacle.swarm_size, "iterations", tentacle.max_iter, "Found value",
                  self.best_values[i],"best position",self.best_positions[i])


    #Adjust group information based on operational results
    def adjust_tentacles(self, paradise):
        global max_range, min_range
        best_value_iteration = min(self.best_values)
        worst_value_iteration = max(self.best_values)
        total_range = max_range - min_range

        for i, params in enumerate(self.params_list):

            if best_value_iteration != worst_value_iteration:
                self.best_values_ratio[i] = (self.best_values[i] - worst_value_iteration) / (
                            best_value_iteration - worst_value_iteration)  # normalization（0,1）
                self.best_values_ratio[i] = self.best_values_ratio[i] * (1.6 - 0.4) + 0.4  # Projection onto（0.4.,1.6）

            swarm_size, max_iter, c1, c2, w, center, radius = params
            swarm_size = np.round(swarm_size * self.best_values_ratio[i]).astype(int)
            max_iter = np.round(max_iter * self.best_values_ratio[i]).astype(int)
            radius = np.round(radius * self.best_values_ratio[i]).astype(int)
            center = self.best_positions[i]
            swarm_size = np.clip(swarm_size, 10, 400)
            radius = np.clip(radius,total_range*0.02,total_range*0.08)
            max_iter = np.clip(max_iter,20,400).astype(int)

            if swarm_size < 12 or max_iter < 20:
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


class OctopusSwarm:
    def __init__(self, num_octopus, num_iteration, cost_func, dim):
        #initialization
        self.num_octopus = num_octopus
        self.num_iteration = num_iteration
        self.cost_func = cost_func
        self.dim = dim
        self.lock = threading.Lock()
        #Example of Octopus
        self.num_tentacles = np.random.randint(10, 11, self.num_octopus)

        #Result parameters
        self.best_value_all = np.inf
        self.best_position_all = None
        self.octopus_swarm = self.generate_octopus_swarm()

    def generate_octopus_swarm(self):
        octopus_swarm_list = []
        for i in range(self.num_octopus):
            octopus_swarm_list.append(Octopus(self.num_tentacles[i], self.cost_func, self.dim))
        return octopus_swarm_list


    def add_octopus(self,num_tentacle):
        self.octopus_swarm.append(Octopus(num_tentacle, self.cost_func, self.dim))

    #Delete the n-th octopus
    def remove_octopus(self,n):
        del self.octopus_swarm[n]

    def run_octopus(self, octopus, i):
        print(f"#############################The {i} -th octopus##############################")
        octopus.run_tentacles()

        with self.lock:
            if octopus.global_best_value < self.best_value_all:
                self.best_value_all = octopus.global_best_value
                self.best_position_all = octopus.global_best_position

    def run_octopus_swarm(self):
        no_improvement_count = 0
        best_value_history = []
        
        for j in range(self.num_iteration):
            threads = []
            for i, octopus in enumerate(self.octopus_swarm):
                thread = threading.Thread(target=self.run_octopus, args=(octopus, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            for octopus in self.octopus_swarm:
                octopus.adjust_tentacles(self.best_position_all)
            
            best_value_history.append(self.best_value_all)
            if len(best_value_history) > 10 and self.best_value_all >= best_value_history[-10]:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            if no_improvement_count >= 50:
                print(f"Stop early from iteration {j+1}，Because there has been no improvement after 50 consecutive iterations")
                break
            
            print(f"iteration {j+1}: optimum：{self.best_value_all}     best position：{self.best_position_all}")



if __name__ == "__main__":
    @jit(nopython=True)
    def cost_func(x):#rastrigin
        A = 10
        n = len(x)
        return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

    max_range = 32
    min_range = -32
    real_min_value = 0

    num_octopus = 1
    num_iteration = 50
    dim = 2

    # Create Octopus object
    octopus_swarm = OctopusSwarm(num_octopus, num_iteration, cost_func, dim)

    # Run optimization algorithms
    octopus_swarm.run_octopus_swarm()

    # Output global optimal solution and optimal value
    print("Number of Octopuses:", num_octopus)
    print("The number of tentacles per octopus:", octopus_swarm.num_tentacles)
    print("Global optimal value:", octopus_swarm.best_value_all)
    print("Global optimal position:", octopus_swarm.best_position_all)