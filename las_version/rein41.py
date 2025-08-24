import numpy as np
import math





#
#Standard version model
#Suction cup: PSO algorithm, can be replaced with 1. Any group optimization algorithm 2. Gradient optimization algorithm
#The interaction between suction cups and tentacles satisfies the Cooperative Co evolution Model
#Tentacles: Part of the Implementation of Swarm Intelligence+Individual Reinforcement Learning
#The interaction between tentacles and individuals satisfies the Master Slave Cooperative Model
#Individual: Reinforcement learning, used to determine the way and state of updates
#








#Calculate Euclidean distance in any dimension
def euclidean_distance(point1, point2):
    return math.sqrt(sum([(x2 - x1) ** 2 for x1, x2 in zip(point1, point2)]))


#tentacle
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
        # Initialize particle position and velocity
        swarm = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        p_best_pos = swarm.copy()
        p_best_val = np.array([self.cost_func(p) for p in swarm])
        g_best_idx = np.argmin(p_best_val)
        self.g_best_pos = swarm[g_best_idx].copy()
        self.g_best_val = p_best_val[g_best_idx]

        for t in range(self.max_iter):
            for i in range(self.swarm_size):
                # Update particle velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (p_best_pos[i] - swarm[i]) + self.c2 * r2 * (self.g_best_pos - swarm[i])

                # Update particle position
                swarm[i] += velocity[i]

                # Boundary treatment
                swarm[i] = np.clip(swarm[i], self.bounds[0], self.bounds[1])

                # Update individual optimal
                if self.cost_func(swarm[i]) < p_best_val[i]:
                    p_best_val[i] = self.cost_func(swarm[i])
                    p_best_pos[i] = swarm[i].copy()

                    # Update global optimum
                    if p_best_val[i] < self.g_best_val:
                        self.g_best_val = p_best_val[i]
                        self.g_best_pos = p_best_pos[i].copy()


        return self.g_best_pos, self.g_best_val


#octopus
class PSOControl:
    def __init__(self, num_pso, cost_func, dim):
        #initialization
        self.num_pso = num_pso
        self.cost_func = cost_func
        self.dim = dim
        #parameter matrix, an example of tentacle structure
        self.params_list = self.generate_random_params()
        #Result parameters
        self.best_values = np.zeros(num_pso)
        self.best_values_ratio = np.ones(num_pso)
        self.best_positions = np.zeros((num_pso, dim))
        self.global_best_value = np.inf
        self.global_best_position = None
        #Status parameters
        self.state = 0


    #Initialization, addition, and reduction of tentacles
    def generate_random_params(self):
        params_list = []
        for _ in range(self.num_pso):
            swarm_size = np.random.randint(20, 100)
            max_iter = np.random.randint(50, 200)
            c1 = np.random.uniform(1.0, 3.0)
            c2 = np.random.uniform(1.0, 3.0)
            w = np.random.uniform(0.5, 1.0)
            center = np.random.uniform(-5.0, 5.0, self.dim)
            radius = np.random.uniform(1.0, 10.0)
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


    #Run according to the parameters and calculate the results
    def run_pso(self):
        for i, params in enumerate(self.params_list):
            swarm_size, max_iter, c1, c2, w, center, radius = params
            # Create a object
            pso = PSO(self.cost_func, self.dim, swarm_size, max_iter, c1, c2, w, center, radius)

            # Run optimization algorithm
            best_position, best_value = pso.optimize()

            # Record the optimal solution and optimal value
            self.best_values[i] = best_value
            self.best_positions[i] = best_position

            # Update the global optimal solution and optimal value
            if best_value < self.global_best_value:
                self.global_best_value = best_value
                self.global_best_position = best_position

            print("The result of the"+str(i)+"-th tentacle is：")
            print("optimal value：", self.best_values[i])
            print("optimal position：", self.best_positions[i])





    #Adjust group information based on operational results
    def adjust_pso(self,paradise,state):
        best_value_iteration = min(self.best_values)
        worst_value_iteration = max(self.best_values)


        for i, params in enumerate(self.params_list):
            if self.state == 1:#Duplicate condition search
                a = 1

            elif self.state == 2:#Expansion and contraction search
                if best_value_iteration != worst_value_iteration:
                    self.best_values_ratio[i] = (self.best_values[i] - worst_value_iteration) / (best_value_iteration - worst_value_iteration)  # normalization（0,1）
                    self.best_values_ratio[i] = self.best_values_ratio[i] * (1.3 - 0.7) + 0.7  # Projection onto（0.7,1.3）

                swarm_size, max_iter, c1, c2, w, center, radius = params
                swarm_size = np.round(swarm_size * self.best_values_ratio[i]).astype(int)
                max_iter = np.round(max_iter * self.best_values_ratio[i]).astype(int)
                radius = np.round(radius * self.best_values_ratio[i]).astype(int)
                if np.random.random() < 0.05:#ε=0.05
                    center = center + np.random.uniform(-1, 1, size=self.dim) * (center - self.best_positions[i])
                else:
                    # Using a probability of 1- ε, select the action with the highest evaluation value in the current state
                    center = self.best_positions[i]

                if swarm_size > 0 and max_iter > 0 and radius > 0:
                    self.params_list[i] = (swarm_size, max_iter, c1, c2, w, center, radius)
                else :
                    self.params_list[i] = (10, 3, c1, c2, w, center,5)

            elif self.state == 3:#Migration Search
                swarm_size, max_iter, c1, c2, w, center, radius = params
               #center = self.best_positions[i] + np.random.uniform(-0.02, 0.1, size=self.dim) * (self.best_positions[i] - paradise)
                center = paradise + np.random.uniform(-0.02, 0.1, size=self.dim) * (paradise - self.best_positions[i])
                self.params_list[i] = (swarm_size, max_iter, c1, c2, w, center, radius)


#octopus group
class Controller:
    def __init__(self, num_control, num_iteration, cost_func, dim):
        #initialization
        self.num_control = num_control
        self.num_iteration = num_iteration
        self.cost_func = cost_func
        self.dim = dim
        #Examples of psocontrol
        self.num_pso = np.random.randint(1, 6, self.num_control)
        self.PSOControls = self.generate_psocontrol()
        #Result parameters
        self.best_value_all = np.inf
        self.best_position_all = None
        #reinforcement
        self.octopus_statues = np.ones(self.num_control)
        self.actions = np.ones(self.num_control)
        self.octopus_statues_list =  self.generate_octopus_statues_list()
        self.actions_list =  self.generate_actions_list()
        ## SARSA parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # ε - growth exploration rate
        ##  Q-value table
        self.q_values_list = self.generate_Q()



    def generate_psocontrol(self):
        psocontrol_list = []
        for i in range(self.num_control):
            psocontrol_list.append(PSOControl(self.num_pso[i], self.cost_func, self.dim))
        return psocontrol_list

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
        # ε - growth strategy selection action
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions_list[i])  # Randomly select actions
        else:
            # Choose the action with the highest Q value
            max_q_value = max([self.q_values_list[i][(state, a)] for a in self.actions_list[i]])
            return np.random.choice([a for a in self.actions_list[i] if self.q_values_list[i][(state, a)] == max_q_value])

    def update_q_value(self, i, state, action, reward, next_state, next_action):
        # Update Q value according to SARSA update formula
        current_q_value = (self.q_values_list[i])[(state, action)]
        next_q_value = self.q_values_list[i][(next_state, next_action)]
        td_error = -reward + self.gamma * next_q_value - current_q_value#What value should be used for reward? The minimum value is still the difference from the previous result
        self.q_values_list[i][(state, action)] += self.alpha * td_error

    def run_psocontrol(self):
        j = 0
        while (j < self.num_iteration) and (self.best_value_all > 0):
            j += 1
            print("The",j,"round")
            for i in range(self.num_control):
                print("##############################################################################")
                print("The result of the"+str(i)+"-th octopus test is")
                #reinforcement
                state = self.octopus_statues[i]
                action = self.actions[i]

                #Logical operation
                self.PSOControls[i].run_pso()
                if self.PSOControls[i].global_best_value < self.best_value_all:
                    self.best_value_all = self.PSOControls[i].global_best_value
                    self.best_position_all = self.PSOControls[i].global_best_position

                reward = -self.best_value_all
                next_action = self.choose_action(i,state)
                next_state = next_action
                self.update_q_value(i, state, action, reward, next_state, next_action)
                self.octopus_statues[i] = next_state
                self.actions[i] = next_action



            for i in range(self.num_control):
                self.PSOControls[i].adjust_pso(self.best_position_all,self.octopus_statues_list[i])



if __name__ == "__main__":
    def cost_func(x):#rastrigin
        A = 10
        n = len(x)
        return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


    num_control = 3
    num_iteration = 5
    dim = 2

    # Create Control Object
    controller = Controller(num_control, num_iteration, cost_func, dim)

    # Run optimization algorithms
    controller.run_psocontrol()

    # Output global optimal solution and optimal value
    print("Number of Octopuses:", num_control)
    print("The number of tentacles per octopus:", controller.num_pso)
    print("Global optimal value:", controller.best_value_all)
    print("Global optimal position:", controller.best_position_all)



