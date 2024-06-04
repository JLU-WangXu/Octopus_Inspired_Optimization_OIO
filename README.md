# Octopus-Algorithm
This project introduces a novel bionic intelligent optimisation algorithm, Octopus-Inspired Optimization (OIO) algorithm, which is inspired by the neural structure of octopus, especially its hierarchical and decentralised interaction properties.The OIO algorithm achieves an effective combination of global and local search by simulating the sensory, decision-making and executive By simulating the octopus' sensory, decision-making and execution capabilities, the OIO algorithm adopts a multi-level hierarchical strategy, including tentacles, suckers, individuals and groups, to achieve an effective combination of global and local search. This hierarchical design not only enhances the flexibility and efficiency of the algorithm, but also significantly improves its search efficiency and adaptability.
![image](https://github.com/JLU-WangXu/Octopus_Inspired_Optimization_OIO/assets/73222849/ea186871-e75f-42be-b993-e810ad3cbbc9)

In performance evaluations, including comparisons with existing mainstream intelligent optimisation algorithms, OIO shows faster convergence and higher accuracy, especially when dealing with multimodal functions and high-dimensional optimisation problems. This advantage is even more pronounced as the required minimum accuracy is higher, with the OIO algorithm showing an average speedup of 2.27 times that of conventional particle swarm optimisation (PSO) and 9.63 times that of differential evolution (DE) on multimodal functions. In particular, when dealing with high-dimensional optimisation problems, OIO achieves an average speed of 10.39 times that of DE, demonstrating its superior computational efficiency. In addition, the OIO algorithm also shows a reduction of about 5% in CPU usage efficiency compared to PSO, which is reflected in the efficiency of CPU resource usage also shows its efficiency. These features make the OIO algorithm show great potential in complex optimisation problems, and it is especially suitable for application scenarios that require fast, efficient and robust optimisation methods, such as robot path planning, supply chain management optimisation, and energy system management.
![image](https://github.com/JLU-WangXu/Octopus_Inspired_Optimization_OIO/assets/73222849/f97c771c-2a65-4080-b90b-ea3439c66f8c)
![image](https://github.com/JLU-WangXu/Octopus_Inspired_Optimization_OIO/assets/73222849/caace2fd-8cb6-4108-9891-042713a4d917)
![image](https://github.com/JLU-WangXu/Octopus_Inspired_Optimization_OIO/assets/73222849/86734ad8-7d74-4b5c-90d0-7e24d7e0d2af)
![image](https://github.com/JLU-WangXu/Octopus_Inspired_Optimization_OIO/assets/73222849/51ab6200-3d81-44f2-9d45-449dea9817c8)
![image](https://github.com/JLU-WangXu/Octopus_Inspired_Optimization_OIO/assets/73222849/588ee333-3048-490b-8eea-4947db07ccc0)
![image](https://github.com/JLU-WangXu/Octopus_Inspired_Optimization_OIO/assets/73222849/059530fc-1dc8-4f1a-aede-3efe8aeaf487)








# 标准版模型
1.*吸盘*：pso算法,可以更换为1.任意群优化算法2.梯度优化算法**吸盘和触手的交互满足协同进化模型(Cooperative Co-evolution Model)**  
2.*触手*：群智能+个体强化学习实现的一部分**触手和个体的交互满足主从协同模型 (Master-Slave Cooperative Model)**  
3.*个体*：强化学习，用以决定更新的方式和状态**强化学习的模型用的SARSA**  

