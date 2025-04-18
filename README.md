Real-Coded Genetic Algorithm (RCGA) to solve all 12 GAP datasets (gap1.txt to gap12.txt).
The algorithm simulates task assignment to servers while minimizing cost and ensuring capacity constraints.
Each dataset may contain multiple instances, and the algorithm processes each by reading user-server configurations, cost,
and resource matrices, and computing an optimal task-server assignment using RCGA.
The RCGA core includes initial random population generation, selection via tournament method, simulated binary crossover (SBX), Gaussian mutation, and elitism-based survivor selection. 
Fitness is calculated as the benefit (cost savings)while applying penalties for overloading servers or incorrect task assignments. 
To ensure feasibility, every solution is normalized to represent valid probability-based assignments.
