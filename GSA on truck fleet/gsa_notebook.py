import numpy as np
import random

# Problem parameters
J, L, K = 10, 10, 3
Q = [13.5, 15, 18]
e = [1339, 1330, 1300]
f = [1628, 1480, 1563]
THC = [2000, 2500, 2800]
DHC = 10000
T_max = 8
service_time = 0.25
travel_time_per_unit = 0.1

# Generate demands
np.random.seed(42)
demand_med = np.random.randint(5, 15, size=J)
demand_food = np.random.randint(5, 15, size=L)

def calculate_cost(solution):
    """Calculate cost with moderate penalties to allow variation"""
    total_cost = 0
    penalty = 0
    
    # Track each truck-trip combination
    truck_trips = np.zeros((K, 10, 3))  # (load, time, count)
    
    # Process medical shops
    for j, (k, t) in enumerate(solution[:J]):
        u = demand_med[j]
        truck_trips[k,t,0] += u  # load
        truck_trips[k,t,1] += service_time + travel_time_per_unit * u  # time
        truck_trips[k,t,2] += 1  # count
        total_cost += e[k] * u
    
    # Process food shops
    for l, (k, t) in enumerate(solution[J:]):
        u = demand_food[l]
        truck_trips[k,t,0] += u
        truck_trips[k,t,1] += service_time + travel_time_per_unit * u
        truck_trips[k,t,2] += 1
        total_cost += f[k] * u
    
    # Add fixed costs and penalties
    for k in range(K):
        for t in range(10):
            if truck_trips[k,t,2] > 0:  # if truck-trip used
                total_cost += THC[k] + DHC
                # Moderate penalties (1e4 instead of 1e6)
                if truck_trips[k,t,0] > Q[k]:
                    penalty += 1e4 * (truck_trips[k,t,0] - Q[k])
                if truck_trips[k,t,1] > T_max:
                    penalty += 1e4 * (truck_trips[k,t,1] - T_max)
    
    return total_cost + penalty

def initialize_population(n_agents):
    """Initialize random solutions"""
    return [[(random.randint(0,K-1), random.randint(0,9)) for _ in range(J+L)] 
            for _ in range(n_agents)]

def gsa_optimize(n_agents=30, max_iter=100):
    """Improved GSA implementation with cost variation"""
    positions = initialize_population(n_agents)
    velocities = np.zeros((n_agents, J+L, 2))  # Separate velocities for k and t
    
    best_solution = None
    best_cost = float('inf')
    cost_history = []
    
    for iteration in range(max_iter):
        # Evaluate current population
        costs = np.array([calculate_cost(pos) for pos in positions])
        
        # Update best solution
        current_best_idx = np.argmin(costs)
        if costs[current_best_idx] < best_cost:
            best_cost = costs[current_best_idx]
            best_solution = positions[current_best_idx].copy()
        cost_history.append(best_cost)
        
        # Calculate masses (normalized)
        worst = np.max(costs)
        best = np.min(costs)
        if worst == best:
            masses = np.ones(n_agents)
        else:
            masses = (worst - costs) / (worst - best)  # for minimization
        masses = masses / np.sum(masses)
        
        # Update gravitational constant
        G = 100 * np.exp(-20 * iteration / max_iter)
        
        # Calculate forces
        forces = np.zeros((n_agents, J+L, 2))
        for i in range(n_agents):
            for d in range(J+L):
                for j in range(n_agents):
                    if i != j:
                        # Calculate distance between solutions
                        diff_k = positions[j][d][0] - positions[i][d][0]
                        diff_t = positions[j][d][1] - positions[i][d][1]
                        distance = np.sqrt(diff_k**2 + diff_t**2)
                        distance = max(distance, 1e-5)
                        
                        # Calculate force components
                        force_magnitude = G * masses[i] * masses[j] / distance
                        forces[i,d,0] += force_magnitude * diff_k * random.random()
                        forces[i,d,1] += force_magnitude * diff_t * random.random()
        
        # Update velocities and positions
        for i in range(n_agents):
            for d in range(J+L):
                # Update velocities
                velocities[i,d,0] = velocities[i,d,0] * random.random() + forces[i,d,0]
                velocities[i,d,1] = velocities[i,d,1] * random.random() + forces[i,d,1]
                
                # Update positions with bounds checking
                new_k = int(positions[i][d][0] + velocities[i,d,0])
                new_t = int(positions[i][d][1] + velocities[i,d,1])
                positions[i][d] = (
                    max(0, min(K-1, new_k)),
                    max(0, min(9, new_t))
                )
        
        print(f"Iteration {iteration+1}: Best Cost = {best_cost:.2f}")
    
    return best_solution, best_cost, cost_history

# Run the optimized GSA
best_sol, best_c, history = gsa_optimize(n_agents=30, max_iter=50)

# Print final results
print(f"\nFinal Best Cost: {best_c:.2f}")
print("Solution Analysis:")

# Analyze the solution
truck_usage = {}
for k in range(K):
    for t in range(10):
        truck_usage[(k,t)] = {'med': [], 'food': [], 'load': 0, 'time': 0}

for j, (k,t) in enumerate(best_sol[:J]):
    u = demand_med[j]
    truck_usage[(k,t)]['med'].append(j)
    truck_usage[(k,t)]['load'] += u
    truck_usage[(k,t)]['time'] += service_time + travel_time_per_unit * u

for l, (k,t) in enumerate(best_sol[J:]):
    u = demand_food[l]
    truck_usage[(k,t)]['food'].append(l)
    truck_usage[(k,t)]['load'] += u
    truck_usage[(k,t)]['time'] += service_time + travel_time_per_unit * u

for (k,t), data in truck_usage.items():
    if data['med'] or data['food']:
        print(f"Truck {k}, Trip {t}:")
        print(f"  Medical shops: {data['med']}")
        print(f"  Food shops: {data['food']}")
        print(f"  Total load: {data['load']:.1f}/{Q[k]}", 
              " Over" if data['load'] > Q[k] else "")
        print(f"  Total time: {data['time']:.2f}/{T_max} hours",
              " Over" if data['time'] > T_max else "")
