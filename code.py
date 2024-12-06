import random
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from a file
def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    returns = list(map(float, lines[0].split()))  # خط اول: بازده‌ها
    covariance = [list(map(float, line.split())) for line in lines[1:]]  # خطوط بعدی: ماتریس کواریانس
    return returns, covariance

# Function to calculate fitness
def calculate_fitness(portfolio, returns, covariance):
    portfolio = np.array(portfolio)
    returns = np.array(returns)
    covariance = np.array(covariance)
    
    # Calculate portfolio return
    portfolio_return = np.dot(portfolio, returns)  # w^T * r
    
    # Calculate portfolio risk
    portfolio_variance = np.dot(portfolio.T, np.dot(covariance, portfolio))  # w^T * C * w
    portfolio_risk = np.sqrt(portfolio_variance)
    
    # Return fitness (Return/Risk)
    return portfolio_return / portfolio_risk if portfolio_risk != 0 else 0

# Beam Search Algorithm
def beam_search(returns, covariance, beam_width=5, iterations=50):
    num_assets = len(returns)
    current_solutions = [[random.random() for _ in range(num_assets)] for _ in range(beam_width)]
    for solution in current_solutions:
        total = sum(solution)
        for i in range(len(solution)):
            solution[i] /= total  # Normalize

    best_solution = None
    best_fitness = 0
    fitness_progression = []

    for _ in range(iterations):
        new_solutions = []
        for solution in current_solutions:
            for i in range(num_assets):
                modified_solution = solution[:]
                modified_solution[i] += random.uniform(-0.1, 0.1)
                modified_solution = [max(0, x) for x in modified_solution]
                total = sum(modified_solution)
                for j in range(len(modified_solution)):
                    modified_solution[j] /= total  # Normalize
                new_solutions.append(modified_solution)

        fitness_values = [calculate_fitness(solution, returns, covariance) for solution in new_solutions]
        sorted_solutions = sorted(
            zip(new_solutions, fitness_values), key=lambda x: x[1], reverse=True
        )
        current_solutions = [sol for sol, _ in sorted_solutions[:beam_width]]
        fitness_progression.append(sorted_solutions[0][1])

        if sorted_solutions[0][1] > best_fitness:
            best_fitness = sorted_solutions[0][1]
            best_solution = sorted_solutions[0][0]

    return best_solution, best_fitness, fitness_progression

# Simulated Annealing Algorithm
def simulated_annealing(returns, covariance, initial_temp=1000, cooling_rate=0.95, iterations=1000):
    num_assets = len(returns)
    current_solution = [random.random() for _ in range(num_assets)]
    total = sum(current_solution)
    for i in range(len(current_solution)):
        current_solution[i] /= total  # Normalize
    current_fitness = calculate_fitness(current_solution, returns, covariance)

    best_solution = current_solution[:]
    best_fitness = current_fitness
    fitness_progression = []

    temperature = initial_temp

    for _ in range(iterations):
        new_solution = current_solution[:]
        for i in range(num_assets):
            new_solution[i] += random.uniform(-0.1, 0.1)
        new_solution = [max(0, x) for x in new_solution]
        total = sum(new_solution)
        for i in range(len(new_solution)):
            new_solution[i] /= total  # Normalize

        new_fitness = calculate_fitness(new_solution, returns, covariance)
        fitness_progression.append(current_fitness)

        if new_fitness > current_fitness or random.random() < (2.718 ** ((new_fitness - current_fitness) / temperature)):
            current_solution = new_solution[:]
            current_fitness = new_fitness

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution[:]

        temperature *= cooling_rate

    return best_solution, best_fitness, fitness_progression

# Random Local Beam Search Algorithm
def random_local_beam_search(returns, covariance, beam_width=5, iterations=50):
    num_assets = len(returns)
    current_solutions = [[random.random() for _ in range(num_assets)] for _ in range(beam_width)]
    for solution in current_solutions:
        total = sum(solution)
        for i in range(len(solution)):
            solution[i] /= total  # Normalize

    best_solution = None
    best_fitness = 0
    fitness_progression = []

    for _ in range(iterations):
        new_solutions = []
        for solution in current_solutions:
            modified_solution = solution[:]
            random_index = random.randint(0, num_assets - 1)
            modified_solution[random_index] += random.uniform(-0.1, 0.1)
            modified_solution = [max(0, x) for x in modified_solution]
            total = sum(modified_solution)
            for i in range(len(modified_solution)):
                modified_solution[i] /= total  # Normalize
            new_solutions.append(modified_solution)

        fitness_values = [calculate_fitness(solution, returns, covariance) for solution in new_solutions]
        sorted_solutions = sorted(
            zip(new_solutions, fitness_values), key=lambda x: x[1], reverse=True
        )
        current_solutions = [sol for sol, _ in sorted_solutions[:beam_width]]
        fitness_progression.append(sorted_solutions[0][1])

        if sorted_solutions[0][1] > best_fitness:
            best_fitness = sorted_solutions[0][1]
            best_solution = sorted_solutions[0][0]

    return best_solution, best_fitness, fitness_progression

# Genetic Algorithm
def genetic_algorithm(returns, covariance, population_size=20, generations=50, mutation_rate=0.1):
    num_assets = len(returns)
    population = [[random.random() for _ in range(num_assets)] for _ in range(population_size)]
    for solution in population:
        total = sum(solution)
        for i in range(len(solution)):
            solution[i] /= total  # Normalize

    best_solution = None
    best_fitness = 0
    fitness_progression = []

    for _ in range(generations):
        fitness_values = [calculate_fitness(solution, returns, covariance) for solution in population]
        sorted_population = sorted(
            zip(population, fitness_values), key=lambda x: x[1], reverse=True
        )
        population = [sol for sol, _ in sorted_population[:population_size // 2]]
        fitness_progression.append(sorted_population[0][1])

        if sorted_population[0][1] > best_fitness:
            best_fitness = sorted_population[0][1]
            best_solution = sorted_population[0][0]

        # Crossover
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            crossover_point = random.randint(1, num_assets - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            new_population.append(child)

        # Mutation
        for solution in new_population:
            if random.random() < mutation_rate:
                random_index = random.randint(0, num_assets - 1)
                solution[random_index] += random.uniform(-0.1, 0.1)
                solution = [max(0, x) for x in solution]
                total = sum(solution)
                for i in range(len(solution)):
                    solution[i] /= total  # Normalize

        population = new_population

    return best_solution, best_fitness, fitness_progression

# Plot fitness progression
def plot_fitness_progression(data, title="Fitness Progression", x_label="Iteration", y_label="Fitness"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data) + 1), data, marker='o', linestyle='-', color='blue', label="Fitness")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main execution
file_path = 'sample_input.txt'  # Replace with your input file path
returns, covariance = read_input(file_path)

random.seed(42)  # Set random state

# Execute Beam Search
best_solution_beam, best_fitness_beam, fitness_progression_beam = beam_search(returns, covariance)

# Execute Simulated Annealing
best_solution_sa, best_fitness_sa, fitness_progression_sa = simulated_annealing(returns, covariance)

# Execute Random Local Beam Search
best_solution_rlbs, best_fitness_rlbs, fitness_progression_rlbs = random_local_beam_search(returns, covariance)

# Execute Genetic Algorithm
best_solution_ga, best_fitness_ga, fitness_progression_ga = genetic_algorithm(returns, covariance)

# Display best solutions and fitness
print("Beam Search Best Solution:", best_solution_beam)
print("Beam Search Best Fitness (Return/Risk):", best_fitness_beam)

print("Simulated Annealing Best Solution:", best_solution_sa)
print("Simulated Annealing Best Fitness (Return/Risk):", best_fitness_sa)

print("Random Local Beam Search Best Solution:", best_solution_rlbs)
print("Random Local Beam Search Best Fitness (Return/Risk):", best_fitness_rlbs)

print("Genetic Algorithm Best Solution:", best_solution_ga)
print("Genetic Algorithm Best Fitness (Return/Risk):", best_fitness_ga)

# Plot fitness progressions
plot_fitness_progression(fitness_progression_beam, title="Beam Search Fitness Progression")
plot_fitness_progression(fitness_progression_sa, title="Simulated Annealing Fitness Progression")
plot_fitness_progression(fitness_progression_rlbs, title="Random Local Beam Search Fitness Progression")
plot_fitness_progression(fitness_progression_ga, title="Genetic Algorithm Fitness Progression")
