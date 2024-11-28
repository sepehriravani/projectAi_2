import random
import matplotlib.pyplot as plt

# محاسبه مقدار Fitness به صورت دستی
def calculate_fitness_manual(portfolio, returns, covariance):
    portfolio_return = sum(portfolio[i] * returns[i] for i in range(len(returns)))
    portfolio_variance = 0
    for i in range(len(returns)):
        for j in range(len(returns)):
            portfolio_variance += portfolio[i] * portfolio[j] * covariance[i][j]
    portfolio_risk = portfolio_variance ** 0.5
    return portfolio_return / portfolio_risk if portfolio_risk != 0 else 0

# پیاده‌سازی الگوریتم Beam Search
def beam_search_manual(returns, covariance, beam_width=5, iterations=50):
    num_assets = len(returns)
    current_solutions = [[random.random() for _ in range(num_assets)] for _ in range(beam_width)]
    for solution in current_solutions:
        total = sum(solution)
        for i in range(len(solution)):
            solution[i] /= total  # نرمال‌سازی پرتفولیوها

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
                    modified_solution[j] /= total  # نرمال‌سازی
                new_solutions.append(modified_solution)

        fitness_values = [calculate_fitness_manual(solution, returns, covariance) for solution in new_solutions]
        sorted_solutions = sorted(
            zip(new_solutions, fitness_values), key=lambda x: x[1], reverse=True
        )
        current_solutions = [sol for sol, _ in sorted_solutions[:beam_width]]
        fitness_progression.append(sorted_solutions[0][1])

        if sorted_solutions[0][1] > best_fitness:
            best_fitness = sorted_solutions[0][1]
            best_solution = sorted_solutions[0][0]

    return best_solution, best_fitness, fitness_progression

# رسم نمودار پیشرفت Fitness
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

# داده‌های نمونه
returns_example = [0.1, 0.2, 0.15, 0.3]
covariance_example = [
    [0.005, 0.002, 0.001, 0.002],
    [0.002, 0.004, 0.002, 0.001],
    [0.001, 0.002, 0.003, 0.002],
    [0.002, 0.001, 0.002, 0.006],
]

# اجرای الگوریتم Beam Search
best_solution, best_fitness, fitness_progression = beam_search_manual(returns_example, covariance_example)

# نمایش نتایج
print("Best Portfolio:", best_solution)
print("Best Fitness (Return/Risk):", best_fitness)

# رسم نمودار پیشرفت Fitness
plot_fitness_progression(fitness_progression, title="Fitness Progression (Beam Search)")
