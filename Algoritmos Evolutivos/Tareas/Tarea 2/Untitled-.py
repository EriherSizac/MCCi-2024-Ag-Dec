# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# # Constants

# %%
# Global constants for problems
ROSENBROCK_BOUNDS = [-2.048, 2.048]
RASTRIGIN_BOUNDS = [-5.12, 5.12]
PC_BINARY = 0.9  # Crossover probability for binary encoding
PC_REAL = 0.9    # Crossover probability for real encoding
PM_BINARY = 1 / 16  # Mutation probability for binary encoding
PM_REAL = 1 / 16    # Mutation probability for real encoding
N_C = 20  # Distribution index for SBX crossover
N_P = 20  # Mutation parameter

# %% [markdown]
# # Target Functions

# %%
def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def rastrigin(x, A=10):
    return A * len(x) + sum(xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x)

# %% [markdown]
# # Encoding Functions

# %%
def real_to_binary(real_values, bounds, n_bits):
    """Convert a vector of real values within bounds to binary representations."""
    binaries = []
    for real_value, bound in zip(real_values, bounds):
        scaled_value = (real_value - bound[0]) / (bound[1] - bound[0])  # Scale to [0, 1]
        max_int = 2 ** n_bits - 1
        int_value = int(scaled_value * max_int)
        binary_str = np.binary_repr(int_value, width=n_bits)  # Get binary string with fixed width
        binaries.extend([int(bit) for bit in binary_str])  # Extend the list with binary digits
    return np.array(binaries)

def binary_to_real(binary_chrom, bounds, n_bits):
    """Convert binary chromosome to vector of real values within the bounds."""
    real_values = []
    bits_per_var = n_bits
    for i in range(len(bounds)):
        start = i * bits_per_var
        end = start + bits_per_var
        bits = binary_chrom[start:end]
        decimal_value = int("".join(map(str, bits)), 2)
        max_decimal = 2 ** bits_per_var - 1
        real_value = bounds[i][0] + (bounds[i][1] - bounds[i][0]) * (decimal_value / max_decimal)
        real_values.append(real_value)
    return real_values

# %% [markdown]
# # Crossover Functions

# %%
def single_point_crossover(p1, p2):
    point = np.random.randint(1, len(p1))
    child1 = np.concatenate((p1[:point], p2[point:]))
    child2 = np.concatenate((p2[:point], p1[point:]))
    return child1, child2

def sbx_crossover(p1, p2, nc):
    child1, child2 = np.zeros_like(p1), np.zeros_like(p2)
    for i in range(len(p1)):
        u = np.random.rand()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (nc + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (nc + 1))
        child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return child1, child2

# %% [markdown]
# # Mutation Functions

# %%
def binary_mutation(chrom, Pm):
    for i in range(len(chrom)):
        if np.random.rand() < Pm:
            chrom[i] = 1 - chrom[i]  # Flip the bit
    return chrom

def pm_mutation(chrom, bounds, gen):
    for i in range(len(chrom)):
        if np.random.rand() < PM_REAL:
            delta = np.random.rand()
            if delta < 0.5:
                delta_q = (2 * delta) ** (1 / (N_P + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - delta)) ** (1 / (N_P + 1))
            chrom[i] += delta_q * (bounds[i][1] - bounds[i][0])
            chrom[i] = np.clip(chrom[i], bounds[i][0], bounds[i][1])
    return chrom

# %% [markdown]
# # Selection Functions

# %%
def roulette_wheel_selection(population, fitness):
    max_fitness = max(fitness)
    min_fitness = min(fitness)
    transformed_fitness = [max_fitness - f + 1e-6 for f in fitness]
    total_fitness = sum(transformed_fitness)
    probabilities = [f / total_fitness for f in transformed_fitness]
    cumulative_probabilities = np.cumsum(probabilities)
    r = np.random.rand()
    for i, cum_prob in enumerate(cumulative_probabilities):
        if r <= cum_prob:
            return population[i]
    return population[-1]

def binary_tournament_selection(population, fitness):
    idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
    if fitness[idx1] < fitness[idx2]:
        return population[idx1]
    else:
        return population[idx2]

# %% [markdown]
# # Genetic Algorithm Function

# %%
def genetic_algorithm(fitness_function, bounds, encoding_type="binary", n_generations=100, pop_size=100, n_bits=16, stop_threshold=1e-6, stagnation_generations=5, verbose=False, dimension=2):
    # Adjust bounds for each dimension
    if isinstance(bounds[0], list) or isinstance(bounds[0], tuple):
        bounds_per_dim = bounds
    else:
        bounds_per_dim = [bounds] * dimension

    # Create initial population
    if encoding_type == "binary":
        population = []
        for _ in range(pop_size):
            real_values = [np.random.uniform(b[0], b[1]) for b in bounds_per_dim]
            chrom = real_to_binary(real_values, bounds_per_dim, n_bits)
            population.append(chrom)
    else:
        population = [np.array([np.random.uniform(b[0], b[1]) for b in bounds_per_dim]) for _ in range(pop_size)]

    best_fitness_per_gen = []
    stagnation_count = 0

    for gen in range(n_generations):
        if encoding_type == "binary":
            real_population = [binary_to_real(ind, bounds_per_dim, n_bits) for ind in population]
            fitness = [fitness_function(ind) for ind in real_population]
        else:
            fitness = [fitness_function(ind) for ind in population]

        new_population = []
        for _ in range(pop_size // 2):
            if encoding_type == "binary":
                p1 = roulette_wheel_selection(population, fitness)
                p2 = roulette_wheel_selection(population, fitness)
                if np.random.rand() < PC_BINARY:
                    c1, c2 = single_point_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = binary_mutation(c1, PM_BINARY)
                c2 = binary_mutation(c2, PM_BINARY)
                new_population.extend([c1, c2])
            else:
                p1 = binary_tournament_selection(population, fitness)
                p2 = binary_tournament_selection(population, fitness)
                if np.random.rand() < PC_REAL:
                    c1, c2 = sbx_crossover(p1, p2, N_C)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = pm_mutation(c1, bounds_per_dim, gen)
                c2 = pm_mutation(c2, bounds_per_dim, gen)
                new_population.extend([c1, c2])
        population = new_population

        # Track best fitness
        best_fitness = min(fitness)
        best_fitness_per_gen.append(best_fitness)

        if verbose:
            print(f"Generation {gen + 1}, Best Fitness: {best_fitness}")

        # Check for stagnation (early stopping)
        if gen >= stagnation_generations:
            recent_fitness = best_fitness_per_gen[-stagnation_generations:]
            fitness_change = max(recent_fitness) - min(recent_fitness)

            if fitness_change < stop_threshold:
                if verbose:
                    print(f"Early stopping at generation {gen + 1} due to insufficient improvement.")
                break

    return best_fitness_per_gen

# %% [markdown]
# # Running Experiments

# %%
# Define the problems and parameters
problems = {
    'Test Problem 1': rosenbrock,
    'Rastrigin n=2': rastrigin,
    'Rastrigin n=5': rastrigin
}

bounds = {
    'Test Problem 1': ROSENBROCK_BOUNDS,
    'Rastrigin n=2': RASTRIGIN_BOUNDS,
    'Rastrigin n=5': RASTRIGIN_BOUNDS
}

dimensions = {
    'Test Problem 1': 2,
    'Rastrigin n=2': 2,
    'Rastrigin n=5': 5
}

encoding_types = ['binary', 'real']

n_experiments = 20

results = {}

for problem_name in problems:
    fitness_function = problems[problem_name]
    bound = bounds[problem_name]
    dim = dimensions[problem_name]
    # Adjust bounds per dimension
    bounds_per_dim = [bound] * dim
    results[problem_name] = {}
    for encoding in encoding_types:
        all_best_fitness = []
        all_convergence = []
        for exp in range(n_experiments):
            print(f"Running {problem_name}, Encoding: {encoding}, Experiment: {exp + 1}")
            best_fitness_per_gen = genetic_algorithm(
                fitness_function=fitness_function,
                bounds=bound,
                encoding_type=encoding,
                n_generations=100,
                pop_size=100,
                n_bits=16,
                stop_threshold=1e-6,
                stagnation_generations=5,
                verbose=False,
                dimension=dim
            )
            all_best_fitness.append(best_fitness_per_gen[-1])  # Best fitness of last generation
            all_convergence.append(best_fitness_per_gen)       # Convergence over generations
        # Store results
        results[problem_name][encoding] = {
            'best_fitness': all_best_fitness,
            'convergence': all_convergence
        }

# %% [markdown]
# # Building the Table

# %%
# Create a DataFrame to store results
table_data = {}
for encoding in encoding_types:
    for problem_name in problems:
        key = f"{problem_name} ({encoding} encoding)"
        best_fitness_list = results[problem_name][encoding]['best_fitness']
        table_data[key] = best_fitness_list

df = pd.DataFrame(table_data)
df.index = [f"Experiment {i + 1}" for i in range(n_experiments)]

# Calculate statistics
stats = df.describe().loc[['mean', 'std', 'min', 'max']]
stats.index = ['Mean', 'Standard Deviation', 'Min', 'Max']
df = df.append(stats)

# Display the table
print(df)

# %% [markdown]
# # Generating Convergence Plots

# %%
selected_experiment = 4  # Choosing the 5th experiment (index starts from 0)

for problem_name in problems:
    plt.figure()
    for encoding in encoding_types:
        convergence = results[problem_name][encoding]['convergence'][selected_experiment]
        plt.plot(convergence, label=f"{encoding} encoding")
    plt.title(f"Convergence Plot for {problem_name}")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.show()
