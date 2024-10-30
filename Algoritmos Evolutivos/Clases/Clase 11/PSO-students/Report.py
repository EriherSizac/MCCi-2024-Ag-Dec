import numpy as np
import copy
from Particle import Particle

class Report:
    def __init__(self, number_executions, Gmax):
        self.number_executions = number_executions
        self.number_generations = Gmax
        self.final_individual_per_execution = np.empty(number_executions, dtype=Particle)
        self.best_individual_per_generation = np.empty((number_executions, Gmax), dtype=Particle)
    
    # Add the best individual obtained during the evolutionary process at a given execution.
    def add_final_individual_from_execution(self, execution, individual):
        self.final_individual_per_execution[execution] = copy.copy(individual)

    # Return the best individual from a given execution
    def get_final_individual_from_execution(self, execution):
        return self.final_individual_per_execution[execution]

    # Add the best individual from a generation to the list corresponding to the current execution
    def add_best_individual_at_generation(self, generation, execution, individual):
        self.best_individual_per_generation[execution][generation] = copy.copy(individual)

    # Return the list of best individuals obtained at each generation at a given execution
    def get_best_individuals_from_generations(self, execution):
        return self.best_individual_per_generation[execution]
    
    def print_report(self, filename="output/optimization.txt"):
        file = open(filename, "wt")
        for individual in self.final_individual_per_execution:
            file.write("{0}\n".format(individual.get_objective_value()))
        file.close()

