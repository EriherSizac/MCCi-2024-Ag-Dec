import imp
from PSO import PSO
from SOP import FunctionFactory
from Parameters import Parameters
from Report import Report
from Plotting import Plot


class Experiment:
    def __init__(self, params_file, function_name, number_executions):
        self.parameters = Parameters(params_file)     
        self.obj_func = FunctionFactory.select_function(function_name, self.parameters.get_nvar())
        self.pso = PSO(self.parameters, self.obj_func)                        
        self.number_executions = number_executions
        self.report = Report(number_executions, self.parameters.get_Gmax())
        self.plot = Plot()
    
    def execute_experiment(self, output_file=None):
        for execution in range(self.number_executions):
            individual = self.pso.run(execution, self.report)
            self.report.add_final_individual_from_execution(execution, individual)
            print("Execution", execution + 1, " fitness=", individual.get_objective_value())
            self.plot.analysis_plot(self.obj_func, self.report.get_best_individuals_from_generations(execution), execution + 1)
        if output_file == None:
            self.report.print_report()
        else:
            self.report.print_report(output_file)
        pass
        