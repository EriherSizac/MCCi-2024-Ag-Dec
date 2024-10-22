import sys
from Experiment import Experiment

if len(sys.argv) < 4:
    print("Syntax error! Main.py params_file function_name number_executions [output_file] ")
    exit(0)

params_file = sys.argv[1]
function_name = sys.argv[2]
number_executions = int(sys.argv[3])
experiment = Experiment(params_file, function_name, number_executions)
if(len(sys.argv) == 5):
    experiment.execute_experiment(sys.argv[4])
else: 
    experiment.execute_experiment()




