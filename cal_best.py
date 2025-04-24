import random
from problem import Problem, AVAIABLE_TERMINALS
from model import CodeSegmentHDR
from evaluate import SimulationBaseEvaluator, Simulator
# Create problem 
random.seed(42)

problem = Problem(AVAIABLE_TERMINALS, pool_size=16)
problem.custom_generate(num_jobs=300, max_oprs_each_job=10, num_machines=20, max_arr_time=150, arrival_type='uniform', proc_dist='uniform', deadline_factor=1.2)

best_hdr = CodeSegmentHDR()
best_hdr.load('tmp/best_surrogate.py')

simulator = Simulator(problem)

makespan = simulator.simulate(best_hdr, True, 0.001)

print(f"Makespan: {makespan}")
