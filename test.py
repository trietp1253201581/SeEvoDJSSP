import random
from model import CodeSegmentHDR
from problem import Problem, AVAIABLE_TERMINALS
from simulate import Simulator
from se_evo import get_template, LLMInitOperator, LLMCrossoverOperator, LLMMutationOperator, CoEvoOperator, SelfEvoOperator, CollectiveRefOperator
from llm import OpenRouterLLM
import prompt_template as pt
from basic_evo import Individual, Population
from typing import List

random.seed(42)

problem = Problem(AVAIABLE_TERMINALS)
problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)

llm_model = OpenRouterLLM('openrouter', 'quasar-alpha', free=False)

def test_simulator():
    print("----- TEST SIMULATOR -----")
    hdr = CodeSegmentHDR()
    hdr.load('tmp/hdr_1.py')
    print(hdr.code)
    global problem
    
    for job in problem.jobs:
        print(str(job))
        
    simulator = Simulator(hdr=hdr, problem=problem, pool_size=2)
    simulator.simulate(debug=True)
    
def test_llm_init():
    print("----- TEST LLM INIT OPR -----")
    global problem, llm_model
    init_size = 5
    func_template = get_template(template_file='template.txt')
    
    llm_init_opr = LLMInitOperator(problem, llm_model, pt.INIT_IND_PROMPT_TEMPLATE)
    
    pop = llm_init_opr(init_size, func_template)
    
    for ind in pop.inds:
        print(ind.chromosome.code)
    
def test_co_evo():
    print("----- TEST CO EVO OPR -----")
    global problem, llm_model
    
    inds: List[Individual] = []
    for i in range(1, 3):
        file_path = f'tmp/hdr_{i}.py'
        new_hdr = CodeSegmentHDR()
        new_hdr.load(file_path)
        ind = Individual(problem)
        ind.chromosome = new_hdr
        ind.fitness = -60 - i
        inds.append(ind)
    
    co_evo_opr = CoEvoOperator(problem, llm_model, pt.CO_EVO_PROMPT_TEMPLATE)
    
    reflection, reflected_hdrs = co_evo_opr(inds, inds[0], inds[1])
    
    print(reflection)
    
def test_crossover():
    print("----- TEST LLM CROSSOVER OPR -----")
    global problem, llm_model

    ind1 = Individual(problem)
    hdr1 = CodeSegmentHDR()
    hdr1.load(f'tmp/hdr_1.py')
    ind1.chromosome = hdr1
    
    ind2 = Individual(problem)
    hdr2 = CodeSegmentHDR()
    hdr2.load(f'tmp/hdr_2.py')
    ind2.chromosome = hdr2
    
    crossover_opr = LLMCrossoverOperator(problem, llm_model, pt.CROSSOVER_PROMPT_TEMPLATE)
    
    off1, off2 = crossover_opr(ind1, ind2)
    print(off1.chromosome.code)
    print(off2.chromosome.code)

def test_self_evo():
    print("----- TEST SELF EVO OPR -----")
    global problem, llm_model
    
    inds_before: List[Individual] = []
    for i in range(1, 2):
        file_path = f'tmp/hdr_1.py'
        new_hdr = CodeSegmentHDR()
        new_hdr.load(file_path)
        ind = Individual(problem)
        ind.chromosome = new_hdr
        ind.fitness = -random.randint(60, 70)
        inds_before.append(ind)
        
    inds_after: List[Individual] = []
    for i in range(1, 2):
        file_path = f'tmp/hdr_2.py'
        new_hdr = CodeSegmentHDR()
        new_hdr.load(file_path)
        ind = Individual(problem)
        ind.chromosome = new_hdr
        ind.fitness = -random.randint(60, 70)
        inds_after.append(ind)
        
    co_evo_reflection = """
To improve HDR effectiveness in dynamic job shop scheduling, 
design scoring functions that integrate both job and machine urgency, 
explicitly linking deadlines and remaining operation times to current system time. 
Incorporate time-dependent features such as how close a job is to its operation deadline,
the overall job deadline, and how long it has already waited, to dynamically adjust ranking priorities. 
Use slack time to penalize jobs that are at risk of missing deadlines, 
and adapt weighting of features based on job arrival and processing dynamics. 
Building composite scores that dynamically balance current and future urgency helps anticipate bottlenecks and prioritizes jobs critical to reducing makespan.
"""
    
    self_evo_opr = SelfEvoOperator(problem, llm_model, pt.SELF_EVO_PROMPT_TEMPLATE)
    
    reflections, reflected_hdrs = self_evo_opr(inds_before, inds_after, co_evo_reflection)
    
    print(reflections)

def test_collective_ref():
    print("----- TEST COLLECTIVE OPR -----")
    global problem, llm_model
    
    reflections = []
    reflections.append("In this improved HDR, we explicitly incorporated a dynamic urgency score based on the proximity of a job's next operation deadline and how long the job has waited since arrival. This mechanism better captures time-sensitive priorities. By adding this urgency to the original composite score, the scheduler became more responsive to imminent deadlines, leading to a reduction in makespan. This highlights the effectiveness of combining static job features with dynamic, time-dependent deadline urgency.")
    reflections.append('This revised HDR combines an urgency metric sensitive to operation deadlines with the base job score, adapting further via conditional adjustments based on slack time. For tight slack (js<5), the function boosts priority, encouraging timely processing; otherwise, it penalizes relaxed slack conditions. This nuanced integration of urgency and slack proved superior, dynamically emphasizing critical jobs and reducing overall makespan compared to a simpler, less adaptive function.')
    reflections.append("""To improve HDR effectiveness in dynamic job shop scheduling,
design scoring functions that integrate both job and machine urgency,
explicitly linking deadlines and remaining operation times to current system time.
Incorporate time-dependent features such as how close a job is to its operation deadline,
the overall job deadline, and how long it has already waited, to dynamically adjust ranking priorities.
Use slack time to penalize jobs that are at risk of missing deadlines,
and adapt weighting of features based on job arrival and processing dynamics.
Building composite scores that dynamically balance current and future urgency helps anticipate bottlenecks and prioritizes jobs critical to reducing makespan.""")
    
    collective_opr = CollectiveRefOperator(problem, llm_model, pt.COLLECTIVE_REF_PROMPT_TEMPLATE)
    
    reflection = collective_opr(reflections)
    print(reflection)
    
def test_mutation():
    print("----- TEST MUTATION OPR -----")
    global problem, llm_model
    
    ind = Individual(problem)
    hdr = CodeSegmentHDR()
    hdr.load('tmp/hdr_1.py')
    ind.chromosome = hdr
    print(hdr)
    
    reflection = "To enhance HDR effectiveness in dynamic job shop scheduling, develop adaptive scoring functions that integrate both static job features and dynamic urgency metrics linked closely to deadlines and elapsed wait times. Explicitly incorporate how near a job is to its operation deadline and overall job deadline, dynamically boosting priority for imminent deadlines and penalizing slack conditions when appropriate. Combine these urgency signals with features reflecting remaining operations, processing times, and machine states to construct composite scores responsive to current system status. By dynamically balancing these time-dependent priorities, the HDR becomes more sensitive to critical jobs at risk of missing deadlines, thereby reducing makespan and improving overall scheduling responsiveness."
    
    mut_opr = LLMMutationOperator(problem, llm_model, pt.MUTATION_PROMPT_TEMPLATE)
    
    new_ind = mut_opr(ind, reflection)
    
    print(new_ind.chromosome.code)
    
def close_connect():
    global llm_model
    llm_model.close()