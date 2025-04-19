from llm import OpenRouterLLM
from model import CodeSegmentHDR
from problem import Problem, AVAIABLE_TERMINALS
import random
from se_evo import LLMInitOperator, LLMCrossoverOperator, LLMMutationOperator, \
    CoEvoOperator, SelfEvoOperator, CollectiveRefOperator, RandomSelectOperator, \
        TopKElitismReplaceOperator, makespan_fitness_func, se_evo

import prompt_template as pt
        
# Create problem 
random.seed(42)

problem = Problem(AVAIABLE_TERMINALS, pool_size=6)
problem.random_generate(num_jobs=15, max_oprs_each_job=6, num_machines=6, max_arr_time=200)

for job in problem.jobs:
    print(str(job))

# Build llm
llm_model = OpenRouterLLM('openrouter', 'optimus-alpha', free=False, timeout=(30, 400))

# Create Operator
llm_init_func = LLMInitOperator(problem, llm_model, prompt_template=pt.INIT_IND_PROMPT_TEMPLATE)
llm_crossover_func = LLMCrossoverOperator(problem, llm_model, prompt_template=pt.CROSSOVER_PROMPT_TEMPLATE)
llm_mutation_func = LLMMutationOperator(problem, llm_model, prompt_template=pt.MUTATION_PROMPT_TEMPLATE)
co_evo_func = CoEvoOperator(problem, llm_model, prompt_template=pt.CO_EVO_PROMPT_TEMPLATE)
self_evo_func = SelfEvoOperator(problem, llm_model, prompt_template=pt.SELF_EVO_PROMPT_TEMPLATE)
collective_evo_func = CollectiveRefOperator(problem, llm_model, prompt_template=pt.COLLECTIVE_REF_PROMPT_TEMPLATE)
selector = RandomSelectOperator(problem)
replace_opr = TopKElitismReplaceOperator(problem, k=2)

# Main Se-Evo process
best = se_evo(120, problem, llm_init_func,
              co_evo_func, self_evo_func, collective_evo_func,
              llm_crossover_func, llm_mutation_func,
              selector, replace_opr, makespan_fitness_func,
              init_size=20, subset_size=8,
              template_file_path='template.txt',
              pc=0.9, pm=0.3)

if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
