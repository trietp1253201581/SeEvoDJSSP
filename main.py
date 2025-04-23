from llm import OpenRouterLLM, GoogleAIStudioLLM
from model import CodeSegmentHDR
from problem import Problem, AVAIABLE_TERMINALS
import random
from se_evo import LLMInitOperator, LLMCrossoverOperator, LLMMutationOperator, \
    CoEvoOperator, SelfEvoOperator, CollectiveRefOperator, RandomSelectOperator, \
        TopKElitismReplaceOperator, se_evo
        
from evaluate import SimulationBaseEvaluator, StaticLLMSurrogateEvaluator
from datetime import datetime
import prompt_template as pt

# Set logging
import logging
logging.basicConfig(
    filename=f'process_{datetime.now().strftime('%Y_%m_%d')}.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(name)s (%(filename)s:%(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
        
# Create problem 
random.seed(42)

problem = Problem(AVAIABLE_TERMINALS, pool_size=6)
problem.random_generate(num_jobs=50, max_oprs_each_job=10, num_machines=15, max_arr_time=300)

#for job in problem.jobs:
   # print(str(job))

# Build llm
#llm_model = OpenRouterLLM('deepseek', 'deepseek-r1-zero', free=True, timeout=(60, 600))
llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600))

# Create Operator
llm_init_func = LLMInitOperator(problem, llm_model, prompt_template=pt.INIT_IND_PROMPT_TEMPLATE)
llm_crossover_func = LLMCrossoverOperator(problem, llm_model, prompt_template=pt.CROSSOVER_PROMPT_TEMPLATE)
llm_mutation_func = LLMMutationOperator(problem, llm_model, prompt_template=pt.MUTATION_PROMPT_TEMPLATE)
co_evo_func = CoEvoOperator(problem, llm_model, prompt_template=pt.CO_EVO_PROMPT_TEMPLATE)
self_evo_func = SelfEvoOperator(problem, llm_model, prompt_template=pt.SELF_EVO_PROMPT_TEMPLATE)
collective_evo_func = CollectiveRefOperator(problem, llm_model, prompt_template=pt.COLLECTIVE_REF_PROMPT_TEMPLATE)
selector = RandomSelectOperator(problem)
replace_opr = TopKElitismReplaceOperator(problem, k=2)
evaluator = StaticLLMSurrogateEvaluator(llm_model, problem, prompt_template=pt.SURROGATE_PROMPT_TEMPLATE)

def load_examples():
    
    hdrs = []
    simulator_evaluate = SimulationBaseEvaluator(problem)
    for i in range(1, 6):
        new_hdr = CodeSegmentHDR()
        new_hdr.load(f'examples/hdr_{i}.py')
        hdrs.append(new_hdr)
        
    evaluated = simulator_evaluate(hdrs)
    for e in evaluated:
        print(e[0])
        print(e[1])
    examples = []
    evaluated.sort(key=lambda x: x[1])
    for i in range(len(evaluated)):
        examples.append((evaluated[0], 50.0 + 50.0 / len(evaluated) * i))
    
    return examples
    
for example in load_examples():
    evaluator.add_example(example[0], example[1])

# Main Se-Evo process
best = se_evo(50, problem, llm_init_func,
              co_evo_func, self_evo_func, collective_evo_func,
              llm_crossover_func, llm_mutation_func,
              selector, replace_opr, evaluator,
              init_size=26, subset_size=10,
              template_file_path='template.txt',
              pc=0.9, pm=0.2)

if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
