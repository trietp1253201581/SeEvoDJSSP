from llm import OpenRouterLLM, GoogleAIStudioLLM
from model import CodeSegmentHDR
from problem import Problem, AVAIABLE_TERMINALS
import random
from se_evo import LLMInitOperator, LLMCrossoverOperator, LLMMutationOperator, \
    CoEvoOperator, SelfEvoOperator, CollectiveRefOperator, RandomSelectOperator, \
        TopKElitismReplaceOperator, SelfEvoEngine
        
from evaluate import SimulationBaseEvaluator, StaticLLMSurrogateEvaluator, EventDrivenLLMSurrogateEvaluator
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

problem = Problem(AVAIABLE_TERMINALS, pool_size=15)
problem.custom_generate(num_jobs=120, max_oprs_each_job=5, num_machines=20, max_arr_time=120, arrival_type='uniform', proc_dist='uniform', deadline_factor=1.4)


# for job in problem.jobs:
#   print(str(job))
    
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
    
#evaluator = EventDrivenLLMSurrogateEvaluator(llm_model, problem, prompt_template=pt.SURROGATE_PROMPT_TEMPLATE, num_segments=4, batch_size=6)
evaluator = SimulationBaseEvaluator(problem)
# Main Se-Evo process
se_engine = SelfEvoEngine(
    problem, llm_init_func, co_evo_func, self_evo_func, collective_evo_func,
    llm_crossover_func, llm_mutation_func, selector, replace_opr, evaluator,
    max_retries=3
)

best = se_engine.run(
    max_fe=500,
    init_size=36, subset_size=12, template_file='template.txt',
    pc=0.8, pm=0.1, state='new'
)

se_engine.save_state('checkpoint_simulation_based.pkl')

if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
    
llm_model.close()
