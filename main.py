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
import os

# Set logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Định dạng chung
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(name)s (%(filename)s:%(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler ghi vào file
file_handler = logging.FileHandler(f'/kaggle/working/SeEvoDJSSP/process_{datetime.now().strftime("%Y_%m_%d")}.log')
file_handler.setFormatter(formatter)

# Handler ghi ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Thêm cả 2 handler vào logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
        
# Create problem 
random.seed(42)

problem = Problem(AVAIABLE_TERMINALS, pool_size=15)
problem.custom_generate(num_jobs=250, max_oprs_each_job=5, num_machines=20, max_arr_time=120, arrival_type='uniform', proc_dist='uniform', deadline_factor=1.4)


# for job in problem.jobs:
#   print(str(job))
    
# Build llm
#llm_model = OpenRouterLLM('deepseek', 'deepseek-r1-zero', free=True, timeout=(60, 600))
llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='/kaggle/working/SeEvoDJSSP/config.json',
                              runtime_config='/kaggle/working/SeEvoDJSSP/llm_runtime_config.json')

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
    num_gen=500,
    init_size=36, subset_size=12, template_file='/kaggle/working/SeEvoDJSSP/template.txt',
    pc=0.8, pm=0.1, state='new'
)

se_engine.save_state('/kaggle/working/SeEvoDJSSP/checkpoint_simulation_based.pkl')

if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
    print(f"Time: {se_engine.solve_time:.2f}s")
    os.makedirs('/kaggle/working/SeEvoDJSSP/best_solution', exist_ok=True)
    best.chromosome.save(f'/kaggle/working/SeEvoDJSSP/best_solution/best_{se_engine.fe}.py')
llm_model.close()
