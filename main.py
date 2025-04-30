from llm import OpenRouterLLM, GoogleAIStudioLLM
from model import CodeSegmentHDR
from problem import Problem, AVAIABLE_TERMINALS
import random
from se_evo import LLMInitOperator, LLMCrossoverOperator, LLMMutationOperator, \
    CoEvoOperator, SelfEvoOperator, CollectiveRefOperator, RandomSelectOperator, \
        TopKElitismReplaceOperator, SelfEvoEngine
        
from evaluate import SimulationBaseEvaluator, EventDrivenLLMSurrogateEvaluator, \
    ChromaMetaVectorStore, MLPSurrogateEvaluator, VectorEmbedding, SentenceEmbedding, \
        MLPSurrogateModel
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
file_handler = logging.FileHandler(f'process_{datetime.now().strftime("%Y_%m_%d")}.log')
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
problem.custom_generate(num_jobs=240, max_oprs_each_job=5, 
                        num_machines=20, max_arr_time=400, 
                        arrival_type='burst', proc_dist='bimodal', 
                        deadline_factor=1.4)


# for job in problem.jobs:
#   print(str(job))
    
# Build llm
#llm_model = OpenRouterLLM('deepseek', 'deepseek-r1-zero', free=True, timeout=(60, 600))
llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='config.json',
                              runtime_config='llm_runtime_config.json')

# Create Operator
llm_init_func = LLMInitOperator(problem, llm_model, prompt_template=pt.INIT_IND_PROMPT_TEMPLATE)
llm_crossover_func = LLMCrossoverOperator(problem, llm_model, prompt_template=pt.CROSSOVER_PROMPT_TEMPLATE)
llm_mutation_func = LLMMutationOperator(problem, llm_model, prompt_template=pt.MUTATION_PROMPT_TEMPLATE)
co_evo_func = CoEvoOperator(problem, llm_model, prompt_template=pt.CO_EVO_PROMPT_TEMPLATE)
self_evo_func = SelfEvoOperator(problem, llm_model, prompt_template=pt.SELF_EVO_PROMPT_TEMPLATE)
collective_evo_func = CollectiveRefOperator(problem, llm_model, prompt_template=pt.COLLECTIVE_REF_PROMPT_TEMPLATE)
selector = RandomSelectOperator(problem)
replace_opr = TopKElitismReplaceOperator(problem, k=2)
    
#evaluator = EventDrivenLLMSurrogateEvaluator(llm_model, problem,
#                                             prompt_template=pt.EVENT_SURROGATE_PROMPT_TEMPLATE, 
#                                             num_segments=4, batch_size=4,
#                                             max_retries=4,
#                                             scaling_schedule='linear',
#                                             start_rate=0.7,
#                                             max_calls_to_end=35)
evaluator = SimulationBaseEvaluator(problem)

'''vector_store = ChromaMetaVectorStore(
    persist_directory='./chroma_db',
    collection_name='se_evo_collection'
)
vector_store.clear()

vector_embedding = VectorEmbedding(input_dim=11, embedding_dim=50)
hdr_embedding = SentenceEmbedding()

surrogate_model = MLPSurrogateModel(input_dim=384 + 50, hidden_dim=128, dropout_rate=0.1)

evaluator = MLPSurrogateEvaluator(problem, vector_store, hdr_embedding, vector_embedding, 
                                  surrogate_model,
                                  prompt_template=pt.SURROGATE_EVALUATION_PROMPT_TEMPLATE, 
                                  llm_model=llm_model, batch_size=12, max_retries=3,
                                  train_cycle=3, n_dropout=4, train_epoch=10, finetune_epoch=3,
                                  ucb_lambda=0.2, max_hdr_to_finetune=4)'''

# Main Se-Evo process
se_engine = SelfEvoEngine(
    problem, llm_init_func, co_evo_func, self_evo_func, collective_evo_func,
    llm_crossover_func, llm_mutation_func, selector, replace_opr, evaluator,
    max_retries=3
)

best = se_engine.run(
    num_gen=2,
    init_size=24, subset_size=8, template_file='template.txt',
    pc=0.8, pm=0.1, state='new'
)

se_engine.save_state('checkpoint_surrogate_based.pkl', fields_to_save=['P', 'gen', 'best', 'solve_time', 'fe'])
if isinstance(evaluator, MLPSurrogateEvaluator):
    evaluator.save_state('evaluator_checkpoint.pkl', fields_to_save=['call_cnt', '_temp_store', 'output_scale_factor', 
                                                                 '_problem_vector', 'is_exact_evaluation'])
    evaluator.surrogate_model.save_state_dict('surrogate_model_checkpoint.pth')

if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
    print(f"Time: {se_engine.solve_time:.2f}s")
    os.makedirs('best_solution', exist_ok=True)
    best.chromosome.save(f'best_after_gen_{se_engine.gen}.py')
llm_model.close()