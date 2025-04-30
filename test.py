from se_evo import SelfEvoEngine
from evaluate import MLPSurrogateModel, SurrogateEvaluator, ChromaMetaVectorStore, VectorEmbedding, SentenceEmbedding
from llm import GoogleAIStudioLLM
from problem import Problem, AVAIABLE_TERMINALS
import random
import prompt_template as pt


# Set logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Định dạng chung
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(name)s (%(filename)s:%(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler ghi ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Thêm cả 2 handler vào logger
#logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Create problem 
random.seed(42)

problem = Problem(AVAIABLE_TERMINALS, pool_size=8)
problem.custom_generate(num_jobs=24, max_oprs_each_job=5, 
                        num_machines=10, max_arr_time=60, 
                        arrival_type='uniform', proc_dist='uniform', 
                        deadline_factor=1.4)
se_engine = SelfEvoEngine(*[None for _ in range(10)])

se_engine.load_state('checkpoint_surrogate_based2.pkl', fields_to_update=['P'])

#for ind in se_engine.P.inds:
#    print(ind.fitness)
    
vector_store = ChromaMetaVectorStore(
    persist_directory='./chroma_db',
    collection_name='se_evo_collection'
)
vector_store.clear()
hdrs = [ind.chromosome for ind in se_engine.P.inds]

llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='config.json',
                              runtime_config='llm_runtime_config.json')

vector_embedding = VectorEmbedding(input_dim=11, embedding_dim=50)
hdr_embedding = SentenceEmbedding()

surrogate_model = MLPSurrogateModel(input_dim=384 + 50, hidden_dim=128, dropout_rate=0.1)

fitness_eval = SurrogateEvaluator(problem, vector_store, hdr_embedding, vector_embedding, 
                                  surrogate_model,
                                  prompt_template=pt.SURROGATE_EVALUATION_PROMPT_TEMPLATE, 
                                  llm_model=llm_model, batch_size=10, max_retries=3,
                                  train_cycle=2, n_dropout=4, train_epoch=8, finetune_epoch=3,
                                  ucb_lambda=0.3)

all_results = []

# Exact evaluation
fitness_eval.is_exact_evaluation = True
results = fitness_eval(hdrs[:10])
all_results.extend(results)
print(len(fitness_eval._temp_store))

# Surrogate evaluation
fitness_eval.is_exact_evaluation = False
results = fitness_eval(hdrs[10:15])
all_results.extend(results)
print(len(fitness_eval._temp_store))

fitness_eval.is_exact_evaluation = False
results = fitness_eval(hdrs[15:20])
all_results.extend(results)
print(len(fitness_eval._temp_store))

for hdr, score in all_results:
    print(f':{score}')
print(fitness_eval.output_scale_factor)
llm_model.close()