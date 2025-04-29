from se_evo import SelfEvoEngine
from evaluate import SurrogateEvaluator, ChromaMetaVectorStore, VectorEmbedding, SentenceEmbedding
from llm import GoogleAIStudioLLM
from problem import Problem, AVAIABLE_TERMINALS
import random
import prompt_template as pt

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

hdrs = [ind.chromosome for ind in se_engine.P.inds]

llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='config.json',
                              runtime_config='llm_runtime_config.json')

vector_embedding = VectorEmbedding(input_dim=11, embedding_dim=50)
hdr_embedding = SentenceEmbedding()


fitness_eval = SurrogateEvaluator(problem, vector_store, hdr_embedding, vector_embedding, 
                                  prompt_template=pt.SURROGATE_EVALUATION_PROMPT_TEMPLATE, 
                                  llm_model=llm_model, batch_size=10, max_retries=3)
#fitness_eval.is_exact_evaluation = True
fitness_eval.is_exact_evaluation = False
results = fitness_eval(hdrs[5:6])

llm_model.close()