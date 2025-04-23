from llm import GoogleAIStudioLLM, OpenRouterLLM
from evaluate import EventDrivenLLMSurrogateEvaluator
from problem import Problem, AVAIABLE_TERMINALS
import random
import prompt_template as pt
from model import CodeSegmentHDR
random.seed(42)

problem = Problem(AVAIABLE_TERMINALS, pool_size=12)
problem.random_generate(num_jobs=200, max_oprs_each_job=30, num_machines=30, max_arr_time=600)

llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600))

evaluator = EventDrivenLLMSurrogateEvaluator(llm_model, problem, prompt_template=pt.SURROGATE_PROMPT_TEMPLATE, num_segments=4)

def test_event_driven_surrogate():
    print("----- TEST EVENT DRIVEN SURROGATE -----")
    hdrs = []
    for i in range(1,6):
        hdr = CodeSegmentHDR()
        hdr.load(f'examples/hdr_{i}.py')
        hdrs.append(hdr)
        
    results = evaluator(hdrs)
    print(results)
    
test_event_driven_surrogate()
    
    
    
        


