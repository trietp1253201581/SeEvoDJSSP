from problem import Problem
from model import Terminal, AVAIABLE_TERMINALS
from typing import List
from llm import OpenRouterLLM

class MissingTemplateException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

def get_template(template_file: str):
    with open(template_file, 'r') as f:
        lines = f.readlines()
        return "".join(lines)
    raise MissingTemplateException("Can't not load template function")

INIT_IND_PROMPT = """Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

We need to generate an init sets of {init_size} HDRs to sort incoming job. Each HDR is a code segment describe a python function with template:

{func_template}

You can use any control structrue like if-then-else or for loop like:

def hdr(...):
    if js > 5:
        return jw
    return jcd

. Note that the return value should be not to large.

Your response MUST ONLY include the init HDRs sets in following JSON format with no additional text.
{{
    "init_inds": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
        ...,
        {{"code": "<hdr_n>"}}
    ]
}}
where n equals to {init_size}.
"""

def init_ind(problem: Problem, init_size: int, 
             func_template: str, terminals: List[Terminal],
             llm_model: OpenRouterLLM):
    # Build prompt
    job_str = ", ".join(str(job) for job in problem.jobs)
    machine_str = ", ".join(str(machine) for machine in problem.machines)
    terminals_str = ", ".join(str(terminal) for terminal in terminals)
    
    prompt = INIT_IND_PROMPT.format(
        job_str=job_str,
        machine_str=machine_str,
        terminal_set=terminals_str,
        init_size=str(init_size),
        func_template=func_template
    )
    
    response = llm_model.get_response(prompt, timeout=(30, 200))
    print(response)
    
def test():
    import random
    random.seed(42)
    
    problem = Problem()
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    init_size = 5
    terminals = AVAIABLE_TERMINALS
    
    func_template = get_template(template_file='template.txt')
    llm_model = OpenRouterLLM('deepseek-v3-0324')
    
    init_ind(problem, init_size, func_template, terminals, llm_model)
    
    import time
    time.sleep(20)
    llm_model.close()
    
test()