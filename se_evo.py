from problem import Problem
from model import Terminal, AVAIABLE_TERMINALS, CodeSegmentHDR
from typing import List
from llm import OpenRouterLLM
from evolution import InitOperator, Operator
import json

class MissingTemplateException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

def get_template(template_file: str):
    with open(template_file, 'r') as f:
        lines = f.readlines()
        return "".join(lines)
    raise MissingTemplateException("Can't not load template function")

class LLMInitOperator(InitOperator):
    
    PROMPT = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

We need to generate an init sets of {init_size} HDRs to sort incoming job. Each HDR is a code segment describe a python function with template:

{func_template}

You can use a single return expression like:

def hdr(...):
    return ...

Or, use any control structrue like if-then-else or for loop like:

def hdr(...):
    if js > 5:
        return jw
    return jcd

. Note that the return value should be not to large.

Your response MUST ONLY include the init HDRs sets in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "init_inds": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
        ...,
        {{"code": "<hdr_n>"}}
    ]
}}
where n equals to {init_size}.
'''
    
    def __init__(self, problem, terminals, init_size, llm_model: OpenRouterLLM, func_template: str):
        super().__init__(problem, terminals, init_size)
        self.llm_model = llm_model
        self.func_template = func_template
        
    def _process_response(self, data):
        init_inds_code = data['init_inds']
        hdrs: List[CodeSegmentHDR] = []
        i = 0
        for code_json in init_inds_code:
            i += 1
            new_hdr = CodeSegmentHDR(code=code_json['code'])
            new_hdr.save(f'temp_code/hdr_{i}.py')
            hdrs.append(new_hdr)
        return hdrs
        
        
    def operate(self):
        # Build prompt
        job_str = ", ".join(str(job) for job in self.problem.jobs)
        machine_str = ", ".join(str(machine) for machine in self.problem.machines)
        terminals_str = ", ".join(str(terminal) for terminal in self.terminals)
        
        prompt = LLMInitOperator.PROMPT.format(
            job_str=job_str,
            machine_str=machine_str,
            terminal_set=terminals_str,
            init_size=str(self.init_size),
            func_template=self.func_template
        )
        
        print(prompt)
        
        response = self.llm_model.get_response(prompt, timeout=(30, 200))
        print(response)
        response_json = self.llm_model.extract_repsonse(response)
        print(response_json)

        return self._process_response(response_json)
    
class CoEvoOperator(Operator):
    PROMPT = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 HDRs describe by a python function (with its makespan) are:
HDR1 with makespan {hdr1_makespan}
-------
{hdr1}
-------

HDR1 with makespan {hdr2_makespan}
-------
{hdr2}
-------

Ưe need to compare these 2 examples and create a reflection to improve the other HDRs to make them more effective.
Reflection is a text created from the 2 examples above, when comparing their effectiveness.
Note that reflection should not contain information about comparing 2 HDRs, but only information about how to improve. That means you will still do the comparison but not add it to the reflection.

After that, we need to apply this reflection to each HDR in a set, to improve HDR effectiveness. The HDRs set is
{hdr_set}

Your response MUST ONLY include the init HDRs sets in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "reflection": "<your_reflection_text>",
    "reflected_hdr": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
        ...,
        {{"code": "<hdr_n>"}}
    ]
}}
where n equals to the size of hdr set and where hdr_i is the i-th element in the hdr set after apply your reflection.
'''
    def __init__(self, problem, terminals: List[Terminal], hdrs: List[CodeSegmentHDR],
                 hdr1_with_makespan: tuple[CodeSegmentHDR, float],
                 hdr2_with_makespan: tuple[CodeSegmentHDR, float],
                 llm_model: OpenRouterLLM):
        super().__init__(problem)
        self.terminals = terminals
        self.hdrs = hdrs
        self.hdr1_with_makespan = hdr1_with_makespan
        self.hdr2_with_makespan = hdr2_with_makespan
        self.llm_model = llm_model
        
    def _make_hdrs_set_str(self):
        hdr_set_str = ""
        for i in range(len(self.hdrs)):
            hdr_set_str += f"HDR {i + 1}:\n"
            hdr_set_str += "----\n"
            hdr_set_str += self.hdrs[i].code
            hdr_set_str += "----\n"
        return hdr_set_str        
    
    def _process_response(self, data):
        reflection = data['reflection']
        reflected_code = data['reflected_hdr']
        hdrs: List[CodeSegmentHDR] = []
        i = 0
        for code_json in reflected_code:
            i += 1
            new_hdr = CodeSegmentHDR(code=code_json['code'])
            new_hdr.save(f'co_reflected/hdr_{i}.py')
            hdrs.append(new_hdr)
        return reflection, hdrs
        
    def operate(self):
        # Build prompt
        job_str = ", ".join(str(job) for job in self.problem.jobs)
        machine_str = ", ".join(str(machine) for machine in self.problem.machines)
        terminals_str = ", ".join(str(terminal) for terminal in self.terminals)
        
        prompt = CoEvoOperator.PROMPT.format(
            job_str=job_str,
            machine_str=machine_str,
            terminal_set=terminals_str,
            hdr1=self.hdr1_with_makespan[0],
            hdr1_makespan=self.hdr1_with_makespan[1],
            hdr2=self.hdr2_with_makespan[0],
            hdr2_makespan=self.hdr2_with_makespan[1],
            hdr_set=self._make_hdrs_set_str()
        )
        
        print(prompt)
        
        response = self.llm_model.get_response(prompt, timeout=(30, 200))
        print(response)
        response_json = self.llm_model.extract_repsonse(response)
        print(response_json)

        return self._process_response(response_json)

def test_llm_init():
    import random
    random.seed(42)
    
    problem = Problem()
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    init_size = 5
    terminals = AVAIABLE_TERMINALS
    
    func_template = get_template(template_file='template.txt')
    
    llm_model = OpenRouterLLM('openrouter/quasar-alpha')
    
    llm_init_opr = LLMInitOperator(problem, AVAIABLE_TERMINALS, init_size, 
                                   llm_model, func_template)
    
    hdrs = llm_init_opr()
    
    for hdr in hdrs:
        print(hdr.code)
        
    llm_model.close()
    
def test_co_evo():
    import random
    random.seed(42)
    
    problem = Problem()
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    init_size = 5
    terminals = AVAIABLE_TERMINALS
    
    func_template = get_template(template_file='template.txt')
    
    llm_model = OpenRouterLLM(brand='openrouter',
                              model='quasar-alpha',
                              free=False)
    
    
    hdrs: List[CodeSegmentHDR] = []
    for i in range(1, 6):
        file_path = f'temp_code/hdr_{i}.py'
        new_hdr = CodeSegmentHDR()
        new_hdr.load(file_path)
        hdrs.append(new_hdr)
    
    co_evo_opr = CoEvoOperator(problem, AVAIABLE_TERMINALS, hdrs,
                               (hdrs[0], 60), (hdrs[1], 65), llm_model)
    
    reflection, reflected_hdrs = co_evo_opr()
    
    print(reflection)
        
    llm_model.close()
    
test_co_evo()