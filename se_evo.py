from problem import Problem
from model import Terminal, AVAIABLE_TERMINALS, CodeSegmentHDR
from typing import List
from llm import OpenRouterLLM
from evolution import InitOperator, Operator, CrossoverOperator, Individual
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

INIT_IND_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
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

class LLMInitOperator(InitOperator):
    
    def __init__(self, problem, terminals, llm_model: OpenRouterLLM, 
                 func_template: str, prompt_template: str):
        super().__init__(problem, terminals)
        self.llm_model = llm_model
        self.func_template = func_template
        self.prompt_template = prompt_template
        
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
        
        
    def operate(self, init_size: int):
        # Build prompt
        job_str = ", ".join(str(job) for job in self.problem.jobs)
        machine_str = ", ".join(str(machine) for machine in self.problem.machines)
        terminals_str = ", ".join(str(terminal) for terminal in self.terminals)
        
        prompt = self.prompt_template.format(
            job_str=job_str,
            machine_str=machine_str,
            terminal_set=terminals_str,
            init_size=str(init_size),
            func_template=self.func_template
        )
        
        response = self.llm_model.get_response(prompt, timeout=(30, 200))

        response_json = self.llm_model.extract_repsonse(response)
      
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
    
    llm_init_opr = LLMInitOperator(problem, AVAIABLE_TERMINALS, 
                                   llm_model, func_template, INIT_IND_PROMPT_TEMPLATE)
    
    hdrs = llm_init_opr(init_size)
    
    for hdr in hdrs:
        print(hdr.code)
        
    llm_model.close()
  
CO_EVO_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 HDRs describe by a python function (with its makespan) are:
HDR1 with makespan {hdr1_makespan}
-------
{hdr1}
-------

HDR2 with makespan {hdr2_makespan}
-------
{hdr2}
-------

We need to compare these 2 examples and create a reflection to improve the other HDRs to make them more effective.
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
  
class CoEvoOperator(Operator):
    
    def __init__(self, problem, terminals: List[Terminal], hdrs: List[CodeSegmentHDR],
                 hdr1_with_makespan: tuple[CodeSegmentHDR, float],
                 hdr2_with_makespan: tuple[CodeSegmentHDR, float],
                 llm_model: OpenRouterLLM,
                 prompt_template: str):
        super().__init__(problem)
        self.terminals = terminals
        self.hdrs = hdrs
        self.hdr1_with_makespan = hdr1_with_makespan
        self.hdr2_with_makespan = hdr2_with_makespan
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        
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
        
        prompt = self.prompt_template.format(
            job_str=job_str,
            machine_str=machine_str,
            terminal_set=terminals_str,
            hdr1=self.hdr1_with_makespan[0],
            hdr1_makespan=self.hdr1_with_makespan[1],
            hdr2=self.hdr2_with_makespan[0],
            hdr2_makespan=self.hdr2_with_makespan[1],
            hdr_set=self._make_hdrs_set_str()
        )
        
        response = self.llm_model.get_response(prompt, timeout=(30, 200))
        response_json = self.llm_model.extract_repsonse(response)
        return self._process_response(response_json)
    
    def __call__(self):
        return self.operate()

def test_co_evo():
    import random
    random.seed(42)
    
    problem = Problem()
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    terminals = AVAIABLE_TERMINALS
    
    
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
                               (hdrs[0], 60), (hdrs[1], 65), llm_model, CO_EVO_PROMPT_TEMPLATE)
    
    reflection, reflected_hdrs = co_evo_opr()
    
    print(reflection)
        
    llm_model.close()
    
CROSSOVER_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 HDRs describe by a python function are:
HDR1
-------
{hdr1}
-------

HDR2
-------
{hdr2}
-------

We need to recombine 2 above parent HDRs to create 2 new children HDRs just use what is already in the 2 parent HDRs.

Your response MUST ONLY include the init HDRs sets in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "recombined_hdr": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
    ]
}}
where hdr_1, hdr_2 are 2 new recombined hdr.
'''  
    
class LLMCrossoverOperator(CrossoverOperator):
    def __init__(self, problem, terminals: List[Terminal], 
                 llm_model: OpenRouterLLM, prompt_template: str):
        super().__init__(problem)
        self.terminals = terminals
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        
    def _process_response(self, data):
        recombined_code = data['recombined_hdr']
        inds: List[Individual] = []
        i = 0
        for code_json in recombined_code:
            i += 1
            new_hdr = CodeSegmentHDR(code=code_json['code'])
            new_hdr.save(f'crossover/hdr_{i}.py')
            new_inds = Individual(self.problem)
            new_inds.chromosome = new_hdr
            inds.append(new_inds)
        return inds[0], inds[1]
        
    def operate(self, p1, p2):
        # Build prompt
        job_str = ", ".join(str(job) for job in self.problem.jobs)
        machine_str = ", ".join(str(machine) for machine in self.problem.machines)
        terminals_str = ", ".join(str(terminal) for terminal in self.terminals)
        
        prompt = self.prompt_template.format(
            job_str=job_str,
            machine_str=machine_str,
            terminal_set=terminals_str,
            hdr1=p1.chromosome.code,
            hdr2=p2.chromosome.code 
        )
        
        print(prompt)
        response = self.llm_model.get_response(prompt, timeout=(30, 200))
        response_json = self.llm_model.extract_repsonse(response)
        return self._process_response(response_json)
    
def test_crossover():
    import random
    random.seed(42)
    
    problem = Problem()
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    terminals = AVAIABLE_TERMINALS
    
    
    llm_model = OpenRouterLLM(brand='openrouter',
                              model='quasar-alpha',
                              free=False)
    
    ind1 = Individual(problem)
    hdr1 = CodeSegmentHDR()
    hdr1.load('temp_code/hdr_1.py')
    ind1.chromosome = hdr1
    
    ind2 = Individual(problem)
    hdr2 = CodeSegmentHDR()
    hdr2.load('co_reflected/hdr_1.py')
    ind2.chromosome = hdr2
    
    print(ind1.chromosome)
    print(ind2.chromosome)
    
    crossover_opr = LLMCrossoverOperator(problem, terminals, llm_model, CROSSOVER_PROMPT_TEMPLATE)
    
    off1, off2 = crossover_opr(ind1, ind2)
    
    print(off1.chromosome)
    print(off2.chromosome)
    
    llm_model.close()
    