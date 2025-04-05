from problem import Problem, Terminal, AVAIABLE_TERMINALS
from model import CodeSegmentHDR
from typing import List, Dict
from llm import OpenRouterLLM
from population import Individual, Population, Operator
import json
from simulate import Simulator
from abc import ABC, abstractmethod
import prompt_template as pt

class MissingTemplateException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
        
def get_template(template_file: str):
    with open(template_file, 'r') as f:
        lines = f.readlines()
        return "".join(lines)
    raise MissingTemplateException("Can't not load template function")

class LLMBaseOperator(Operator):
    def __init__(self, problem, llm_model: OpenRouterLLM, prompt_template: str, 
                 timeout: float|tuple[float, float]=(30, 200)):
        super().__init__(problem)
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self.timeout = timeout
        
    def _build_prompt(self, **config):
        return self.prompt_template.format(**config)
    
    @abstractmethod
    def _build_config(self, **kwargs) -> dict[str, str]:
        pass
    
    @abstractmethod
    def _process_json_response(self, data: dict):
        pass
    
    def _build_str_from_lst(self, data: list):
        return ", ".join(str(x) for x in data)
    
    def __call__(self, **kwargs):
        config = self._build_config(**kwargs)
        prompt = self._build_prompt(**config)
        response = self.llm_model.get_response(prompt, self.timeout)
        json_repsonse = self.llm_model.extract_repsonse(response)
        return self._process_json_response(json_repsonse)

class LLMInitOperator(LLMBaseOperator):

    def _build_config(self, **kwargs):
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'init_size': str(kwargs.get('init_size')),
            'func_template': kwargs.get('func_template')
        }
    
    def _process_json_response(self, data):
        init_inds_code = data['init_inds']
        hdrs: List[CodeSegmentHDR] = []
        i = 0
        for code_json in init_inds_code:
            i += 1
            new_hdr = CodeSegmentHDR(code=code_json['code'])
            new_hdr.save(f'temp_code/hdr_{i}.py')
            hdrs.append(new_hdr)
        return hdrs
    
    def __call__(self, init_size: int, func_template: str) -> List[CodeSegmentHDR]:
        return super().__call__(init_size=init_size, func_template=func_template)
    
def test_llm_init():
    import random
    random.seed(42)
    
    terminals = AVAIABLE_TERMINALS
    
    problem = Problem(terminals)
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    init_size = 5
    
    func_template = get_template(template_file='template.txt')
    
    llm_model = OpenRouterLLM(brand='openrouter',
                              model='quasar-alpha',
                              free=False)
    
    llm_init_opr = LLMInitOperator(problem, llm_model, pt.INIT_IND_PROMPT_TEMPLATE)
    
    hdrs = llm_init_opr(init_size, func_template)
    
    for hdr in hdrs:
        print(hdr.code)
        
    llm_model.close()
  
class CoEvoOperator(LLMBaseOperator):
        
    def _make_hdrs_set_str(self, inds: List[Individual]):
        hdr_set_str = ""
        for i in range(len(inds)):
            hdr_set_str += f"HDR {i + 1}:\n"
            hdr_set_str += "----\n"
            hdr_set_str += inds[i].chromosome.code
            hdr_set_str += "----\n"
        return hdr_set_str        
    
    def _build_config(self, **kwargs):
        inds: List[Individual] = kwargs.get('inds')
        ind1: Individual = kwargs.get('ind1')
        ind2: Individual = kwargs.get('ind2')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'hdr1': ind1.chromosome.code,
            'hdr1_makespan': -ind1.fitness,
            'hdr2': ind2.chromosome.code,
            'hdr2_makespan': -ind2.fitness,
            'hdr_set': self._make_hdrs_set_str(inds)
        }
        
    def _process_json_response(self, data):
        reflection = data['reflection']
        reflected_code = data['reflected_hdr']
        inds: List[Individual] = []
        i = 0
        for code_json in reflected_code:
            i += 1
            new_hdr = CodeSegmentHDR(code=code_json['code'])
            new_hdr.save(f'co_reflected/hdr_{i}.py')
            new_ind = Individual(self.problem)
            new_ind.chromosome = new_hdr
            inds.append(new_ind)
        return reflection, inds
    
    def __call__(self, inds: List[Individual],
                 ind1: Individual,
                 ind2: Individual) -> tuple[str, list[Individual]]:
        return super().__call__(inds=inds, ind1=ind1, ind2=ind2)

def test_co_evo():
    import random
    random.seed(42)
    
    problem = Problem(AVAIABLE_TERMINALS)
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    
    llm_model = OpenRouterLLM(brand='openrouter',
                              model='quasar-alpha',
                              free=False)
    
    
    inds: List[Individual] = []
    for i in range(1, 6):
        file_path = f'temp_code/hdr_{i}.py'
        new_hdr = CodeSegmentHDR()
        new_hdr.load(file_path)
        ind = Individual(problem)
        ind.chromosome = new_hdr
        ind.fitness = -60 - i
        inds.append(ind)
    
    co_evo_opr = CoEvoOperator(problem, llm_model, pt.CO_EVO_PROMPT_TEMPLATE)
    
    reflection, reflected_hdrs = co_evo_opr(inds, inds[0], inds[1])
    
    print(reflection)
        
    llm_model.close()
    
class LLMCrossoverOperator(LLMBaseOperator):
    def _build_config(self, **kwargs):
        p1: Individual = kwargs.get('p1')
        p2: Individual = kwargs.get('p2')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'hdr1': p1.chromosome.code,
            'hdr2': p2.chromosome.code
        }
        
    def _process_json_response(self, data):
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
        
    def __call__(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        return super().__call__(p1=p1, p2=p2)
    
def test_crossover():
    import random
    random.seed(42)
    
    problem = Problem(AVAIABLE_TERMINALS)
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    
    llm_model = OpenRouterLLM(brand='openrouter',
                              model='quasar-alpha',
                              free=False)
    
    for i in range(1, 3):
        ind1 = Individual(problem)
        hdr1 = CodeSegmentHDR()
        hdr1.load(f'temp_code/hdr_{i}.py')
        ind1.chromosome = hdr1
        
        ind2 = Individual(problem)
        hdr2 = CodeSegmentHDR()
        hdr2.load(f'co_reflected/hdr_{i}.py')
        ind2.chromosome = hdr2
        
        crossover_opr = LLMCrossoverOperator(problem, llm_model, pt.CROSSOVER_PROMPT_TEMPLATE)
        
        off1, off2 = crossover_opr(ind1, ind2)
        print(off1.chromosome.code)
    
    llm_model.close()

class SelfEvoOperator(LLMBaseOperator):
    def _make_hdrs_set_str(self, inds:List[Individual]):
        hdr_set_str = ""
        for i in range(len(inds)):
            hdr_set_str += f"HDR {i + 1}:\n"
            hdr_set_str += "----\n"
            hdr_set_str += inds[i].chromosome.code
            hdr_set_str += "----\n"
        return hdr_set_str 
    
    def _build_config(self, **kwargs):
        inds_before: List[Individual] = kwargs.get('inds_before')
        inds_after: List[Individual] = kwargs.get('inds_after')
        co_evo_reflection: str = kwargs.get('co_evo_reflection')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'co_evo_reflection': co_evo_reflection,
            'hdr_before': self._make_hdrs_set_str(inds_before),
            'hdr_after': self._make_hdrs_set_str(inds_after)
        }
        
    def _process_json_response(self, data):
        reflected = data['reflected_hdr']
        inds: List[Individual] = []
        reflections: List[str] = []
        i = 0
        for json_obj in reflected:
            i += 1
            new_hdr = CodeSegmentHDR(code=json_obj['code'])
            new_hdr.save(f'self_evo_reflected/hdr_{i}.py')
            new_ind = Individual(self.problem)
            new_ind.chromosome = new_hdr
            inds.append(new_ind)
            
            reflections.append(json_obj['reflection'])
        return reflections, inds
    
    def __call__(self, inds_before: List[Individual],
                 inds_after: List[Individual],
                 co_evo_reflection: str) -> tuple[List[str], List[Individual]]:
        return super().__call__(inds_before=inds_before,
                                inds_after=inds_after,
                                co_evo_reflection=co_evo_reflection)
    
def test_self_evo():
    import random
    random.seed(42)
    
    problem = Problem(AVAIABLE_TERMINALS)
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
       
    llm_model = OpenRouterLLM(brand='openrouter',
                              model='quasar-alpha',
                              free=False)
    
    inds_before: List[Individual] = []
    for i in range(1, 3):
        file_path = f'temp_code/hdr_{i}.py'
        new_hdr = CodeSegmentHDR()
        new_hdr.load(file_path)
        ind = Individual(problem)
        ind.chromosome = new_hdr
        ind.fitness = -random.randint(60, 70)
        inds_before.append(ind)
        
    inds_after: List[Individual] = []
    for i in range(1, 3):
        file_path = f'crossover/hdr_{i}.py'
        new_hdr = CodeSegmentHDR()
        new_hdr.load(file_path)
        ind = Individual(problem)
        ind.chromosome = new_hdr
        ind.fitness = -random.randint(60, 70)
        inds_after.append(ind)
        
    co_evo_reflection = """
To improve HDR effectiveness in dynamic job shop scheduling, 
design scoring functions that integrate both job and machine urgency, 
explicitly linking deadlines and remaining operation times to current system time. 
Incorporate time-dependent features such as how close a job is to its operation deadline,
the overall job deadline, and how long it has already waited, to dynamically adjust ranking priorities. 
Use slack time to penalize jobs that are at risk of missing deadlines, 
and adapt weighting of features based on job arrival and processing dynamics. 
Building composite scores that dynamically balance current and future urgency helps anticipate bottlenecks and prioritizes jobs critical to reducing makespan.
"""
    
    self_evo_opr = SelfEvoOperator(problem, llm_model, pt.SELF_EVO_PROMPT_TEMPLATE)
    
    reflections, reflected_hdrs = self_evo_opr(inds_before, inds_after, co_evo_reflection)
    
    print(reflections)
    llm_model.close()
    
class CollectiveRefOperator(LLMBaseOperator):
        
    def _build_reflections(self, reflections: list[str]):
        res = ""
        for i in range(len(reflections)):
            res += f'Reflection {i}: {reflections[i]},\n'
        return res
    
    def _build_config(self, **kwargs):
        reflections: List[str] = kwargs.get('reflections')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'reflections': self._build_reflections(reflections)
        }
        
    def _process_json_response(self, data):
        return data['reflection']
    
    def __call__(self, reflections: list[str]) -> str:
        return super().__call__(reflections=reflections)
    
def test_collective_ref():
    import random
    random.seed(42)
    
    problem = Problem(AVAIABLE_TERMINALS)
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    llm_model = OpenRouterLLM(brand='openrouter',
                              model='quasar-alpha',
                              free=False)
    
    reflections = []
    reflections.append("In this improved HDR, we explicitly incorporated a dynamic urgency score based on the proximity of a job's next operation deadline and how long the job has waited since arrival. This mechanism better captures time-sensitive priorities. By adding this urgency to the original composite score, the scheduler became more responsive to imminent deadlines, leading to a reduction in makespan. This highlights the effectiveness of combining static job features with dynamic, time-dependent deadline urgency.")
    reflections.append('This revised HDR combines an urgency metric sensitive to operation deadlines with the base job score, adapting further via conditional adjustments based on slack time. For tight slack (js<5), the function boosts priority, encouraging timely processing; otherwise, it penalizes relaxed slack conditions. This nuanced integration of urgency and slack proved superior, dynamically emphasizing critical jobs and reducing overall makespan compared to a simpler, less adaptive function.')
    reflections.append("""To improve HDR effectiveness in dynamic job shop scheduling,
design scoring functions that integrate both job and machine urgency,
explicitly linking deadlines and remaining operation times to current system time.
Incorporate time-dependent features such as how close a job is to its operation deadline,
the overall job deadline, and how long it has already waited, to dynamically adjust ranking priorities.
Use slack time to penalize jobs that are at risk of missing deadlines,
and adapt weighting of features based on job arrival and processing dynamics.
Building composite scores that dynamically balance current and future urgency helps anticipate bottlenecks and prioritizes jobs critical to reducing makespan.""")
    
    collective_opr = CollectiveRefOperator(problem, llm_model, pt.COLLECTIVE_REF_PROMPT_TEMPLATE)
    
    reflection = collective_opr(reflections)
    print(reflection)