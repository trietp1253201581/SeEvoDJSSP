from problem import Problem, Terminal, AVAIABLE_TERMINALS
from model import CodeSegmentHDR
from typing import List, Dict
from llm import OpenRouterLLM
from basic_evo import Individual, Population, Operator
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
        i = 0
        pop = Population(size=len(data['init_inds']), problem=self.problem)
        for code_json in init_inds_code:
            i += 1
            new_hdr = CodeSegmentHDR(code=code_json['code'])
            new_ind = Individual(self.problem)
            new_ind.chromosome = new_hdr
            pop.inds.append(new_ind)
        return pop
    
    def __call__(self, init_size: int, func_template: str) -> Population:
        return super().__call__(init_size=init_size, func_template=func_template)
  
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
            new_ind = Individual(self.problem)
            new_ind.chromosome = new_hdr
            inds.append(new_ind)
        return reflection, inds
    
    def __call__(self, inds: List[Individual],
                 ind1: Individual,
                 ind2: Individual) -> tuple[str, list[Individual]]:
        return super().__call__(inds=inds, ind1=ind1, ind2=ind2)
    
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
            new_inds = Individual(self.problem)
            new_inds.chromosome = new_hdr
            inds.append(new_inds)
        return inds[0], inds[1]
        
    def __call__(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        return super().__call__(p1=p1, p2=p2)

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
    
class LLMMutationOperator(LLMBaseOperator):
    
    def _make_hdrs_set_str(self, inds:List[Individual]):
        hdr_set_str = ""
        for i in range(len(inds)):
            hdr_set_str += f"HDR {i + 1}:\n"
            hdr_set_str += "----\n"
            hdr_set_str += inds[i].chromosome.code
            hdr_set_str += "----\n"
        return hdr_set_str 
    
    def _build_config(self, **kwargs):
        reflection = kwargs.get('reflection')
        p: Individual = kwargs.get('p')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'reflection': reflection,
            'hdr': p.chromosome.code
        }
        
    def _process_json_response(self, data):
        code = data['rephrased_hdr']
        new_hdr = CodeSegmentHDR(code)
        new_ind = Individual(self.problem)
        new_ind.chromosome = new_hdr
        return new_ind
    
    def __call__(self, p: Individual, reflection: str) -> Individual:
        return super().__call__(p=p, reflection=reflection)