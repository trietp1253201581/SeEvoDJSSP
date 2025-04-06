from model import CodeSegmentHDR
from typing import List
from llm import OpenRouterLLM
from basic_evo import Individual, Population, Operator, validate
from abc import abstractmethod
import copy
import random
from problem import Problem
from simulate import Simulator

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
    
class RandomSelectOperator(Operator):
    def __init__(self, problem, random_seed: int=0):
        super().__init__(problem)
        self.random_seed=random_seed
        
    def __call__(self, population: Population, sub_size: int) -> Population:
        if sub_size > population.size:
            return copy.deepcopy(population)
        
        random.seed(self.random_seed)
        
        sub_pop = Population(size=sub_size, problem=population.problem)
        sub_pop.inds = random.sample(population.inds, k=sub_size)
        return sub_pop
    
class TopKElitismReplaceOperator(Operator):
    def __init__(self, problem, k: int):
        super().__init__(problem)
        self.k = k
        
    def __call__(self, old_pop: Population, new_pop: Population, max_size: int) -> Population:
        sorted_inds = sorted(old_pop.inds, key=lambda x: x.fitness, reverse=True)
        elites = sorted_inds[:self.k]
        num_random = max_size - self.k
        remaining_inds = sorted_inds[self.k:] + new_pop.inds
        if num_random > 0:
            choosen = random.choices(remaining_inds, k=num_random)
        else:
            choosen = []
            
        inds = elites + choosen
        pop = Population(size=max_size, problem=old_pop.problem)
        pop.inds = inds
        return pop
        
def makespan_fitness_func(sol: CodeSegmentHDR, problem: Problem):
    simulator = Simulator(hdr=sol, problem=problem, pool_size=3)
    makespan = simulator.simulate(debug=False)
    return -makespan

def se_evo(
    max_fe: int,
    problem: Problem,
    llm_init_func: LLMInitOperator,
    co_evo_func: CoEvoOperator,
    self_evo_func: SelfEvoOperator,
    collective_func: CollectiveRefOperator,
    llm_crossover_func: LLMCrossoverOperator,
    llm_mutation_func: LLMMutationOperator,
    subset_selector: RandomSelectOperator,
    replace_func: TopKElitismReplaceOperator,
    init_size: int,
    pool_size: int,
    pc: float = 0.8,
    pm: float = 0.1,
):
    # 1. Khởi tạo quần thể ban đầu
    P: Population = llm_init_func(init_size=init_size,
                                  func_template=get_template("hdr_template.py"))
    for ind in P.inds:
        ind.cal_fitness(makespan_fitness_func)
    fe = len(P.inds)

    # Vòng lặp chính
    while fe < max_fe:
        P = validate(P)
        # 2. Chọn tập con S_p
        S_p: Population = subset_selector(P, pool_size)

        # 3. Co‑Evolution: chọn 2 cá thể ngẫu nhiên, reflect toàn bộ S_p
        ind1, ind2 = random.sample(S_p.inds, 2)
        R, S_r_inds = co_evo_func(inds=S_p.inds, ind1=ind1, ind2=ind2)
        S_r = Population(size=len(S_r_inds), problem=problem)
        S_r.inds = S_r_inds

        # 4. Crossover S_p với S_r → P_inter
        P_inter = Population(size=len(S_p.inds), problem=problem)
        inter_inds = []
        for a, b in zip(S_p.inds, S_r.inds):
            if random.random() < pc:
                off1, off2 = llm_crossover_func(p1=a, p2=b)
                inter_inds.extend([off1, off2])
            else:
                inter_inds.extend([copy.deepcopy(a), copy.deepcopy(b)])
        inter_inds = inter_inds[:len(S_p.inds)]
        P_inter.inds = inter_inds

        # 5. Evaluate P_inter
        for ind in P_inter.inds:
            ind.cal_fitness(makespan_fitness_func)
        fe += len(P_inter.inds)

        # 6. Self‑Evolution → RM, I_rm
        RM, I_rm_inds = self_evo_func(
            inds_before=S_p.inds,
            inds_after=P_inter.inds,
            co_evo_reflection=R
        )

        # 7. Crossover I_rm với P_inter → P_self
        P_self = Population(size=len(I_rm_inds), problem=problem)
        self_inds = []
        for a, b in zip(I_rm_inds, P_inter.inds):
            if random.random() < pc:
                off1, off2 = llm_crossover_func(p1=a, p2=b)
                self_inds.extend([off1, off2])
            else:
                self_inds.extend([copy.deepcopy(a), copy.deepcopy(b)])
        self_inds = self_inds[:len(I_rm_inds)]
        P_self.inds = self_inds

        # 8. Evaluate P_self
        for ind in P_self.inds:
            ind.cal_fitness(makespan_fitness_func)
        fe += len(P_self.inds)

        # 9. Collective Reflection → MR
        MR = collective_func(reflections=[R] + RM)

        # 10. Mutation guided by MR → P_new
        P_new = Population(size=len(P_self.inds), problem=problem)
        new_inds = []
        for ind in P_self.inds:
            if random.random() < pm:
                mutated = llm_mutation_func(p=ind, reflection=MR)
                new_inds.append(mutated)
            else:
                new_inds.append(copy.deepcopy(ind))
        P_new.inds = new_inds

        # 11. Evaluate P_new
        for ind in P_new.inds:
            ind.cal_fitness(makespan_fitness_func)
        fe += len(P_new.inds)

        # 12. Cập nhật quần thể với elitism + random replacement
        P = replace_func(old_pop=P, new_pop=P_new, max_size=P.size)

    # Kết thúc: trả về best individual
    best = min(P.inds, key=lambda ind: ind.fitness)
    return best