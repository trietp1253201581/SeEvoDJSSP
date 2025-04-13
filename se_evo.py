from model import CodeSegmentHDR, HDRException
from typing import List, Tuple
from llm import OpenRouterLLM, LLMException
from basic_evo import Individual, Population, Operator, validate, FITNESS_FUNC_TYPE
from abc import abstractmethod
import copy
import random
from problem import Problem
from simulate import Simulator
import datetime

import time
import logging
logging.basicConfig(filename='process.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

random.seed(42)

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
            try:
                new_hdr = CodeSegmentHDR(code=code_json['code'])
                new_ind = Individual(self.problem)
                new_ind.chromosome = new_hdr
                pop.inds.append(new_ind)
            except HDRException as e:
                logging.error(str(type(e)) + e.msg)
                continue
        return pop
    
    def __call__(self, init_size: int, func_template: str) -> Population:
        return super().__call__(init_size=init_size, func_template=func_template)
  
class CoEvoOperator(LLMBaseOperator):
    
    def _build_pairs(self, inds: List[Individual]):
        inds_copy = copy.deepcopy(inds)
        pairs = []
        while len(inds_copy) >= 2:
            ind1_idx = random.randint(0, len(inds_copy) - 1)
            ind1 = inds_copy.pop(ind1_idx)  # Xoá và lấy ra ind1

            ind2_idx = random.randint(0, len(inds_copy) - 1)
            ind2 = inds_copy.pop(ind2_idx)  # Xoá và lấy ra ind2

            pairs.append((ind1, ind2))
            
        pair_str = "--------------\n"
        for i in range(len(pairs)):
            pair_str += f"Pair {i}:\n"
            ind1, ind2 = pairs[i]
            pair_str += f"HDR 1 with makespan {-ind1.fitness}: \n"
            pair_str += str(ind1.chromosome.code)
            pair_str += "\n"
            pair_str += f"HDR 2 with makespan {-ind2.fitness}: \n"
            pair_str += str(ind2.chromosome.code)
            pair_str += "--------------\n"
            
        return pair_str 
    
    def _build_config(self, **kwargs):
        inds: List[Individual] = kwargs.get('inds')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'pairs': self._build_pairs(inds)
        }
        
    def _process_json_response(self, data):
        results = data['results']
        inds: List[Individual] = []
        for json_obj in results:
            try:
                new_hdr = CodeSegmentHDR(code=json_obj['code'])
                new_ind = Individual(self.problem)
                new_ind.chromosome = new_hdr
                new_ind.reflection = json_obj['reflection']
                inds.append(new_ind)
            except HDRException as e:
                logging.error(str(type(e)) + e.msg)
                continue
        return inds
    
    def __call__(self, inds: List[Individual]) -> List[Individual]:
        return super().__call__(inds=inds)
    
class LLMCrossoverOperator(LLMBaseOperator):
    def _build_config(self, **kwargs):
        p1: Individual = kwargs.get('p1')
        p2: Individual = kwargs.get('p2')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'hdr1': p1.chromosome.code,
            'hdr2': p2.chromosome.code,
            'ref1': p1.reflection,
            'ref2': p2.reflection
        }
        
    def _process_json_response(self, data):
        recombined_code = data['recombined_hdr']
        inds: List[Individual] = []
        i = 0
        for code_json in recombined_code:
            try:
                i += 1
                new_hdr = CodeSegmentHDR(code=code_json['code'])
                new_ind = Individual(self.problem)
                new_ind.chromosome = new_hdr
                inds.append(new_ind)
            except HDRException as e:
                logging.error(str(e) + ":" + e.msg)
                return None, None

        return inds[0], inds[1]
        
    def __call__(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        return super().__call__(p1=p1, p2=p2)

class SelfEvoOperator(LLMBaseOperator):
    def _build_pairs(self, compare_hdrs: List[Tuple[Individual, Individual, str]]):
        pair_str = ""
        for i in range(len(compare_hdrs)):
            pair_str += f"Pair {i+1}:\n"
            ind1, ind2, co_ref = compare_hdrs[i]
            pair_str += f"Co-Evo Reflection have used: {co_ref}. \n"
            pair_str += f"HDR before apply reflection with makespan {-ind1.fitness}: \n"
            pair_str += str(ind1.chromosome.code)
            pair_str += "\n"
            pair_str += f"HDR after apply reflection with makespan {-ind2.fitness}: \n"
            pair_str += str(ind2.chromosome.code)
            pair_str += "--------------\n"
            
        return pair_str 
    
    def _build_config(self, **kwargs):
        compare_hdrs = kwargs.get('compare_hdrs')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'pairs': self._build_pairs(compare_hdrs)
        }
        
    def _process_json_response(self, data):
        reflected = data['reflected_hdr']
        inds: List[Individual] = []
        i = 0
        for json_obj in reflected:
            i += 1
            try:
                new_hdr = CodeSegmentHDR(code=json_obj['code'])
                new_ind = Individual(self.problem)
                new_ind.chromosome = new_hdr
                new_ind.reflection = json_obj['reflection']
                inds.append(new_ind)
            except HDRException as e:
                logging.error(str(type(e)) + e.msg)
                continue
        return inds
    
    def __call__(self, compare_hdrs: List[Tuple[Individual, Individual, str]]) -> List[Individual]:
        return super().__call__(compare_hdrs=compare_hdrs)
    
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
    
    def _build_config(self, **kwargs):
        p: Individual = kwargs.get('p')
        reflection = kwargs.get('reflection')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'reflection': reflection,
            'hdr': p.chromosome.code
        }
        
    def _process_json_response(self, data):
        try:
            code = data['rephrased_hdr']
            new_hdr = CodeSegmentHDR(code)
            new_ind = Individual(self.problem)
            new_ind.chromosome = new_hdr
        except HDRException as e:
            logging.error(str(type(e)) + ":" + e.msg)
            return None
        return new_ind
    
    def __call__(self, p: Individual, reflection: str) -> Individual:
        return super().__call__(p=p, reflection=reflection)
        
    
class RandomSelectOperator(Operator):
    def __init__(self, problem):
        super().__init__(problem)
        
    def __call__(self, population: Population, sub_size: int) -> Population:
        if sub_size > population.size:
            return copy.deepcopy(population)
        
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
    simulator = Simulator(hdr=sol, problem=problem)
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
    fitness_func: FITNESS_FUNC_TYPE,
    init_size: int,
    subset_size: int,
    template_file_path: str,
    pc: float = 0.8,
    pm: float = 0.1,
):
    
    logging.info(f"Start trial {datetime.datetime.now()}")
    
    logging.info("Init phase")
    # 1. Khởi tạo quần thể ban đầu
    try:
        P: Population = llm_init_func(init_size=init_size,
                                  func_template=get_template(template_file_path))
    except LLMException as e:
        logging.error(str(type(e)) + ":" + e.msg)
        return None
    for ind in P.inds:
        try:
            ind.cal_fitness(makespan_fitness_func)
        except:
            ind.fitness = Individual.DEFAULT_FITNESS
            logging.error(f"Failed to cal fitness.")
    fe = len(P.inds)
    best = max(P.inds, key=lambda ind: ind.fitness)
    logging.info(f"FE = {fe}, best HDR with fitness {best.fitness:.2f}")
    # Vòng lặp chính
    num_gen = 1
    while fe < max_fe:
        logging.info(f"Gen {num_gen}:")
        num_gen += 1
        if len(P.inds) == 0:
            return None
        try:
            # 2. Chọn tập con S_p
            S_p: Population = subset_selector(P, subset_size)
            logging.info(f'Success select {S_p.size} inds from population with size {P.size}')

            # 3. Co‑Evolution: 
            S_r_inds = co_evo_func(S_p.inds)
            S_p.inds = S_r_inds
            logging.info(f'Successfully co-evo with {S_p.size} inds')

            # Tạo compare HDR dùng sau
            compare_hdrs: List[Tuple[Individual, Individual, str]] = []

            # 4. Crossover S_p → P_inter
            P_inter = Population(size=len(S_p.inds), problem=problem)
            inter_inds = []
            while len(inter_inds) < S_p.size:
                if random.random() < pc:
                    selected = subset_selector(S_p, 2)
                    p1, p2 = selected.inds[0], selected.inds[1]
                    off1, off2 = llm_crossover_func(p1=p1, p2=p2)
                    if off1 is None:
                        continue
                    
                    inter_inds.extend([off1, off2])
                    
                    if random.random() < 0.5:
                        compare_hdrs.append((p1, off1, p1.reflection))
                        compare_hdrs.append((p2, off2, p2.reflection))
                    else:
                        compare_hdrs.append((p1, off2, p1.reflection))
                        compare_hdrs.append((p2, off1, p2.reflection))
            inter_inds = inter_inds[:len(S_p.inds)]
            P_inter.inds = inter_inds
            logging.info(f'Successfully crossover with {P_inter.size} inds')
            
            compare_hdrs = compare_hdrs[:len(S_p.inds)]

            # 5. Evaluate P_inter
            for ind in P_inter.inds:
                ind.cal_fitness(fitness_func)
            fe += len(P_inter.inds)
            best = max(P_inter.inds, key=lambda ind: ind.fitness)
            logging.info(f"FE = {fe}, best HDR with fitness {best.fitness:.2f}")
            
            # 6. Self‑Evolution → RM, I_rm
            I_rm_inds = self_evo_func(compare_hdrs=compare_hdrs)
            P_inter.inds = I_rm_inds
            logging.info(f'Successfully self-evo with {len(I_rm_inds)} inds')

            # 7. Crossover 
            # Crossover P_inter với chỉ dẫn từ IRm → P_self
            P_self = Population(size=len(I_rm_inds), problem=problem)
            self_inds = []
            while len(self_inds) < P_inter.size:
                if random.random() < pc:
                    selected = subset_selector(P_inter, 2)
                    p1, p2 = selected.inds[0], selected.inds[1]
                    off1, off2 = llm_crossover_func(p1=p1, p2=p2)
                    if off1 is None:
                        continue
                    self_inds.extend([off1, off2])
            self_inds = self_inds[:P_inter.size]
            P_self.inds = self_inds
            logging.info(f'Successfully crossover with {P_self.size} inds')

            # 8. Evaluate P_self
            for ind in P_self.inds:
                ind.cal_fitness(makespan_fitness_func)
            fe += len(P_self.inds)
            best = max(P_self.inds, key=lambda ind: ind.fitness)
            logging.info(f"FE = {fe}, best HDR with fitness {best.fitness:.2f}")

            # 9. Collective Reflection → MR
            co_refs = [ind.reflection for ind in S_r_inds if ind.reflection is not None]
            self_refs = [ind.reflection for ind in I_rm_inds if ind.reflection is not None]
            MR = collective_func(reflections=co_refs + self_refs)
            logging.info(f'Successfully collective reflections from {len(co_refs) + len(self_refs)} reflections')

            # 10. Mutation guided by MR → P_new
            P_new = Population(size=len(P_self.inds), problem=problem)
            new_inds = []
            muts = 0
            for ind in P_self.inds:
                if random.random() < pm:
                    mutated = llm_mutation_func(p=ind, reflection=MR)
                    if mutated is None:
                        new_inds.append(copy.deepcopy(ind))
                        continue
                    muts += 1
                    new_inds.append(mutated)
                else:
                    new_inds.append(copy.deepcopy(ind))
            P_new.inds = new_inds
            logging.info(f'Successfully mutation {muts} inds of {P_new.size} inds')

            # 11. Evaluate P_new
            for ind in P_new.inds:
                ind.cal_fitness(makespan_fitness_func)
            fe += len(P_new.inds)

            # 12. Cập nhật quần thể với elitism + random replacement
            P = replace_func(old_pop=P, new_pop=P_new, max_size=P.size)
            
            best = max(P.inds, key=lambda ind: ind.fitness)
            best.chromosome.save(f'tmp/best_{num_gen}.py')
            logging.info(f"FE = {fe}, best HDR with fitness {best.fitness:.2f}")
            logging.info(f"Save best HDR to best/best_{num_gen-1}.py")
        
        except LLMException as e:
            logging.error(str(type(e)) + ":" + e.msg)
            continue
        except TypeError as e:
            logging.error(str(type(e)) + ":" + str(e))
            continue
        except HDRException as e:
            logging.error(str(type(e)) + ":" + e.msg)

    # Kết thúc: trả về best individual
    best = max(P.inds, key=lambda ind: ind.fitness)
    
    logging.info(f"Best HDR with fitness {best.fitness:.2f}")
    logging.info("Done!!!")
    return best