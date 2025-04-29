from model import CodeSegmentHDR, HDRException
from typing import List, Literal, Tuple
from llm import LLM, LLMException
from basic_evo import Individual, Population, Operator
from abc import abstractmethod
import copy
import random
from problem import Problem
from evaluate import Evaluator, SurrogateEvaluator
import datetime
import logging
import pickle
import time

random.seed(42)

class MissingTemplateException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
        
def get_template(template_file: str):
    try:
        with open(template_file, 'r') as f:
            lines = f.readlines()
            return "".join(lines)
    except FileNotFoundError:
        raise MissingTemplateException("Can't not load template function")

class LLMBaseOperator(Operator):
    def __init__(self, problem, llm_model: LLM, prompt_template: str):
        super().__init__(problem)
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self._logger = logging.getLogger(__name__)
        
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
        response = self.llm_model.get_response(prompt)
        json_repsonse = self.llm_model.extract_response(response)
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
                self._logger.error(str(type(e)) + e.msg, exc_info=True)
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
            pair_str += f"HDR 1 with makespan {-ind1.fitness if ind1.fitness < 0 else 2000 - ind1.fitness}: \n"
            pair_str += str(ind1.chromosome.code)
            pair_str += "\n"
            pair_str += f"HDR 2 with makespan {-ind2.fitness if ind2.fitness < 0 else 2000 - ind2.fitness}: \n"
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
                self._logger.error(str(type(e)) + e.msg, exc_info=True)
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
                self._logger.error(str(e) + ":" + e.msg, exc_info=True)
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
            pair_str += f"HDR before apply reflection with makespan {-ind1.fitness if ind1.fitness < 0 else 2000 - ind1.fitness}: \n"
            pair_str += str(ind1.chromosome.code)
            pair_str += "\n"
            pair_str += f"HDR after apply reflection with makespan {-ind2.fitness if ind2.fitness < 0 else 2000 - ind2.fitness}: \n"
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
                self._logger.error(str(type(e)) + e.msg, exc_info=True)
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
            self._logger.error(str(type(e)) + ":" + e.msg, exc_info=True)
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
            choosen = random.sample(remaining_inds, k=num_random)
        else:
            choosen = []
            
        inds = elites + choosen
        pop = Population(size=max_size, problem=old_pop.problem)
        pop.inds = inds
        return pop

class SelfEvoEngine:
    def __init__(
        self,
        problem,
        llm_init: Operator,
        co_evo: Operator,
        self_evo: Operator,
        collective: Operator,
        crossover: Operator,
        mutation: Operator,
        selector: Operator,
        replacer: Operator,
        fitness_eval: Evaluator,
        max_retries: int = 3
    ):
        self.problem = problem
        self.llm_init = llm_init
        self.co_evo = co_evo
        self.self_evo = self_evo
        self.collective = collective
        self.crossover = crossover
        self.mutation = mutation
        self.selector = selector
        self.replacer = replacer
        self.fitness_eval = fitness_eval
        self.max_retries = max_retries
        self.fe = 0
        self.gen = 0
        self.P = None
        self.best = None
        self.solve_time = 0
        self.log = logging.getLogger(__name__)

    def initialize(self, init_size: int, template: str) -> Population:
        return self._retry(self._do_init, init_size, template)

    def _do_init(self, init_size, template):
        self.log.info(f"Initializing population with {init_size} individuals")
        pop: Population = self.llm_init(init_size=init_size, func_template=template)
        self.log.info(f"Population initialized with {len(pop.inds)} individuals")
        return pop
    
    def evaluate_pop(self, pop: Population):
        return self._retry(lambda: self._evaluate_pop(pop))
    
    def _evaluate_pop(self, pop: Population):
        pop.cal_fitness(self.fitness_eval)
        return pop

    def coevolution(self, S_p: Population) -> List[Individual]:
        return self._retry(lambda: self.co_evo(inds=S_p.inds), )

    def crossover_pop(self, parents: List[Individual], size: int, pc: float) -> List[Individual]:
        def _step():
            off = []
            while len(off) < size:
                if random.random() < pc:
                    p1, p2 = random.sample(parents, 2)
                    c1, c2 = self.crossover(p1=p1, p2=p2)
                    if c1 is None: continue
                    off.extend([c1,c2])
                else:
                    off.extend(random.sample(parents, 2))
            return off[:size]
        return self._retry(_step)

    def selfevolution(self, compare: List[Tuple[Individual, Individual, str]]):
        return self._retry(lambda: self.self_evo(compare_hdrs=compare))

    def collective_reflect(self, refs: List[str]):
        return self._retry(lambda: self.collective(reflections=refs))

    def mutate(self, inds: List[Individual], MR: str, pm: float) -> List[Individual]:
        def _step():
            out=[]
            cnt = 0
            for ind in inds:
                if random.random() < pm:
                    m = self.mutation(p=ind, reflection=MR)
                    if m is not None:
                        out.append(m)
                        cnt += 1
                    else:
                        out.append(copy.deepcopy(ind))
                else:
                    out.append(copy.deepcopy(ind))
            return out, cnt
        return self._retry(_step)

    def _retry(self, fn, *args, **kwargs):
        for attempt in range(1, self.max_retries+1):
            try:
                return fn(*args, **kwargs)
            except (LLMException, HDRException) as e:
                self.log.warning(f"Attempt {attempt}/{self.max_retries} failed in {fn.__name__}: {e.msg}")
            except Exception as e:
                self.log.error(f"Attempt {attempt}/{self.max_retries} failed in {fn.__name__}: {e}")
        raise Exception(f"All {self.max_retries} retries failed for {fn.__name__}")

    def run(
        self,
        num_gen: int,
        init_size: int,
        subset_size: int,
        template_file: str,
        pc: float = 0.8,
        pm: float = 0.1,
        state: str | Literal['new', 'resume'] = 'new',
        checkpoint_path: str|None = None
    ) -> Individual:
        self.log.info(f"Start se_evo at {datetime.datetime.now()}")
        template = get_template(template_file)
        start_time = time.time()
        # 1. Initialize
        if state == 'new':
            try:
                self.P = self.initialize(init_size, template)
                if isinstance(self.fitness_eval, SurrogateEvaluator):
                    self.fitness_eval.set_exact_evaluation(True)
                self.P = self.evaluate_pop(self.P)
            except Exception as e:
                self.log.error(f"Error in initialize: {e}")
                self.solve_time += time.time() - start_time
                return None
            self.fe = len(self.P.inds)
            self.best = max(self.P.inds, key=lambda i: i.fitness)
            self.log.info(f"Init FE={self.fe}, best={self.best.fitness:.2f}")
            self.gen = 1
            
        else:
            self.load_state(checkpoint_path, fields_to_update=['P', 'best', 'fe', 'gen', 'solve_time'])
            self.log.info(f"Resumed from checkpoint at gen {self.gen}, FE={self.fe}, best={self.best.fitness:.2f}, num_inds={len(self.P.inds)}, solve time={self.solve_time:.2f}")
            
        if isinstance(self.fitness_eval, SurrogateEvaluator):
            self.fitness_eval.set_exact_evaluation(False)
        while self.gen <= num_gen:
            try:
                self.log.info(f"Gen {self.gen}")
                # 2. Selection
                S_p = self.selector(self.P, subset_size)
                self.log.info(f"Selected {len(S_p.inds)} individuals")

                # 3. Co-evolution
                S_r = self.coevolution(S_p)
                self.log.info(f"Co-evolution done with {len(S_r)} individuals")

                # 4. Crossover → P_inter
                inter = self.crossover_pop(S_r, len(S_p.inds), pc)
                P_inter = Population(size=len(inter), problem=self.problem)
                P_inter.inds = inter
                self.log.info(f"Crossover done with {len(P_inter.inds)} individuals")

                # 5. Evaluate P_inter
                P_inter = self.evaluate_pop(P_inter)
                self.fe += len(P_inter.inds)
                self.log.info(f"After P_inter FE={self.fe}")
                #if self.fe > max_fe:
                #    self.log.info("Reached max FE, return best individua in " + f"tmp/best_{self.fe}.py")
                #    self.solve_time += time.time() - start_time
                #    return self.best

                # Update best so far
                self.best = max(P_inter.inds, key=lambda i: i.fitness)

                # 6. Self-evolution
                # You must build compare list beforehand
                compare_list = [(p, c, p.reflection) for p,c in zip(S_p.inds, P_inter.inds)]
                I_rm = self.selfevolution(compare_list)
                self.log.info(f"Self-evolution done with {len(I_rm)} individuals")

                # 7. Crossover self → P_self
                P_self = Population(size=len(I_rm), problem=self.problem)
                P_self.inds = self.crossover_pop(I_rm, len(I_rm), pc)
                self.log.info(f"Crossover self done with {len(P_self.inds)} individuals")
                # 8. Evaluate P_self
                P_self = self.evaluate_pop(P_self)
                self.fe += len(P_self.inds)
                self.log.info(f"After P_self FE={self.fe}")
                #if self.fe > max_fe:
                #    self.log.info("Reached max FE, return best individua in " + f"tmp/best_{self.fe}.py")
                #    self.solve_time += time.time() - start_time
                #    return self.best

                # 9. Collective reflection
                co_refs = [i.reflection for i in S_r if i.reflection]
                self_refs = [i.reflection for i in I_rm if i.reflection]
                MR = self.collective_reflect(co_refs + self_refs)
                self.log.info(f"Collective reflection done with {len(co_refs + self_refs)} reflections")

                # 10. Mutation → P_new
                mutated, cnt = self.mutate(P_self.inds, MR, pm)
                P_new = Population(size=len(mutated), problem=self.problem)
                P_new.inds = mutated
                self.log.info(f"Mutation done with {len(P_new.inds)} individuals, {cnt} individuals mutated")
                
                # 11. Evaluate P_new
                P_new = self.evaluate_pop(P_new)
                self.fe += len(P_new.inds)
                self.log.info(f"After P_new FE={self.fe}")

                # 12. Replacement
                self.P = self.replacer(old_pop=self.P, new_pop=P_new, max_size=self.P.size)

                self.best = max(self.P.inds, key=lambda i: i.fitness)
                self.log.info(f"Gen {self.gen} done FE={self.fe}, best={self.best.fitness:.2f}")

                self.gen += 1
            except Exception as e:
                self.log.error(f"Error in gen {self.gen}: {e}")
                continue

        self.log.info(f"Done, best overall fitness {self.best.fitness:.2f}")
        self.solve_time += time.time() - start_time
        return self.best
    
    def save_state(self, checkpoint_path: str, fields_to_save: list|None = None):
        with open(checkpoint_path, 'wb') as f:
            if fields_to_save is None:
                pickle.dump(self, f)
            else:
                data = {field: getattr(self, field) for field in fields_to_save if hasattr(self, field)}
                pickle.dump(data, f)
            
    def load_state(self, checkpoint_path: str, fields_to_update: list | None = None):
        with open(checkpoint_path, 'rb') as f:
            loaded = pickle.load(f)
    
            if isinstance(loaded, dict):
                # Nếu file chứa dict, thì lấy từ dict
                if fields_to_update is None:
                    for field, value in loaded.items():
                        setattr(self, field, value)
                else:
                    for field in fields_to_update:
                        if field in loaded:
                            setattr(self, field, loaded[field])
            else:
                # Nếu file chứa nguyên object
                if fields_to_update is None:
                    self.__dict__.update(loaded.__dict__)
                else:
                    for field in fields_to_update:
                        if hasattr(loaded, field):
                            setattr(self, field, getattr(loaded, field))