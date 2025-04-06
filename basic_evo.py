from model import HDR, InvalidKwargsException
from problem import Problem
from typing import Callable, List
from abc import ABC

FITNESS_FUNC_TYPE = Callable[[HDR, Problem], float]

class Operator(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
    
    def __call__(self, *args, **kwds):
        pass
     
class Individual:
    DEFAULT_FITNESS: float = -1e8
    def __init__(self, problem: Problem):
        self.chromosome: HDR = None
        self.fitness = Individual.DEFAULT_FITNESS
        self.problem = problem
        
    def decode(self):
        return self.chromosome
        
    def cal_fitness(self, fitness_func: FITNESS_FUNC_TYPE):
        sol = self.decode()
        if sol is None:
            self.fitness = Individual.DEFAULT_FITNESS
        else:
            self.fitness = fitness_func(sol, self.problem)
            
            
class Population:
    def __init__(self, size: int, problem: Problem):
        self.size = size
        self.problem = problem
        self.inds: List[Individual] = []
        
def validate(pop: Population) -> Population:
    valid_inds = []
    for ind in pop.inds:
        hdr = ind.chromosome
        
        if hdr is None:
            continue
        if not hdr.is_valid():
            continue
        if not hdr.is_evaluatable():
            continue
        
        valid_inds.append(ind)
        
    new_pop = Population(size=len(valid_inds), problem=pop.problem)
    new_pop.inds = valid_inds
    return new_pop