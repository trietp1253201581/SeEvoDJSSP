from model import HDR
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
        return str(self.chromosome)
        
    def cal_fitness(self, fitness_func: FITNESS_FUNC_TYPE):
        if self.chromosome is None:
            self.fitness = Individual.DEFAULT_FITNESS
        else:
            self.fitness = fitness_func(self.chromosome, self.problem)
            
            
class Population:
    def __init__(self, size: int, problem: Problem):
        self.size = size
        self.problem = problem
        self.inds: List[Individual] = []

        
    

    