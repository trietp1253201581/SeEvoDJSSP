from model import HDR, CodeSegmentHDR, Terminal
from problem import Problem
from typing import Callable, List
import ast
import random
from abc import ABC, abstractmethod

FITNESS_FUNC_TYPE = Callable[[HDR, Problem], float]

class Operator(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
        
    @abstractmethod
    def operate(self, **kwargs):
        pass
    
    def __call__(self, **kwargs):
        return self.operate(**kwargs)
        
class InitOperator(Operator):
    def __init__(self, problem, terminals: List[Terminal]):
        super().__init__(problem)
        self.terminals = terminals
        
    @abstractmethod
    def operate(self, init_size: int) -> List[HDR]:
        pass
        
    def __call__(self, init_size: int):
        return self.operate(init_size)
    
class SingleInitOperator(InitOperator):
    def __init__(self, problem, terminals):
        super().__init__(problem, terminals)
        
    @abstractmethod
    def operate(self) -> HDR:
        pass
    
    def __call__(self):
        return self.operate()


class Individual:
    DEFAULT_FITNESS: float = -1e8
    def __init__(self, problem: Problem):
        self.chromosome: HDR = None
        self.fitness = Individual.DEFAULT_FITNESS
        self.problem = problem
        
    def generate(self, init_func: SingleInitOperator):
        self.chromosome = init_func()
        
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
        
    def _generate_each(self, init_func: SingleInitOperator,
                       fitness_func: FITNESS_FUNC_TYPE):
        self.inds.clear()
        for _ in range(self.size):
            new_ind = Individual(self.problem)
            new_ind.generate(init_func)
            new_ind.cal_fitness(fitness_func)
            self.inds.append(new_ind)
            
    def _generate_all(self, init_func: InitOperator,
                      fitness_func: FITNESS_FUNC_TYPE):
        self.inds.clear()
        hdrs = init_func(self.size)
        
        for hdr in range(hdrs):
            new_ind = Individual(self.problem)
            new_ind.chromosome = hdr
            new_ind.cal_fitness(fitness_func)
            self.inds.append(new_ind)
        
    def generate(self, init_func: InitOperator, 
                 fitness_func: FITNESS_FUNC_TYPE):
        if isinstance(init_func, SingleInitOperator):
            self._generate_each(init_func, fitness_func)
        else:
            self._generate_all(init_func, fitness_func)
            
class CrossoverOperator(Operator):
    def __init__(self, problem):
        super().__init__(problem)
        
    @abstractmethod
    def operate(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        pass
    
    def __call__(self, p1: Individual, p2: Individual):
        return self.operate(p1, p2)
    

    