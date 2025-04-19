from model import HDR
from problem import Problem
from typing import List
from abc import ABC
from evaluate import Evaluator

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
        self.reflection = None
        
    def decode(self):
        return self.chromosome
        
    def cal_fitness(self, fitness_evaluator: Evaluator):
        sol = self.decode()
        if sol is None:
            self.fitness = Individual.DEFAULT_FITNESS
        else:
            self.fitness = fitness_evaluator([sol])[0][1]
            
            
class Population:
    def __init__(self, size: int, problem: Problem):
        self.size = size
        self.problem = problem
        self.inds: List[Individual] = []
        
    def cal_fitness(self, fitness_evaluator: Evaluator):
        hdrs = [ind.decode() for ind in self.inds]
        results = fitness_evaluator(hdrs)
        self.inds.clear()
        self.size = len(results)
        for i in range(self.size):
            new_ind = Individual(self.problem)
            new_ind.chromosome = results[i][0]
            new_ind.fitness = results[i][1]
            self.inds.append(new_ind)
            