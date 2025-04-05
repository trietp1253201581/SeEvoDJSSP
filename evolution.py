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
    def operate(self):
        pass
    
    def __call__(self, *args, **kwds):
        return self.operate()
        
class InitOperator(Operator):
    def __init__(self, problem, terminals: List[Terminal], init_size: int):
        super().__init__(problem)
        self.terminals = terminals
        self.init_size = init_size
        
    @abstractmethod
    def operate(self) -> List[HDR]:
        pass
        
    def __call__(self):
        return self.operate()
    
class SingleInitOperator(InitOperator):
    def __init__(self, problem, terminals):
        super().__init__(problem, terminals, 1)
        
    @abstractmethod
    def operate(self) -> HDR:
        pass


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
        init_func.init_size = self.size
        hdrs = init_func()
        
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
    
def _get_random_subtree_ast(node):
    candidates = []
    
    def visit(n):
        accessed = [ast.expr, ast.Call, ast.If, ast.For, ast.While, ast.Constant, ast.Name]
        random.shuffle(accessed)
        if isinstance(n, tuple(accessed)):
            candidates.append(n)
        for child in ast.iter_child_nodes(n):
            visit(child)
    
    visit(node)
    return random.choice(candidates) if candidates else None

def _replace_subtree(parent, old_child, new_child):
    """Thay thế một node con trong AST."""
    for field, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is old_child:
                    value[i] = new_child
                    return True
        elif value is old_child:
            setattr(parent, field, new_child)
            return True
    return False

def crossover_ast(p1: Individual, p2: Individual):
    chro1: CodeSegmentHDR = p1.chromosome
    chro2: CodeSegmentHDR = p2.chromosome
    
    ast1 = chro1.to_ast()
    ast2 = chro2.to_ast()
    
    node1 = _get_random_subtree_ast(ast1)
    node2 = _get_random_subtree_ast(ast2)
    
    if not node1 or not node2:
        return None
    
    # Tìm parent của node trong cây AST
    def find_parent(root, target):
        for node in ast.walk(root):
            for child in ast.iter_child_nodes(node):
                if child is target:
                    return node
        return None
    
    parent1 = find_parent(ast1, node1)
    parent2 = find_parent(ast2, node2)
    
    if parent1 and parent2:
        _replace_subtree(parent1, node1, node2)
        _replace_subtree(parent2, node2, node1)
        
    
    c1 = Individual(p1.problem)
    c2 = Individual(p2.problem)
    
    newhdr1 = CodeSegmentHDR(code=None)
    newhdr1.from_ast(ast1)
    c1.chromosome = newhdr1
    
    newhdr2 = CodeSegmentHDR(code=None)
    newhdr2.from_ast(ast2)
    c2.chromosome = newhdr2
    
    return c1, c2

def mutation_ast(p: Individual):
    chro: CodeSegmentHDR = p.chromosome
    tree = chro.to_ast()
    node = _get_random_subtree_ast(tree)
    if node is None:
        return None
    
    if isinstance(node, ast.BinOp):
        node.op = random.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            node.value += random.randint(1, 10)
    elif isinstance(node, ast.Name):
        node.id = random.choice([param['name'] for param in chro.params])
    
    newhdr = CodeSegmentHDR(code=None)
    newhdr.from_ast(tree)
    c = Individual(p.problem)
    c.chromosome = newhdr
    
    return c
    