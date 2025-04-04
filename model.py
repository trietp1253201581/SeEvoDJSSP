from abc import ABC, abstractmethod
import ast
import re
from typing import Dict

class Terminal:
    def __init__(self, label: str, description: str=""):
        self.label = label
        self.description = description
        
    def __str__(self):
        return f'{self.label}({self.description})'
    
    def __eq__(self, value):
        if not isinstance(value, Terminal):
            return False
        return self.label == value.label
    
JNPT = Terminal('jnpt', "Next operation processing time of a job")
JTPT = Terminal('japt', "Total process time of job")
JRT = Terminal('jrt', "Remaining time to complete job")
JRO = Terminal('jro', "Remaining operations to complete job")
JWT = Terminal('jwt', "Job waiting time in waiting pool and job pool")
JAT = Terminal('jat', "Arriving time of a job")
JD = Terminal('jd', "Job Deadline")
JCD = Terminal('jcd', "Deadline of next operation of a job")
JS = Terminal('js', "Slack time of a job")
JW = Terminal('jw', "Weight of job")
ML = Terminal('ml', "Time from process current job")
MR = Terminal('mr', "Remaining time to completed current job")
MREL = Terminal('mrel', "Relaxing Time of machine")
MPR = Terminal('mpr', "Num of processed opr in this machine")
MUTIL = Terminal('mutil', "Now utilization")
TNOW = Terminal('tnow', "Current time of system")
UTIL = Terminal('util', "Now utilization of system")
AVGWT = Terminal('avgwt', "Average wait time of all jobs")

class TerminalDictMaker:
    def __init__(self):
        self.var_dicts: Dict[str, any] = {}
        
    def add_terminal(self, new_terminal: Terminal, new_value: int|float):
        self.var_dicts[new_terminal] = new_value
        

class HDR(ABC):
    @abstractmethod
    def execute(self, **kwargs) -> any:
        pass
    
class InvalidHDRException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
        
class InvalidKwargsException(Exception):
    def __init__(self, msg: str = "Invalid kwargs for function!"):
        super().__init__()
        self.msg = msg
    
class CodeSegmentHDR(HDR):
    def __init__(self, code: str|None = None):
        self.code = code
        self._extract_func(code)
        
    def _extract_func(self, code: str|None):
        if code is None:
            self.func_name = None
            self.params = None
            return
        m = re.search(r'def (?P<func_name>[a-zA-Z0-9_]+)\((?P<params>.*)\)\:', code)
        if m is None:
            raise InvalidHDRException('Missed function name')
        self.func_name = m.group('func_name')
        params_extracted = m.group('params').replace(' ', '').split(',')
        self.params = []
        for param in params_extracted:
            m = re.search(r'(?P<var>[a-zA-Z0-9_]+)(:(?P<type>[a-zA-Z0-9_]*))?', param)
            if m is None:
                raise InvalidHDRException(f'Invalid parameter: {param}')
            self.params.append({'name': m.group('var'), 
                                'type': m.group('type') if m.group('type') is not None else 'any'})
        
    def __str__(self):
        return str(self.code)
        
    def to_ast(self):
        return ast.parse(source=self.code)
    
    def from_ast(self, ast_rule: ast.AST):
        self.code = ast.unparse(ast_rule)
        self._extract_func(self.code)
        
    def execute(self, **kwargs):
        local_vars = {}
        exec(self.code, globals(), local_vars)
        func = local_vars['hdr']
        
        try:
            return func(**kwargs)
        except TypeError as e:
            raise InvalidKwargsException(f"Invalid kwargs for {self.func_name} " + str(e))
    
