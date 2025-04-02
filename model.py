from abc import ABC, abstractmethod
import ast
import re

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
    
