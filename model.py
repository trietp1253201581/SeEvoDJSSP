from abc import ABC, abstractmethod
import ast
import re
import hashlib
class HDR(ABC):
    """
    Abstract class for Heuristic Dispatching Rule (HDR)
    """
    @abstractmethod
    def execute(self, **kwargs) -> any:
        """
        Execute the HDR with the given kwargs
        """
        pass
    
    @abstractmethod
    def hash_code(self) -> str:
        """
        Hash the code of the HDR
        """
        pass
    
class HDRException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
    
class InvalidHDRException(HDRException):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg
        
class InvalidKwargsException(HDRException):
    def __init__(self, msg: str = "Invalid kwargs for function!"):
        super().__init__(msg)
        self.msg = msg
    
class CodeSegmentHDR(HDR):
    def __init__(self, code: str|None = None):
        self.code = code
        
    @property
    def code(self):
        return self._code
    
    @code.setter
    def code(self, code: str|None):
        self._code = code
        if code is None: 
            return
        self._extract_func(code)
        res = self.execute(jnpt=1.0, japt=2.5, jrt=3.6, jro=4.1, jwt=6.7, jat=6.0, jd=8.0, jcd=3.0, js=2.0, jw=1.0, ml=89.0, mr=20.0, mrel=20.0, mpr=20.1, mutil=3.9, tnow=15, util=60, avgwt=20.0)
        res = self.execute(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0)
        if not isinstance(res, (int, float)):
            raise HDRException('Not support return type')
        
    def save(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self._code.strip() + "\n")
            
    def load(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.code = f.read()
                 
    def _extract_func(self, code: str|None):
        if code is None:
            self.func_name = None
            self.params = None
            return
        m = re.search(r'def (?P<func_name>[a-zA-Z0-9_]+)\((?P<params>.*)\)\:', code)
        if m is None:
            raise InvalidHDRException('Invalid function')
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
        return str(self._code)
        
    def to_ast(self):
        return ast.parse(source=self._code)
    
    def from_ast(self, ast_rule: ast.AST):
        self.code = ast.unparse(ast_rule)
        
    def execute(self, **kwargs):
        local_vars = {}     
        try:
            exec(self._code, globals(), local_vars)
            func = local_vars['hdr']
            return func(**kwargs)
        except UnboundLocalError as e:
            raise HDRException(str(e))
        except TypeError as e:
            raise InvalidKwargsException(f"Invalid kwargs " + str(e))
        except SyntaxError as e:
            raise InvalidKwargsException(f"Invalid kwargs " + str(e))
        except NameError as e:
            raise InvalidKwargsException(f"Invalid kwargs " + str(e))
        except ZeroDivisionError as e:
            raise HDRException(str(e))
        except ArithmeticError as e:
            raise HDRException(str(e))
        except ValueError as e:
            raise HDRException(str(e))
        
    def hash_code(self):
        return hashlib.sha256(self._code.strip().encode('utf-8')).hexdigest()
        
        
    
