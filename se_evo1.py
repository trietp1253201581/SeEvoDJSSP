from llm import OpenRouterLLM
import llm
import random
from model import HDR, CodeSegmentHDR
from simulate import Simulator
from evolution import Population, Individual

def ind_coevo_reflection(sp: Population, model: OpenRouterLLM):
    # Random choose 2 individual
    hdr1, hdr2 = random.choices(sp.inds, k=2)
    if hdr1.fitness == Individual.DEFAULT_FITNESS:
        hdr1.cal_fitness(problem=sp.problem)
    if hdr2.fitness == Individual.DEFAULT_FITNESS:
        hdr2.cal_fitness(problem=sp.problem)
        
    # Build prompt
    job_str = ", ".join(str(job) for job in sp.problem.jobs)
    machine_str = ", ".join(str(machine) for machine in sp.problem.machines)
    ast_lst = []
    for ind in sp.inds:
        ast_lst.append('{"Expression": "' + str(ind.chromosome.ast) + '"}')
    S_p_str = '[' + ', '.join(str(ast) for ast in ast_lst) + ']'
    func_str = f'[{", ".join(str(func.name) for func in sp.problem.funcs)}]'
    var_str = f'[{", ".join(str(v) for v in sp.problem.vars)}]'
    prompt = llm.CO_EVO_PROMPT.format(
        hdr1_str=hdr1.decode(),
        hdr1_makespan=f'{-hdr1.fitness:.2f}',
        hdr2_str=hdr2.decode(),
        hdr2_makespan=f'{-hdr2.fitness:.2f}',
        job_str=job_str,
        machine_str=machine_str,
        S_p_str=S_p_str,
        func_str=func_str,
        var_str=var_str
    )
    
    # Extract response
    response = model.get_response(prompt)
    print(response)
    json_response = OpenRouterLLM.extract_repsonse(response)
    reflection = json_response.get('Reflection', 'Do nothing, keep they!')
    sr_json = json_response.get('S_r', None)
    if sr_json is None:
        raise Exception()
    
    sr = Population(size=len(sr_json), problem=sp.problem)
    for sr_obj in sr_json:
        expr_str = sr_obj['Expression']
        expr_str = fix_infix_to_prefix(expr_str, in_pre_map={
            '+': 'add',
            '-': 'sub',
            '*': 'mul',
            '/': 'div'
        })
        new_ind = Individual()
        try:
            new_ind.parse_from(expr_str, refer_funcs=sp.problem.funcs, refer_vars=sp.problem.vars)
            new_ind.cal_fitness(sp.problem)
            sr.inds.append(new_ind)
        except Exception:
            print(f"error at parse {expr_str}")
        
    return reflection, sr

def ind_self_evo_reflection(sp: Population, p_inter: Population, co_evo_reflection: str, model: OpenRouterLLM):
    # Build prompt
    job_str = ", ".join(str(job) for job in sp.problem.jobs)
    machine_str = ", ".join(str(machine) for machine in sp.problem.machines)
    ast_lst = []
    for ind in sp.inds:
        ast_lst.append('{"Expression": "' + str(ind.chromosome.ast) 
                       + '", "makespan": ' + str(-ind.fitness) + '}')
    S_p_str = '[' + ', '.join(str(ast) for ast in ast_lst) + ']'
    ast_lst.clear()
    for ind in p_inter.inds:
        ast_lst.append('{"Expression": "' + str(ind.chromosome.ast) 
                       + '", "makespan": ' + str(-ind.fitness) + '}')
    p_inter_str = '[' + ', '.join(str(ast) for ast in ast_lst) + ']'
    func_str = f'[{", ".join(str(func.name) for func in sp.problem.funcs)}]'
    var_str = f'[{", ".join(str(v) for v in sp.problem.vars)}]'
    
    prompt = IndSelfEvoReflectionSupporter.PROMPT.format(
        job_str=job_str,
        machine_str=machine_str,
        hdr_makespan_sp=S_p_str,
        hdr_makespan_pinter=p_inter_str,
        func_str=func_str,
        var_str=var_str,
        co_evo_reflection=co_evo_reflection
    )
    
    response = model.get_response(prompt)
    print(response)
    json_response = OpenRouterLLM.extract_repsonse(response)
    reflections: List[str] = []
    reflections_json = json_response.get('Reflections', None)
    if reflections_json is None:
        raise Exception()
    for obj in reflections_json:
        reflections.append(obj['Reflection'])
    
    l_json = json_response.get('L', None)
    if l_json is None:
        raise Exception()
    
    sr = Population(size=len(l_json), problem=sp.problem)
    for sr_obj in l_json:
        expr_str = sr_obj['Expression']
        expr_str = fix_infix_to_prefix(expr_str, in_pre_map={
            '+': 'add',
            '-': 'sub',
            '*': 'mul',
            '/': 'div'
        })
        new_ind = Individual()
        try:
            new_ind.parse_from(expr_str, refer_funcs=sp.problem.funcs, refer_vars=sp.problem.vars)
            new_ind.cal_fitness(sp.problem)
            sr.inds.append(new_ind)
        except Exception:
            print(f"error at parse {expr_str}")
        
    return reflections, sr

def collective_reflection(problem: Problem, reflections: list[str], model: OpenRouterLLM):
    # Build prompt
    job_str = ", ".join(str(job) for job in problem.jobs)
    machine_str = ", ".join(str(machine) for machine in problem.machines)
    var_str = f'[{", ".join(str(v) for v in problem.vars)}]'
    reflections_str = ""
    for i in range(len(reflections)):
        reflections_str += f"Reflection {i}: {reflections[i]} \n" 
    
    prompt = CollectiveReflectionSupporter.PROMPT.format(
        job_str=job_str,
        machine_str=machine_str,
        var_str=var_str,
        reflections=reflections_str
    )
    
    print(prompt)
    
    response = model.get_response(prompt)
    print(response)
    json_repsonse = OpenRouterLLM.extract_repsonse(response)
    return json_repsonse['Reflection']
   