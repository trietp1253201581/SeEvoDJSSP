from typing import Literal
import requests
import json
import re

class BadAPIException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
        
class BadResponseException(Exception):
    def __init__(self, msg: str = "Bad response in invalid structure"):
        super().__init__()
        self.msg = msg

class OpenRouterLLM:
    def __init__(self, model: str|Literal['deepseek-v3-0324', 'deepseek-r1-zero',
                                          'gemini-2.5-pro-exp']):
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.key = "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"
        
    def get_model(self):
        if self.model == 'deepseek-v3-0324':
            return 'deepseek/deepseek-chat-v3-0324:free'
        if self.model == 'deepseek-r1-zero':
            return 'deepseek/deepseek-r1-zero:free'
        if self.model == 'gemini-2.5-pro-exp':
            return 'google/gemini-2.5-pro-exp-03-25:free'
        return self.model
    
    def get_response(self, prompt: str, timeout: float = 100.0):
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            'model': self.get_model(),
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        })
        
        try:
            response = requests.post(url=self.url, headers=headers, data=data, timeout=timeout)
            response.raise_for_status()  # Kiểm tra lỗi HTTP (4xx, 5xx)
            return response.json()['choices'][0]['message']['content']
        except requests.Timeout:
            raise BadAPIException("Timeout Request!")
        except requests.RequestException as e:
            raise BadAPIException(str(e))
        
    @staticmethod
    def extract_repsonse(response: str) -> dict:
        m = re.search(r'(json)?(?P<obj>[^\`]+)', response)
        if m is not None:
            json_str = m.group('obj')
            
            json_obj = json.loads(json_str)
            
            return json_obj
        else:
            raise BadResponseException()
    
INIT_PROMPT = """
"""    

    
SELF_EVO_PROMPT = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}.

We need to evaluate how HDRs in S_p (old HDRs subset) perform compared to their new HDR in P_inter created by the reflection {co_evo_reflection}.

We have the HDR with their makespan in S_p: {hdr_makespan_sp}.
And the HDR with their makespan in P_inter: {hdr_makespan_pinter}.

Note that for each old HDR S_p[i], the new HDR corresponding to it is P_inter[i].

Base on two set of HDRs and their makespan, provide a list of reflections to evaluate performance of each old and new HDR:
- If P_inter[i] have lower makespan (better perfomance) then generate a positive reflection that describes the beneficial transformation from S_p[i] to P_inter[i].
- Else if P_inter[i] have higher makespan (worse performance) then generate a reverse reflection highlighting the mistake to avoid the similar transformation in future HDRs.

The reflection should be a concise piece of text that can be directly applied to improve other HDRs by calling the API (or a similar mechanism). 
Good reflection has based on reference vars and their discription to modified HDR.

The response of reflections contains n reflection corresponding to n S_p[i] and P_inter[i], like:
[{{"Reflection": "Rep1"}}, {{"Reflection": "Rep2"}}]

After generating the reflection, apply it to each individual in S_p and P_inter to generate new set of HDRs called L, ensuring that:  
- At any position i, L[i] is the one that has better perfomance in S_p[i] and P_inter[i].
    For example, if makespan(S_p[i])=50, makespan(P_inter[i])=100 then L[i] = S_p[i].
The response of L like:
[{{"Expression": "add(ta, p)"}}, {{"Expression": "mul(ta, p)"}}]

Note: You must use pre-expression in any position, i.e. use add(p, 0.5) instead of p + 0.5, use mul(0.5, d) instead of 0.5*d
The reference functions you can use in {func_str}, and the variables you can use in {var_str} (with its mean).

Your response MUST ONLY include the reflection text in following JSON format with no additional commentary and no special character:
{{"Reflections": [<list_of_reflection_text_in_json_with_Reflection_key>],
  "L": [<list_of_updated_expressions>]}}.'''

CO_EVO_PROMPT = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}.

Two HDR have to compare are:
1. HDR1: {hdr1_str} with makespan {hdr1_makespan}.
2. HDR2: {hdr2_str} with makespan {hdr2_makespan}.

Based on the 2 HDR examples and their makespan, provide a single reflection to improve performance of them.
The reflection should be a concise piece of text that can be directly applied to improve other HDRs by calling the API (or a similar mechanism). 
Good reflection has based on reference vars and their discription to modified HDR.

After generating the reflection, apply it to each individual in {S_p_str}, ensuring that:  
- The transformation follows the logic of the reflection.  
- The order of individuals in the updated set S_r MUST be the same as in S_p, i.e., S_r[i] corresponds to S_p[i].  
The response of S_r like:
[{{"Expression": "add(ta, p)"}}, {{"Expression": "mul(ta, p)"}}]

Note: You must use pre-expression in any position, i.e. use add(p, 0.5) instead of p + 0.5, use mul(0.5, d) instead of 0.5*d
The reference functions you can use in {func_str}, and the variables you can use in {var_str} (with its mean).

Your response MUST ONLY include the reflection text in following JSON format with no additional commentary and no special character:
{{"Reflection": <your_reflection_text>,
  "S_r": [<list_of_updated_expressions>]}}.'''

COLLECTIVE_PROMPT = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}.
The variables you can use in {var_str} (with its mean).

I have received list of reflections, are: \n
{reflections}.

Based on the above individual reflections, provide a single consolidated reflection that captures the collective insight.

Your response MUST ONLY include the reflection text in following JSON format with no additional commentary and no special character:
{{"Reflection": <your_collective_reflection_text>]}}.'''
    

