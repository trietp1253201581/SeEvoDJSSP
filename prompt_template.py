INIT_IND_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

We need to generate an init sets of {init_size} HDRs to sort incoming job. Each HDR is a code segment describe a python function with template:

{func_template}

You can use a single return expression like:

def hdr(...):
    return ...

Or, use any control structrue like if-then-else or for loop like:

def hdr(...):
    if js > 5:
        return jw
    return jcd

. Note that the return value should be not to large.

Your response MUST ONLY include the init HDRs sets in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "init_inds": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
        ...,
        {{"code": "<hdr_n>"}}
    ]
}}
where n equals to {init_size}.
'''

CO_EVO_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 HDRs describe by a python function (with its makespan) are:
HDR1 with makespan {hdr1_makespan}
-------
{hdr1}
-------

HDR2 with makespan {hdr2_makespan}
-------
{hdr2}
-------

We need to compare these 2 examples and create a reflection to improve the other HDRs to make them more effective.
Reflection is a text created from the 2 examples above, when comparing their effectiveness.
Note that reflection should not contain information about comparing 2 HDRs, but only information about how to improve. That means you will still do the comparison but not add it to the reflection.

After that, we need to apply this reflection to each HDR in a set, to improve HDR effectiveness. The HDRs set is
{hdr_set}

Your response MUST ONLY include the reflection and the list of reflected HDR in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "reflection": "<your_reflection_text>",
    "reflected_hdr": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
        ...,
        {{"code": "<hdr_n>"}}
    ]
}}
where n equals to the size of hdr set and where hdr_i is the i-th element in the hdr set after apply your reflection.
'''  

CROSSOVER_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 HDRs describe by a python function are:
HDR1
-------
{hdr1}
-------

HDR2
-------
{hdr2}
-------

We need to recombine 2 above parent HDRs to create 2 new children HDRs just use what is already in the 2 parent HDRs.

Your response MUST ONLY include the 2 recombined HDRs in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "recombined_hdr": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
    ]
}}
where hdr_1, hdr_2 are 2 new recombined hdr.
'''  
    
SELF_EVO_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 set of HDRs, one is the HDRs before apply co-evolution reflection, one is after apply this reflection:
The co evo reflection: {co_evo_reflection}.
*****
hdr_before:
{hdr_before}

*****
hdr_after
{hdr_after}

We need to compare each hdr_before[i] with hdr_after[i] to see the effect of applying co_evo reflection on the hdrs, then for each pair of hdr_before[i] and hdr_after[i], create a reflection to reflect that change.
If the change is good (ie makespan of hdr_after[i] is smaller than makespan of hdr_before[i]), the reflection will highlight the change. 
If the change is bad, the reflection will figure out why the change is bad and will be used to avoid similar mistakes.

Your response MUST ONLY include the reflected HDR with reflection in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "reflected_hdr": [
        {{"code": "<hdr_1>", "reflection": "<ref_1>"}},
        {{"code": "<hdr_2>", "reflection": "<ref_2>"}},
        ...,
        {{"code": "<hdr_n>", "reflection": "<ref_n>"}}
    ]
}}
where ref_i is the reflection corresponding to pair(hdr_before[i],hdr_after[i]) and hdr_i is the better hdr (with lower makespan) between hdr_before[i] and hdr_after[i]
'''  

COLLECTIVE_REF_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

In this problem, job (with its operation) is {job_str}, machine is {machine_str}. 
Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have the sets of reflections generated to improve effectiveness of HDRs. Those are
{reflections}

We need to summary these reflections into AN UNIQUE reflection to describe the suggestion to improve HDRs.

Your response MUST ONLY include the summary reflection in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "reflection": "<your summary reflection>"
}}
'''  