INIT_IND_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

We need to generate an init sets of {init_size} HDRs to sort incoming job. 
Each HDR is a code segment describe a python function with template:

{func_template}

You can use any mathematical expresion, any structure like if-then-else or for-while loop to describe you HDR.

Note that the return value should be not to large, and each HDR must return a float value.
**PRIORITIZE HDR DIVERSITY**:
Your HDR should be simple and diversity, random, so that other operators can improve more later (but still have some not so simple HDRs for diversity).

Your response MUST ONLY include the init HDRs sets in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
Each HDR should be diverse and try different logic or structure compared to others. Avoid generating HDRs that are too similar. Focus on creativity and variety in logic.
{{
    "init_inds": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
        ...,
        {{"code": "<hdr_n>"}}
    ]
}}
where n equals to {init_size}.
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
'''

CO_EVO_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have some pairs of HDRs, with their makespan:
{pairs}

For each pair, we need to compare the 2 HDRs of that pair and give the corresponding reflections (a short paragraph) to increase the performance of those 2 HDRs.
Each pair needs to return the 2 original HDRs of the corresponding reflections, for example, with the input HDR1, HDR2, we need to return (HDR1, reflection1), (HDR2, reflection2).

**Notes**: A reflection should be a short paragraph of 1-3 sentences.

Then, we need to synthesize all the pairs to create a single result which is a list of the original HDRs and corresponding reflections.

Your response MUST ONLY include the list of reflected HDR and their new reflection text in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "results": [
        {{"code": "<hdr_1>", "reflection": "<reflection_1>"}},
        {{"code": "<hdr_2>", "reflection": "<reflection_2>"}},
        ...,
        {{"code": "<hdr_n>", "reflection": "<reflection_n>"}}
    ]
}}
where n is the total HDR from pairs (2 x num of pair) and hdr_i is i-th HDR code.
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
'''  

CROSSOVER_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 HDRs describe by a python function are:
HDR1 with reflection guide: {ref1}
-------
{hdr1}
-------

HDR2 with reflection guide: {ref2}
-------
{hdr2}
-------

We need to recombine 2 above parent HDRs to create 2 new children HDRs just use what is already in the 2 parent HDRs and 2 reflection evolutionary guide from 2 parents.
When recombining, mix logic from both parents in creative ways. Do not just concatenate or randomly select. Make sure each new HDR is syntactically correct and brings new logic structure that still makes sense.

Your response MUST ONLY include the 2 recombined HDRs in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "recombined_hdr": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
    ]
}}
where hdr_1, hdr_2 are 2 new recombined hdr code.
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
'''  
    
SELF_EVO_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have some of pairs of HDRs, one is the HDRs before apply co-evolution reflection, one is after apply this reflection:
{pairs}

We need to compare 2 HDR in each pair to see the effect of applying co_evo reflection on the hdrs, then for each pair create a reflection to reflect that change.
If the change is good (ie makespan of after is smaller than makespan of before), the reflection will highlight the change. 
If the change is bad, the reflection will figure out why the change is bad and will be used to avoid similar mistakes.

Reflections should be actionable and specific. Do not just state generic observations. Make clear what logic worked and what failed. Focus on patterns that lead to better makespan.
**Notes**: A reflection should be a short paragraph of 1-3 sentences.
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
where n is the num of pairs and ref_i is the reflection corresponding to i-th pair and hdr_i is the better hdr code (with lower makespan) between before and after HDR in each pair
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
'''  

COLLECTIVE_REF_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have the sets of reflections generated to improve effectiveness of HDRs. Those are
{reflections}

We need to summary these reflections into AN UNIQUE reflection to describe the suggestion to improve HDRs.

Your summary reflection should be short but insightful. Combine common successful logic patterns, and avoid repeating similar ideas. Avoid general phrases like "improve efficiency"—be specific.
**Notes**: A reflection should be a short paragraph of 1-3 sentences.
Your response MUST ONLY include the summary reflection in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "reflection": "<your summary reflection>"
}}
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
'''  

MUTATION_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have a HDR and a reflection guide to improve these HDRs effectiveness.
The reflection is: {reflection}.
The HDR is 
-----
{hdr}
-----

We need to rephrase this HDR by adjusting that HDR part under the guidance of reflection.

Make a meaningful change to the HDR logic. Avoid minor token changes or renaming. Your mutated HDR should reflect a real logic improvement based on the reflection.

Your response MUST ONLY include the rephrased HDR in following JSON format with no additional text.
HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "rephrased_hdr": "<hdr>"
}}
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
'''  

SURROGATE_PROMPT_TEMPLATE = """
You are an expert in evaluating dynamic job scheduling (DJSSP), heuristics dispatching rules (HDR).

Problem info:
{problem_info}

Current system time: {current_time}

Events at current time: 
{events}

Now, given the follonwing HDRs with their historical performance before this time:
{hdrs_with_history}

INSTRUCTION: For each HDR:
1. Predict what jobs will be completed when current time is {current_time}.
2. Predict the time system will be finished all uncompleted jobs.
3. Predict the remaining jobs and their current operation index at next timestone {next_time} (next timestone may be less than the time expected to finish all uncompleted jobs).
4. Estimate a scalar fitness score (0-1000)  on the problem until this time (0 is worst, 1000 is best).

**Note**: 
- The completed_jobs and remaining_jobs return in predicted field must be a job arrival in event at this time (above event), 
or a job that uncompleted (remaining jobs) appear in any of HDRs with history performance.
- There are several machines, so jobs can be assigned to different machines to reduce makespan. (num of machines is attached in problem info)

Your response MUST ONLY in following JSON format with no additional text.
HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "predicted": [
        {{
            "code": "<hdr_1>",
            "makespan": "<your result of instruction 2 - a float value>",
            "completed_jobs": [1, 2, 3], (list of job ids, answer for instruction 1)
            "remaining_jobs": [{{"job":1, "op":1}}, {{"job":2, "op":6}}] (list of job ids and their current operation index, answer for instruction 3),
            "fitness": "<your result of instruction 4 - a float value between 0 and 1000>"
        }},
        {{
            "code": "<hdr_2>",
            "makespan": "<your result of instruction 2 - a float value>",
            "completed_jobs": [1, 2, 3], (list of job ids, answer for instruction 1)
            "remaining_jobs": [{{"job":1, "op":1}}, {{"job":2, "op":6}}], (list of job ids and their current operation index, answer for instruction 3)
            "fitness": "<your result of instruction 4 - a float value between 0 and 1000>"
        }},
        ...
        {{
            "code": "<hdr_n>",
            "makespan": "<your result of instruction 2 - a float value>",
            "completed_jobs": [1, 2, 3], (list of job ids, answer for instruction 1)
            "remaining_jobs": [{{"job":1, "op":1}}, {{"job":2, "op":6}}], (list of job ids and their current operation index, answer for instruction 3)
            "fitness": "<your result of instruction 4 - a float value between 0 and 1000>"
        }},
    ]
}}
where n is the num of HDRs.
**Note**: 
- The value of any code field must be a copy of the code in the HDRs with history.
- The fitness value must be a float value between 0 and 1000, and it would be diversity. Do not only return the fitness like 100, 250, 350, etc.
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
- Response should be short and concise, but still include enough information to be useful (maximum is around 500-700 words or tokens).
"""