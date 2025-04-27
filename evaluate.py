import math
import random
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import LLM, LLMException
from model import HDR, CodeSegmentHDR, HDRException
import problem
from problem import Problem, Job, Machine, TerminalDictMaker
import copy
from typing import List, Tuple, Literal
import time
from abc import ABC, abstractmethod
import logging
import collections
        
class Simulator:
    DEFAULT_PRIOR: int = -1e9
    
    def __init__(self, problem: Problem):
        self.problem = problem
        self.waiting_pool: List[Job] = []
        self.job_pool: List[Job] = []
        self.pool_size = self.problem.pool_size
        self._logger = logging.getLogger(__name__)
        
    def _print_with_debug(self, msg: str, debug: bool=False):
        if debug:
            print(msg)

    def _reset(self, jobs: List[Job], machines: List[Machine]):
        for machine in machines:
            machine.clear()
        self.waiting_pool.clear()
        self.job_pool.clear()

    def _update_waiting_pool(self, jobs: List[Job], curr_time: int, debug: bool):
        for job in jobs:
            if job.time_arr <= curr_time:
                if job.status != Job.Status.ARRIVED:
                    if job.status in (Job.Status.WAITING, Job.Status.READY):
                        job.wait_time += job.status.value
                    continue
                self.waiting_pool.append(job)
                job.wait_time = 0
                job.status = Job.Status.WAITING
                self._print_with_debug(f'\tAdd job {job.id} into waiting pool!', debug)
        pool_str = f'[{", ".join(str(job.id) for job in self.waiting_pool)}]'
        self._print_with_debug(f"\tWaiting pool: {pool_str}", debug)

    def _calculate_priorities(self, hdr: HDR, jobs: List[Job], machines: List[Machine], curr_time: int):
        for job in jobs:
            next_opr = job.oprs[job.next_opr]
            terminal_maker = TerminalDictMaker()
            terminal_maker.add_terminal(problem.JAT, job.time_arr)
            terminal_maker.add_terminal(problem.JCD, job.get_next_deadline())
            terminal_maker.add_terminal(problem.JD, job.get_job_deadline())
            terminal_maker.add_terminal(problem.JNPT, next_opr.get_avg_process_time())
            terminal_maker.add_terminal(problem.JRO, job.get_remain_opr())
            terminal_maker.add_terminal(problem.JRT, job.get_remain_process_time())
            terminal_maker.add_terminal(problem.JTPT, job.get_total_process_time())
            terminal_maker.add_terminal(problem.JS, job.get_slack_time(curr_time))
            terminal_maker.add_terminal(problem.JW, job.weight)
            terminal_maker.add_terminal(problem.JWT, job.wait_time)
            terminal_maker.add_terminal(problem.TNOW, curr_time)
            terminal_maker.add_terminal(problem.UTIL, sum(m.get_util(curr_time) for m in machines) / len(machines))
            try:
                job.prior = hdr.execute(**terminal_maker.var_dicts)
            except HDRException as e:
                self._logger.warning(f"HDR Exception: {e.msg}, use DEFAULT PRIOR instead", exc_info=True)
                job.prior = Simulator.DEFAULT_PRIOR

    def _update_job_pool(self):
        self.waiting_pool.sort(key=lambda x: x.prior, reverse=True)
        num_jobs_ready = min(self.pool_size, len(self.waiting_pool))
        for job in self.waiting_pool[:num_jobs_ready]:
            if len(self.job_pool) < self.pool_size:
                self.job_pool.append(job)
                job.status = Job.Status.READY
                self.waiting_pool.remove(job)

    def _assign_jobs_to_machines(self, machines: List[Machine], curr_time: int, debug: bool):
        for job in self.job_pool[:]:
            self._print_with_debug(f"\tEvaluate job {job.id}", debug)
            next_opr = job.oprs[job.next_opr]
            available_machines = list(next_opr.available_machines.keys())
            best_machine = None
            best_score = float('-inf')

            for machine in machines:
                if machine.get_status() != Machine.Status.RELAX or machine not in available_machines:
                    continue
                score = next_opr.available_machines.get(machine, float('-inf'))
                if score > best_score:
                    best_score = score
                    best_machine = machine

            if best_machine:
                best_machine.curr_job = job
                best_machine.finish_time = curr_time + next_opr.available_machines[best_machine]
                job.status = Job.Status.PROCESSING
                self.job_pool.remove(job)
                self._print_with_debug(f"\tAssign job {job.id} to machine {best_machine.id}", debug)

    def _update_machine_statuses(self, machines: List[Machine], curr_time: int, debug: bool):
        scheduled_jobs = 0
        for machine in machines:
            if machine.get_status() == Machine.Status.RELAX:
                continue
            if curr_time >= machine.finish_time:
                job = machine.curr_job
                job.next_opr += 1
                if job.next_opr == len(job.oprs):
                    job.status = Job.Status.FINISHED
                    scheduled_jobs += 1
                else:
                    job.status = Job.Status.ARRIVED
                    job.wait_time = 0
                job.finish_time = curr_time
                machine.processed_count += 1
                machine.curr_job = None
                self._print_with_debug(f"\tMachine {machine.id} completed operation {job.next_opr - 1} of job {job.id}.", debug)
        return scheduled_jobs

    def simulate(self, hdr: HDR, debug: bool = False, sleep_time: int | None = None):
        jobs = copy.deepcopy(self.problem.jobs)
        machines = copy.deepcopy(self.problem.machines)

        total_jobs = len(jobs)
        scheduled_jobs = 0
        curr_time = 0

        self._reset(jobs, machines)

        while scheduled_jobs < total_jobs:
            self._print_with_debug(f"Current time: {curr_time}---------", debug)

            self._update_waiting_pool(jobs, curr_time, debug)
            self._calculate_priorities(hdr, self.waiting_pool, machines, curr_time)
            self._update_job_pool()

            job_str = f'[{", ".join(str(j.id) for j in self.job_pool)}]'
            self._print_with_debug(f"\tJob pool: {job_str}", debug)

            self._assign_jobs_to_machines(machines, curr_time, debug)
            scheduled_jobs += self._update_machine_statuses(machines, curr_time, debug)

            self._print_with_debug("\tStatus of machines:", debug)
            for machine in machines:
                self._print_with_debug(f"\t\tMachine {machine.id}: {machine.get_status()}", debug)

            curr_time += 1
            if debug:
                time.sleep(0.02 if sleep_time is None else sleep_time)

        makespan = max(m.finish_time for m in machines)
        self._print_with_debug(f"Done!, makespan = {makespan}", debug)
        return makespan 
    
class Evaluator(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
        
    @abstractmethod
    def __call__(self, hdrs: List[HDR]) -> List[Tuple[HDR, float]]:
        pass
    
class SimulationBaseEvaluator(Evaluator):
    def __init__(self, problem):
        super().__init__(problem)
        self.simulator = Simulator(self.problem)
        self._logger = logging.getLogger(__name__)
        
    def __call__(self, hdrs) -> List[Tuple[HDR, float]]:
        self._logger.info(f'Start evaluate {len(hdrs)} HDR.')
        results = []
        for id, hdr in enumerate(hdrs):
            self._logger.info(f'Evaluate HDR {id + 1}/{len(hdrs)}')
            makespan = self.simulator.simulate(hdr, debug=False)
            fitness = -makespan
            results.append((hdr, fitness))
        self._logger.info(f'Successfully evaluate {len(results)}/{len(hdrs)} HDR.')
        return results
    
class StaticLLMSurrogateEvaluator(Evaluator):
    def __init__(self, llm_model: LLM, problem: Problem, prompt_template: str):
        super().__init__(problem)
        self.llm_model = llm_model
        self.context: str = ""
        self.vector_store = None
        self.examples: List[Tuple[HDR, float]] = []
        self.prompt_template = prompt_template
        self._summary_chunk = None
        self._criteria_chunk = None
        self._logger = logging.getLogger(__name__)
    
    def add_example(self, new_hdr: HDR, fitness: float):
        self.examples.append((new_hdr, fitness))
        
    def _update_min_max(self, need_update: list, new_value: int|float):
        if need_update[0] is None:
            need_update[0] = new_value
        else:
            need_update[0] = min(need_update[0], new_value)
            
        if need_update[1] is None:
            need_update[1] = new_value
        else:
            need_update[1] = max(need_update[1], new_value)
            
    def get_general_chunk(self):
        general_chunk = "Dynamic Job Shop Scheduling Problem\n"
        general_chunk += "Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan\n"
        general_chunk += f"Each machine have not a specific queue. All machine use a share job pool with size {self.problem.pool_size}.\n"
        general_chunk += f"We use a HDR to sort unorder job in waiting pool (infinity) and put them orderly into job pool, where these job is immediately match to corresponding machine to process if these machine is available.\n"
        general_chunk += f"**Note**: The value of HDR function is the priority (The higher the priority, the earlier it is assigned.)"
        return general_chunk
            
    def chunks_descript(self, problem: Problem) -> List[str]:
        chunks: List[str] = []

        # Chunk 1: General description
        # chunks.append(self.get_general_chunk())

        # Chunk 2-n: Job info grouped in batches of 12
        job_batch_size = 12
        summary = {
            "num_oprs": [None, None],
            "arrival_time": [None, None],
            "process_time": [None, None],
            "deadlines": [None, None],
            "priority": [None, None]
        }

        for i in range(0, len(problem.jobs), job_batch_size):
            batch = problem.jobs[i:i + job_batch_size]
            job_chunk = f"#### Job batch {i // job_batch_size + 1} (Jobs {batch[0].id} to {batch[-1].id}):\n"
            for job in batch:
                num_oprs = len(job.oprs)
                self._update_min_max(summary['num_oprs'], num_oprs)

                arrival_time = job.time_arr
                self._update_min_max(summary['arrival_time'], arrival_time)

                total_process_time = job.get_total_process_time()
                self._update_min_max(summary['process_time'], total_process_time)

                deadline = job.get_job_deadline()
                self._update_min_max(summary['deadlines'], deadline)

                prior = job.prior
                self._update_min_max(summary['priority'], prior)

                job_chunk += (
                    f"* Job {job.id}: "
                    f"Num operations: {num_oprs}, "
                    f"Arrival time: {arrival_time}, "
                    f"Total process time: {total_process_time:.2f}, "
                    f"Deadline: {deadline:.2f}, "
                    f"Priority: {prior:.2f}\n"
                )
            chunks.append(job_chunk)

        # Chunk X: Criteria
        criteria_chunk = "### Evaluation Criteria:\n"
        criteria_chunk += "\n".join(str(t) for t in problem.terminals)
        #chunks.append(criteria_chunk)
        self._criteria_chunk = criteria_chunk

        # Chunk X+1: Summary
        summary_chunk = "### Problem Summary:\n"
        summary_chunk += f"- Num machines: {len(problem.machines)}\n"
        summary_chunk += f"- Num jobs: {len(problem.jobs)}\n"
        summary_chunk += f"- Operation/job: min={summary['num_oprs'][0]}, max={summary['num_oprs'][1]}\n"
        summary_chunk += f"- Arrival time: min={summary['arrival_time'][0]}, max={summary['arrival_time'][1]}\n"
        summary_chunk += f"- Process time: min={summary['process_time'][0]:.2f}, max={summary['process_time'][1]:.2f}\n"
        summary_chunk += f"- Deadline: min={summary['deadlines'][0]:.2f}, max={summary['deadlines'][1]:.2f}\n"
        summary_chunk += f"- Priority: min={summary['priority'][0]:.2f}, max={summary['priority'][1]:.2f}\n"
        summary_chunk += "**Note**: All time-related fields use the same time unit.\n"
        # chunks.append(summary_chunk)
        self._summary_chunk = summary_chunk

        return chunks

    
    def store_vector(self, chunk: list[str]):
        embed_model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_texts(chunk, embed_model)
        
    def init_vector_store(self):
        chunk = self.chunks_descript(self.problem)
        self.store_vector(chunk)
        
        
    def init_problem_context(self):
        query = f"Dynamic job shop scheduling with {len(self.problem.jobs)} jobs, {len(self.problem.machines)} machines, shared pool of size {self.problem.pool_size}. Jobs have arrival times and multiple operations."
        retrieved_chunks = self.vector_store.similarity_search(query, k=4)
        job_chunks = "\n".join(doc.page_content for doc in retrieved_chunks)
        self.context = self.get_general_chunk() + "\n" + self._summary_chunk + \
            "\n" + self._criteria_chunk + "\n" + "### Some of job description: \n" + job_chunks 
        
    def _build_hdrs(self, hdrs: List[HDR]):
        hdrs_str = ""
        for i in range(len(hdrs)):
            hdrs_str += "---------\n"
            hdrs_str += f"HDR {i + 1}:\n"
            hdrs_str += str(hdrs[i]) + '\n'
        return hdrs_str
    
    def _build_examples(self):
        examples_str = ""
        for i in range(len(self.examples)):
            hdr, fitness = self.examples[i]
            examples_str += "------\n"
            examples_str += f"Example {i + 1} with fitness {fitness:.2f} \n"
            examples_str += str(hdr) + '\n'
        return examples_str
    
    def _process_json_response(self, data: dict):
        evaluated = data['evaluated_hdrs']
        evaluated_hdrs: List[Tuple[HDR, float]] = []
        for json_obj in evaluated:
            try:
                new_hdr = CodeSegmentHDR(code=json_obj['code'])
                fitness = float(json_obj['fitness'])
                evaluated_hdrs.append((new_hdr, fitness))
            except HDRException as e:
                self._logger.error(f'HDR Exception: {e.msg} when process response from LLM', exc_info=True)
                continue
        return evaluated_hdrs
            
    def __call__(self, hdrs: List[HDR]):
        self._logger.info(f'Start evaluate {len(hdrs)} HDR.')
        examples = self._build_examples()
        hdrs_str = self._build_hdrs(hdrs)
        
        prompt = self.prompt_template.format(
            context=self.context,
            examples=examples,
            hdrs=hdrs_str
        )
        
        response = self.llm_model.get_response(prompt)
        json_repsonse = self.llm_model.extract_response(response)
        results = self._process_json_response(json_repsonse)
        self._logger.info(f'Successfully evaluate {len(results)}/{len(hdrs)} HDR.')
        return results

class EventDrivenLLMSurrogateEvaluator(Evaluator):
    def __init__(self, llm_model: LLM, problem: Problem, 
                 prompt_template: str, num_segments: int, batch_size: int,
                 max_retries: int = 3, scaling_schedule: str|Literal['linear', 'sin', 'random']|None = None,
                 start_rate: float = 0.8, max_calls_to_end: int = 100):
        super().__init__(problem)
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._logger = logging.getLogger(__name__)
        self.event_store = collections.defaultdict(list)
        self.history_store = []
        self.job_map = {j.id: j for j in self.problem.jobs}
        self.scaling_schedule = scaling_schedule
        self.call_cnt = 0
        self.start_rate = start_rate
        self.max_calls_to_end = max_calls_to_end
        
        self._build_event_map()
        self.times = self._build_times(list(self.event_store.keys()), num_segments)
        
    def _get_scaling_factor(self):
        if self.scaling_schedule is None:
            return 1.0
        else:
            t = min(self.call_cnt/self.max_calls_to_end, 1.0)
            if self.scaling_schedule == 'linear':
                return self.start_rate + (1.0 - self.start_rate) * t
            elif self.scaling_schedule == 'sin':
                return self.start_rate + (1.0 - self.start_rate) * (1 + math.sin(t * math.pi)) / 2
            elif self.scaling_schedule == 'random':
                return self.start_rate + (1.0 - self.start_rate) * random.uniform(0, 1)
            else:
                self._logger.warning(f"Invalid scaling schedule: {self.scaling_schedule}")
                return 1.0
        
    def _build_times(self, times: List[int], num_segments: int):
        # Chia theo time density
        # Tính trọng số bằng số sự kiện tại mỗi time
        weights = {t: len(self.event_store.get(t, [])) for t in times}
        total_weight = sum(weights.values())
        target_weight = total_weight / num_segments
        
        segments = []
        curr_sum = 0
        curr_bucket = []
        for t in sorted(times):
            curr_bucket.append(t)
            curr_sum += weights[t]
            if curr_sum >= target_weight:
                segments.append(curr_bucket[-1])
                curr_bucket = []
                curr_sum = 0
                
        if curr_bucket:
            if len(segments) < num_segments - 1:
                segments.append(curr_bucket[-1])
            
        segments.append(1e6)
            
        return segments
        
    def _build_event_map(self):
        for job in self.problem.jobs:
            t_arr = job.time_arr
            deadline = job.get_job_deadline()
            self.event_store[t_arr].append({'type': 'arrival', 'job': job.id})
            self.event_store[deadline].append({'type': 'deadline', 'job': job.id})
            
        self.event_store[1e6].append({'type': 'end', 'job': None})
            
    def _format_event_chunk(self, last_time: int, now_time: int):
        arr_evs = []
        dl_evs = []
        for t in range(last_time, now_time + 1):
            evs = self.event_store.get(t, [])
            arr_evs.extend([e for e in evs if e['type'] == 'arrival'])
            dl_evs.extend([e for e in evs if e['type'] == 'deadline'])
        
        lines = [f'### Event at time {now_time}']
        arr_evs_avg_process_time = sum(self.job_map[e['job']].get_total_process_time() for e in arr_evs) / len(arr_evs) if len(arr_evs) > 0 else 0
        lines.append(f'- Arrival job: {", ".join(str(e["job"]) for e in arr_evs)} with avg process time {arr_evs_avg_process_time:.2f}')
        lines.append(f'- Meet deadline jobs: {", ".join(str(e["job"]) for e in dl_evs)}')
        
        return '\n'.join(lines)
    
    def _last_event_time(self, end: int):
        return f"### Event at last time {end}. All jobs must be completed."
    
    def _get_general_chunk(self):
        general_chunk = "# Dynamic Job Shop Scheduling Problem\n"
        general_chunk += "- Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan\n"
        general_chunk += f"- We have {len(self.problem.machines)} machines.\n"
        general_chunk += f"- Each machine have not a specific queue. All machine use a share job pool with size {self.problem.pool_size}.\n"
        general_chunk += f"- We use a HDR to sort unorder job in waiting pool (infinity) and put them orderly into job pool, where these job is immediately match to corresponding machine to process if these machines is available.\n"
        general_chunk += f"- **Note**: The value of HDR function is the priority (The higher the priority, the earlier it is assigned.)"
        return general_chunk
    
    def _build_hdr_with_history(self):
        hdrs_str = ""
        for i in range(len(self.history_store)):
            lines = [f'--- HDR {i + 1} ---']
            lines.append('**Code**')
            lines.append(self.history_store[i]['code'])
            lines.append('**Historical Performance**')
        
            completed_jobs = self.history_store[i].get('completed_jobs', [])
            lines.append(f'- Completed jobs: {", ".join(str(j) for j in completed_jobs) if completed_jobs else "None"}')
            
            predicted_makespan = self.history_store[i].get('predicted_makespan', None)
            lines.append(f'- Predicted makespan: {predicted_makespan if predicted_makespan is not None else "None"}')
            
            remaining_jobs = self.history_store[i].get('remaining_jobs', [])
            # Remaining jobs: {j_id: completed_opr_id}
            total_remain_oprs = 0
            min_remain_oprs = float('inf')
            max_remain_oprs = 0
            
            total_remain_process_time = 0
            min_remain_process_time = float('inf')
            max_remain_process_time = 0
            for j_dict in remaining_jobs:
                next_opr = j_dict['op'] + 1
                j_id = int(j_dict['job'])
                self.job_map[j_id].next_opr = next_opr
                
                remain_opr = self.job_map[j_id].get_remain_opr()
                remain_process_time = self.job_map[j_id].get_remain_process_time()
                
                total_remain_oprs += remain_opr
                total_remain_process_time += remain_process_time
                
                min_remain_oprs = min(min_remain_oprs, remain_opr)
                max_remain_oprs = max(max_remain_oprs, remain_opr)
                
                min_remain_process_time = min(min_remain_process_time, remain_process_time)
                max_remain_process_time = max(max_remain_process_time, remain_process_time)
                
            lines.append(f'- Remaining jobs: {", ".join(str(j_dict["job"]) for j_dict in remaining_jobs)}')
            lines.append(f'- Remaining operations: min={min_remain_oprs}, max={max_remain_oprs}, avg={total_remain_oprs / len(remaining_jobs) if len(remaining_jobs) > 0 else 0:.2f}')
            lines.append(f'- Remaining process time: min={min_remain_process_time:.2f}, max={max_remain_process_time:.2f}, avg={total_remain_process_time / len(remaining_jobs) if len(remaining_jobs) > 0 else 0:.2f}')

            hdrs_str += '\n'.join(lines)
        return hdrs_str
    
    def _build_hdrs(self, hdrs: List[HDR]):
        hdrs_str = ""
        for i in range(len(hdrs)):
            hdrs_str += "---------\n"
            hdrs_str += f"HDR {i + 1}:\n"
            hdrs_str += str(hdrs[i]) + '\n'
        return hdrs_str
    
    def _process_json_response(self, data: dict):
        predicted = data['predicted']
        
        self.history_store.clear()
        for i in range(len(predicted)):
            self.history_store.append({
                'code': predicted[i]['code'],
                'completed_jobs': predicted[i]['completed_jobs'],
                'predicted_makespan': predicted[i]['makespan'],
                'remaining_jobs': predicted[i]['remaining_jobs']
            })
            
            
        self._logger.info(f'Successfully update history store with {len(predicted)} HDRs.')
        
        evaluated_hdrs: List[Tuple[HDR, float]] = []
        for json_obj in predicted:
            try:
                new_hdr = CodeSegmentHDR(code=json_obj['code'])
                fitness = float(json_obj['fitness'])
                evaluated_hdrs.append((new_hdr, fitness))
            except HDRException as e:
                self._logger.error(f'HDR Exception: {e.msg} when process response from LLM', exc_info=True)
                continue
        return evaluated_hdrs
    
    def evaluate_batch(self, hdrs: List[HDR]):
        self._logger.info(f'Start evaluate {len(hdrs)} HDR.')
        for i in range(len(self.times)):
            t = int(self.times[i])
            last_t = int(self.times[i - 1]) if i > 0 else 0
            event_chunk = self._format_event_chunk(last_t, t) if i < len(self.times) - 1 else self._last_event_time(t)
            history_chunk = self._build_hdr_with_history() if i > 0 else self._build_hdrs(hdrs)
            
            prompt = self.prompt_template.format(
                problem_info=self._get_general_chunk(),
                current_time=t,
                events=event_chunk,
                hdrs_with_history=history_chunk,
                next_time=self.times[i + 1] if i < len(self.times) - 1 else 1e6
            )
            
            response = self.llm_model.get_response(prompt)
            json_repsonse = self.llm_model.extract_response(response)
            results = self._process_json_response(json_repsonse)
        
        return results
    
    def _retry(self, fn, max_retries: int, *args, **kwargs):
        for attempt in range(1, max_retries+1):
            try:
                return fn(*args, **kwargs)
            except (LLMException, HDRException) as e:
                self._logger.warning(f"Attempt {attempt}/{max_retries} failed in {fn.__name__}: {e.msg}")
        raise LLMException(f"All {max_retries} retries failed for {fn.__name__}")
    
    def __call__(self, hdrs: List[HDR]):
        all_results = []
        self.call_cnt += 1
        scaling_factor = self._get_scaling_factor()
        for i in range(0, len(hdrs), self.batch_size):
            self._logger.info(f"Processing HDR batch {i // self.batch_size + 1} of {((len(hdrs) - 1) // self.batch_size + 1)}")
            batch = hdrs[i:i + self.batch_size]
            results = self._retry(self.evaluate_batch, self.max_retries, batch)
            all_results.extend(results)
        scaled_results = [(hdr, fitness * scaling_factor) for hdr, fitness in all_results]
        return scaled_results
        
        
        
        
        
    
        
