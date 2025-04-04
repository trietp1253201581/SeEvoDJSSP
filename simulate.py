from model import HDR, TerminalDictMaker
import model
from problem import Problem, Job, Machine
import copy
from typing import List
import time

def _print_with_debug(msg: str, debug: bool=False):
    if debug:
        print(msg)
        
class Simulator:
    def __init__(self, hdr: HDR, problem: Problem, pool_size: int):
        self.hdr = hdr
        self.problem = problem
        self.waiting_pool: List[Job] = []
        self.job_pool: List[Job] = []
        self.pool_size = pool_size
        
    def add_job_into_pool(self, new_job: Job):
        if len(self.job_pool) < self.pool_size:
            self.job_pool.append(new_job)
            return True
        return False
        
    def simulate(self, debug: bool=False, sleep_time: int|None=None):
        jobs = copy.deepcopy(self.problem.jobs)
        machines = copy.deepcopy(self.problem.machines)
        
        total_jobs = len(jobs)
        scheduled_jobs = 0
        
        # Reset các máy
        for machine in machines:
            machine.clear()
            
        self.waiting_pool.clear()    
        self.job_pool.clear()
        
        curr_time = 0
        
        while scheduled_jobs < total_jobs:
            _print_with_debug(f"Current time: {curr_time}---------", debug)
            
            # Add job into pool
            for job in jobs:
                if job.time_arr <= curr_time:
                    if job.status != Job.Status.ARRIVED:
                        if job.status == Job.Status.WAITING or job.status == Job.Status.READY:
                            job.wait_time += job.status.value
                        continue
                    else: #job.status == Job.Status.ARRIVED:
                        self.waiting_pool.append(job)
                        job.wait_time = 0
                        job.status = Job.Status.WAITING
                        _print_with_debug(f'\tAdd job {job.id} into waiting pool!', debug)
            
            # Current waiting pool
            pool_str = f'[{", ".join(str(job.id) for job in self.waiting_pool)}]'
            _print_with_debug(f"\tWaiting pool: {pool_str}", debug)
            
            # Calculate job priority from hdr and problem
            for job in self.waiting_pool:
                next_opr = job.oprs[job.next_opr]

                terminal_maker = TerminalDictMaker()
                terminal_maker.add_terminal(model.JAT, job.time_arr)
                terminal_maker.add_terminal(model.JCD, job.get_next_deadline())
                terminal_maker.add_terminal(model.JD, job.get_job_deadline())
                terminal_maker.add_terminal(model.JNPT, next_opr.get_avg_process_time())
                terminal_maker.add_terminal(model.JRO, job.get_remain_opr())
                terminal_maker.add_terminal(model.JRT, job.get_remain_process_time())
                terminal_maker.add_terminal(model.JTPT, job.get_total_process_time())
                terminal_maker.add_terminal(model.JS, job.get_slack_time(curr_time))
                terminal_maker.add_terminal(model.JW, job.weight)
                terminal_maker.add_terminal(model.JWT, job.wait_time)
                terminal_maker.add_terminal(model.TNOW, curr_time)
                terminal_maker.add_terminal(model.UTIL, sum(m.get_util(curr_time) for m in machines)/len(machines))
                
                job.prior = self.hdr.execute(
                    **terminal_maker.var_dicts
                )
            
            self.waiting_pool.sort(key=lambda x: x.prior, reverse=True)
            num_jobs_ready = min(self.pool_size, len(self.waiting_pool))
            for job in self.waiting_pool[:num_jobs_ready]:
                if len(self.job_pool) < self.pool_size:
                    self.job_pool.append(job)
                    job.status = Job.Status.READY
                    self.waiting_pool.remove(job)

            job_str = f'[{", ".join(str(j.id) for j in self.job_pool)}]'
            _print_with_debug(f"\tJob pool: {job_str}", debug)
                
            # Assign ready job into machine if available, choose a machine that 
            # next operation process in least time
            for i in range(len(self.job_pool)):
                print(f"\tEvaluate job {self.job_pool[i].id}")
                job = self.job_pool[i]
                next_opr = job.oprs[job.next_opr]
                best_machine = None
                best_score = float('-inf')
                available_machines = list(next_opr.available_machines.keys())
                for machine in machines:
                    if machine.get_status() != Machine.Status.RELAX:
                        continue
                    if machine not in available_machines:
                        continue
                    score = next_opr.available_machines.get(machine, float('-inf'))
                    if score > best_score:
                        best_score = score
                        best_machine = machine
                if best_machine is None:
                    continue
                best_machine.curr_job = job
                best_machine.finish_time = curr_time + next_opr.available_machines.get(best_machine)
                job.status = Job.Status.PROCESSING
                
                _print_with_debug(f"\tAssign job {job.id} to machine {best_machine.id}", debug)
            
            for job in self.job_pool[:]:
                if job.status == Job.Status.PROCESSING:
                    self.job_pool.remove(job)
            
            # Check status of each machine
            for machine in machines:
                if machine.get_status() == Machine.Status.RELAX:
                    continue
                if curr_time >= machine.finish_time:
                    job = machine.curr_job
                    # Chuyển lại vào pool hoặc gắn cờ finish
                    job.next_opr += 1
                    if job.next_opr == len(job.oprs):
                        job.status = Job.Status.FINISHED
                        scheduled_jobs += 1
                    else:
                        # CHuyển về ARRIVED để đầu vòng lặp sau đc thêm vào Job Pool
                        job.status = Job.Status.ARRIVED
                        job.wait_time = 0
                    job.finish_time = curr_time
                    machine.processed_count += 1
                    machine.curr_job = None
                    _print_with_debug(f"\tMachine {machine.id} completed operation {job.next_opr - 1} of job {job.id}.", debug)
            
            _print_with_debug("\tStatus of machines:", debug)
            for machine in machines:
                print(f"\t\tMachine {machine.id}: {machine.get_status()}")
            
            curr_time += 1 
            if debug:
                time.sleep(1 if sleep_time is None else sleep_time)
        makespan = max(m.finish_time for m in machines)
        _print_with_debug(f"Done!, makespan = {makespan}", debug)
        return makespan        
                
def test():
    from model import CodeSegmentHDR
    with open('template.txt', 'r') as f:
        lines = f.readlines()
        code = "".join(lines)
    hdr = CodeSegmentHDR(code=code)
    import random
    random.seed(42)
    
    problem = Problem()
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    for job in problem.jobs:
        print(str(job))
        
    simulator = Simulator(hdr=hdr, problem=problem, pool_size=2)
    simulator.simulate(debug=True)
    
test()
                
            
                        
                        