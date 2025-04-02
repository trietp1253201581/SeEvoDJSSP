from model import HDR
from problem import Problem, Job, Machine
import copy
from typing import List
import time

def _print_with_debug(msg: str, debug: bool=False):
    if debug:
        print(msg)
        
class Simulator:
    def __init__(self, hdr: HDR, problem: Problem):
        self.hdr = hdr
        self.problem = problem
        
    def simulate(self, top_k: int, debug: bool=False):
        jobs = copy.deepcopy(self.problem.jobs)
        machines = copy.deepcopy(self.problem.machines)
        
        total_jobs = len(jobs)
        scheduled_jobs = 0
        
        # Reset các máy
        for machine in machines:
            machine.clear()
            
        job_pool: List[Job] = []
        
        curr_time = 0
        
        while scheduled_jobs < total_jobs:
            _print_with_debug(f"Current time: {curr_time}---------", debug)
            
            # Add job into pool
            for job in jobs:
                if job.time_arr <= curr_time:
                    if job.status != Job.Status.ARRIVED:
                        if job.status == Job.Status.IN_POOL or job.status == Job.Status.READY:
                            job.wait_time += job.status.value
                        continue
                    else: #job.status == Job.Status.ARRIVED:
                        job_pool.append(job)
                        job.wait_time = 0
                        job.status = Job.Status.IN_POOL
                        _print_with_debug(f'\tAdd job {job.id} into job pool!', debug)
            
            # Current pool
            pool_str = f'[{", ".join(str(job.id) for job in job_pool)}]'
            _print_with_debug(f"\tJob pool: {pool_str}", debug)
            
            # Calculate job priority from hdr and problem
            for job in job_pool:
                next_opr = job.oprs[job.next_opr]
                mean_p = 0
                for m, p in next_opr.available_machines.items():
                    mean_p += p
                mean_p /= len(next_opr.available_machines)
                job.prior = self.hdr.execute(
                    d=next_opr.deadline,
                    ct=curr_time,
                    p=mean_p,
                    now_opr=job.next_opr-1,
                    ta=job.time_arr,
                    wt=job.wait_time
                )
            
            job_pool.sort(key=lambda x: x.prior, reverse=True)
            for i in range(min(len(job_pool), top_k) - 1):
                job_pool[i].status = Job.Status.READY
            job_str = f'{", ".join(str(j.id) for j in job_pool[:top_k])}'
            _print_with_debug(f"\tTop {top_k} ready job: {job_str}", debug)
                
            # Assign ready job into machine if available, choose a machine that 
            # next operation process in least time
            for i in range(min(len(job_pool), top_k)):
                print(f"\tEvaluate job {job_pool[i].id}")
                job = job_pool[i]
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
                
            # Clear processing job
            for job in job_pool[:]:
                if job.status == Job.Status.PROCESSING:
                    job_pool.remove(job)
                
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
                time.sleep(1)
        makespan = max(m.finish_time for m in machines)
        _print_with_debug(f"Done!, makespan = {makespan}", debug)
        return makespan        
                
def test():
    hdr_code = """
def hdr(p, d, ct, tax, now_opr, wt):
    return p + ct/2 + wt
    """
    from model import CodeSegmentHDR
    hdr = CodeSegmentHDR(code=hdr_code)
    import random
    random.seed(42)
    
    problem = Problem()
    problem.random_generate(num_jobs=5, max_oprs_each_job=3, num_machines=2, max_arr_time=16)
    
    for job in problem.jobs:
        print(str(job))
        
    simulator = Simulator(hdr=hdr, problem=problem)
    simulator.simulate(top_k=2, debug=True)
    
test()
                
            
                        
                        