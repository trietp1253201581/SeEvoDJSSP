from enum import Enum
from typing import List
import random

class Operation:
    def __init__(self, deadline: float, 
                 available_machines: dict['Machine', float]):
        
        self.deadline = deadline
        self.available_machines = available_machines
        self.avg_process_time = None

    def get_avg_process_time(self):
        if self.avg_process_time is not None:
            return self.avg_process_time
        total_process_time = 0.0
        for _, p in self.available_machines.items():
            total_process_time += p
        return total_process_time/len(self.available_machines)
    
    def __str__(self):
        available_m_str = ""
        for machine, process_time in self.available_machines.items():
            available_m_str += str(machine.id) + ":" + str(f'{process_time:.1f}') + ", "
        return f'Operation(deadline={self.deadline}, available_machines=[{available_m_str}])'
    
class Job:
    class Status(Enum):
        ARRIVED = 1
        WAITING = 2
        READY = 3
        PROCESSING = 0
        FINISHED = -1
        
    def __init__(self, id: int):
        self.id = id
        self.next_opr = 0
        self.time_arr = 0
        self.finish_time = 0
        self.wait_time = 0
        self.weight = 1.0
        self.oprs: List[Operation] = []
        self.status = Job.Status.ARRIVED
        self.prior = float('-inf')
        
    def get_remain_opr(self):
        return len(self.oprs) - self.next_opr
    
    def get_remain_process_time(self):
        remain_process_time = 0
        for i in range(self.next_opr, len(self.oprs)):
            remain_process_time += self.oprs[i].get_avg_process_time()
        return remain_process_time
    
    def get_total_process_time(self):
        total_process_time = 0
        for opr in self.oprs:
            total_process_time += opr.get_avg_process_time()
        return total_process_time
    
    def get_next_deadline(self):
        return self.oprs[self.next_opr].deadline
    
    def get_job_deadline(self):
        return self.oprs[-1].deadline
    
    def get_slack_time(self, curr_time: float):
        return max(0, self.get_job_deadline() - curr_time - self.get_remain_process_time())
        
    def add_opr(self, new_opr: Operation):
        self.oprs.append(new_opr)
    
    def __str__(self):
        return f'Job(id={self.id}, status={self.status}, time_arr={self.time_arr}, next_opr={self.next_opr}, oprs=[{", ".join(str(x) for x in self.oprs)}])'
    
    def __eq__(self, value):
        if not isinstance(value, Job):
            return False
        return self.id == value.id
    
    def __hash__(self):
        return hash(self.id)
        
        
class Machine:
    class Status(Enum):
        RELAX = 0
        PROCESSING = 1
        
    def __init__(self, id: int):
        self.id = id
        self.curr_job: Job = None
        self.finish_time = 0
        self.processed_count = 0
        self.processed_time = 0
        
    def clear(self):
        self.curr_job = None
        self.finish_time = 0
        self.processed_count = 0
        
    def get_status(self) -> Status:
        if self.curr_job is None:
            return Machine.Status.RELAX
        return Machine.Status.PROCESSING
    
    def get_relax_time(self, curr_time: float):
        return max(0, curr_time - self.finish_time)
    
    def get_remain_time(self, curr_time: float):
        return max(0, self.finish_time - curr_time)
    
    def get_util(self, curr_time: float):
        return self.processed_time / curr_time if curr_time > 0 else 0
        
    def __str__(self):
        return f'Machine(id={self.id}, curr={self.curr_job.id if self.curr_job is not None else None}, finish_time={self.finish_time}, processed={self.processed_count}])'
    
    def __eq__(self, value):
        if not isinstance(value, Machine):
            return False
        return self.id == value.id
    
    def __hash__(self):
        return hash(self.id)

class Problem:
    def __init__(self):
        self.jobs: List[Job] = []
        self.machines: List[Machine] = []
        
    def random_generate(self, num_jobs: int, max_oprs_each_job: int, num_machines: int, max_arr_time = 1000):
        self.jobs = []
        self.machines = [Machine(i) for i in range(num_machines)]
        for i in range(num_jobs):
            new_job = Job(id=i)
            new_job.time_arr = random.randint(0, max_arr_time)
            num_opr = random.randint(1, max_oprs_each_job)
            for pos in range(num_opr):
                last_d = new_job.oprs[-1].deadline if len(new_job.oprs) > 0 else 0
                d = random.randint(last_d + 1, max_arr_time * 3 // 2 + last_d)
                K_ids = random.sample(range(num_machines), random.randint(1, num_machines))
                K = [self.machines[id] for id in K_ids]
                available_machines = dict()
                for m in K:
                    p = random.randint(1, max_arr_time * 4 // 3)
                    available_machines[m] = p
                new_opr = Operation(d, available_machines)
                new_job.add_opr(new_opr)
            self.jobs.append(new_job)
            

