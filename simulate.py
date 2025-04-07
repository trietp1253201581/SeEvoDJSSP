from model import HDR
import problem
from problem import Problem, Job, Machine, TerminalDictMaker
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
        self.waiting_pool: List[Job] = []
        self.job_pool: List[Job] = []
        self.pool_size = self.problem.pool_size

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
                _print_with_debug(f'\tAdd job {job.id} into waiting pool!', debug)
        pool_str = f'[{", ".join(str(job.id) for job in self.waiting_pool)}]'
        _print_with_debug(f"\tWaiting pool: {pool_str}", debug)

    def _calculate_priorities(self, jobs: List[Job], machines: List[Machine], curr_time: int):
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
            job.prior = self.hdr.execute(**terminal_maker.var_dicts)

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
            _print_with_debug(f"\tEvaluate job {job.id}", debug)
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
                _print_with_debug(f"\tAssign job {job.id} to machine {best_machine.id}", debug)

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
                _print_with_debug(f"\tMachine {machine.id} completed operation {job.next_opr - 1} of job {job.id}.", debug)
        return scheduled_jobs

    def simulate(self, debug: bool = False, sleep_time: int | None = None):
        jobs = copy.deepcopy(self.problem.jobs)
        machines = copy.deepcopy(self.problem.machines)

        total_jobs = len(jobs)
        scheduled_jobs = 0
        curr_time = 0

        self._reset(jobs, machines)

        while scheduled_jobs < total_jobs:
            _print_with_debug(f"Current time: {curr_time}---------", debug)

            self._update_waiting_pool(jobs, curr_time, debug)
            self._calculate_priorities(self.waiting_pool, machines, curr_time)
            self._update_job_pool()

            job_str = f'[{", ".join(str(j.id) for j in self.job_pool)}]'
            _print_with_debug(f"\tJob pool: {job_str}", debug)

            self._assign_jobs_to_machines(machines, curr_time, debug)
            scheduled_jobs += self._update_machine_statuses(machines, curr_time, debug)

            _print_with_debug("\tStatus of machines:", debug)
            for machine in machines:
                _print_with_debug(f"\t\tMachine {machine.id}: {machine.get_status()}", debug)

            curr_time += 1
            if debug:
                time.sleep(0.02 if sleep_time is None else sleep_time)

        makespan = max(m.finish_time for m in machines)
        _print_with_debug(f"Done!, makespan = {makespan}", debug)
        return makespan 