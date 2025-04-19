from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import OpenRouterLLM
from problem import Problem
from typing import List, Tuple
from model import HDR, CodeSegmentHDR

class LLMSurogate:
    def __init__(self, llm_model: OpenRouterLLM, problem: Problem, prompt_template: str):
        self.llm_model = llm_model
        self.context: str = ""
        self.vector_store = None
        self.examples: List[Tuple[HDR, float]] = []
        self.problem = problem
        self.prompt_template = prompt_template
        self._summary_chunk = None
        self._criteria_chunk = None
    
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
        general_chunk += f"We use a HDR to sort unordere job in waiting pool (infinity) and put them orderly into job pool, where these job is immediately match to corresponding machine to process if these machine is available.\n"
        return general_chunk
            
    def chunks_descript(self, problem: Problem) -> List[str]:
        chunks: List[str] = []

        # Chunk 1: General description
        # chunks.append(self.get_general_chunk())

        # Chunk 2-n: Job info grouped in batches of 20
        job_batch_size = 20
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
            new_hdr = CodeSegmentHDR(code=json_obj['code'])
            fitness = float(json_obj['fitness'])
            evaluated_hdrs.append((new_hdr, fitness))
        return evaluated_hdrs
            
    def evaluate(self, hdrs: List[HDR]):
        examples = self._build_examples()
        hdrs_str = self._build_hdrs(hdrs)
        
        prompt = self.prompt_template.format(
            context=self.context,
            examples=examples,
            hdrs=hdrs_str
        )
        print(prompt)
        
        response = self.llm_model.get_response(prompt)
        print(response)
        json_repsonse = self.llm_model.extract_response(response)
        
        return self._process_json_response(json_repsonse)
        

        
        
    

