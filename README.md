# SeEvoDJSSP
## Cấu trúc mã nguồn
* [model.py](./model.py): Chứa các mô hình đại diện cho các HDR  hiện đang dùng CodeSegmentHDR.
* [problem.py](./problem.py): Chứa các mô hình về bài toán lập lịch động, bao gồm Operation, Job, Machine và các Terminal được sử dụng.
* [llm.py](./llm.py): Chứa các lớp và phương thức để gọi và lấy response từ LLM (đang dùng LLM trên OpenRouter, có các bản free)
* [evaluate.py](./evaluate.py): Chứa Simulator để chạy mô phỏng với 1 HDR và 1 Problem và các phương thức đánh giá fitness của 1 HDR (simulaton-based và surrogate)
* [basic_evo.py](./basic_evo.py): Chứa các lớp cơ bản của tính toán tiến hóa như Individual, Population và định
nghĩa interface Operator (các toán tử tiến hóa được sử dụng)
* [se_evo.py](./se_evo.py): Triển khai các toán tử Se-Evo, phần chính, chứa các LLM-Base Operator và các Reflection Operator.
* [prompt_template.py](./prompt_template.py): Chứa các string template của các prompt được sử dụng trong các LLM-Base Operator
* [template.txt](./template.txt): Chứa template của 1 code segment nên được trả về từ LLM.
## Môi trường phát triển
- Ngôn ngữ: Python (3.13.3)
- IDE: VS Code
- OS: Windows
- Hỗ trợ chạy thử nghiệm: Kaggle Notebook
## Chạy chương trình
Chạy file `main.py`.
```bash
python -u main.py
```

Hoặc tạo 1 file tương tự với các bước:
0. Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```
1. Tạo logger (tùy chọn)
```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Định dạng chung
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(name)s (%(filename)s:%(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler ghi vào file
file_handler = logging.FileHandler(f'process_{datetime.now().strftime("%Y_%m_%d")}.log')
file_handler.setFormatter(formatter)

# Handler ghi ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Thêm cả 2 handler vào logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

2. Tạo Problem.
Hiện đã cung cấp cả 2 phương thức tạo problem: Random hoàn toàn (`problem.random_generate`) và Random một phần (`problem.custom_generate`). Ở Random 1 phần có thể chọn phân phối thời gian đến (uniform, burst, v.v) cũng như phân phối thời gian xử lý, cùng với deadline factor.
Sau đây là ví dụ tạo Problem với 240 job, không quá 5 operation mỗi job, có 20 máy với pool size là 15.
```python
# Nên đặt random.seed cho giống nhau
random.seed(42)

problem = Problem(AVAIABLE_TERMINALS, pool_size=15)
problem.custom_generate(num_jobs=240, max_oprs_each_job=5, 
                        num_machines=20, max_arr_time=120, 
                        arrival_type='uniform', proc_dist='uniform', 
                        deadline_factor=1.4)
```

3. Thêm API Key và tạo LLM.
Thêm API Key cần thiết vào file `config.json` trong cùng thư mục dự án. Hiện hỗ trợ OpenRouterAPI và GoogleAIStudioAPI.
- Với OpenRouterAPI, thêm các trường sau:
```json
{
    "OPEN_ROUTER_PROVISION_KEY": "Your-provison-key",
    "OPEN_ROUTER_API_KEY": "your-api-key",
    "OPEN_ROUTER_API_KEY_HASH": "your-api-key-hash"
}
```
 trong đó trường `OPEN_ROUTER_PROVISION_KEY` là bắt buộc. 2 trường khác sẽ được tự khởi tạo khi bạn dùng lần đầu và sẽ được dùng lại cho các lần kế tiếp, hoặc có thể tự set nếu đã có API key.
- Với Google AI Studio API, thêm trường
```json
{
    "GOOGLE_AI_API_KEY": "your-api-key"
}
```

Sau đó, tùy vào chọn API nào mà gọi class tương ứng.
```python
# Nếu dùng Open Router API
llm_model = OpenRouterLLM('deepseek', 'deepseek-r1-zero', free=True, timeout=(60, 600),
                          core_config='config.json',
                          runtime_config='llm_runtime_config.json')

# Nếu dùng Google AI Studio
llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='config.json',
                              runtime_config='llm_runtime_config.json')
```

4. Tạo các Operator (có sẵn hoặc tự tạo thêm, miễn tuân thủ prototype)
```python
# Create Operator
llm_init_func = LLMInitOperator(problem, llm_model, prompt_template=pt.INIT_IND_PROMPT_TEMPLATE)
llm_crossover_func = LLMCrossoverOperator(problem, llm_model, prompt_template=pt.CROSSOVER_PROMPT_TEMPLATE)
llm_mutation_func = LLMMutationOperator(problem, llm_model, prompt_template=pt.MUTATION_PROMPT_TEMPLATE)
co_evo_func = CoEvoOperator(problem, llm_model, prompt_template=pt.CO_EVO_PROMPT_TEMPLATE)
self_evo_func = SelfEvoOperator(problem, llm_model, prompt_template=pt.SELF_EVO_PROMPT_TEMPLATE)
collective_evo_func = CollectiveRefOperator(problem, llm_model, prompt_template=pt.COLLECTIVE_REF_PROMPT_TEMPLATE)
selector = RandomSelectOperator(problem)
replace_opr = TopKElitismReplaceOperator(problem, k=2)
```

5. Chọn cách đánh giá (Simulation hoặc Surrogate)
```python
# Nếu dùng Simulation
evaluator = SimulationBaseEvaluator(problem)

# Nếu dùng Surrogate
evaluator = EventDrivenLLMSurrogateEvaluator(llm_model, problem,
                                             prompt_template=pt.SURROGATE_PROMPT_TEMPLATE, 
                                             num_segments=4, batch_size=4,
                                             max_retries=4,
                                             scaling_schedule='linear',
                                             start_rate=0.7,
                                             max_calls_to_end=35)
```

6. Tạo Engine solve
```python 
se_engine = SelfEvoEngine(
    problem, llm_init_func, co_evo_func, self_evo_func, collective_evo_func,
    llm_crossover_func, llm_mutation_func, selector, replace_opr, evaluator,
    max_retries=3
)
```
7. Run Engine, có hai tùy chọn là run lần đầu và run tiếp tục. Nếu chọn Run tiếp tục thi phải cung cấp file checkpoint (`.pkl`)
- Run lần đầu
```python
best = se_engine.run(
    num_gen=18,
    init_size=36, subset_size=12, template_file='template.txt',
    pc=0.8, pm=0.1, state='new'
)
```
- Run lần tiếp theo
```python
best = se_engine.run(
    num_gen=18,
    init_size=36, subset_size=12, template_file='template.txt',
    pc=0.8, pm=0.1, state='resume', checkpoint_path='checkpoint.pkl'
)
```
- Có thể lưu lại state cho các lần giải sau
```python
se_engine.save_state('checkpoint.pkl')
```
8. In kết quả.
```python
if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
    print(f"Time: {se_engine.solve_time:.2f}s")
    os.makedirs('best_solution', exist_ok=True)
    best.chromosome.save(f'best_after_gen_{se_engine.gen}.py')
```
Nếu dùng surrogate, bước cuối nên đánh giá chính xác lại bằng simulation
```python
simulator = Simulator(problem)
print(simulator.simulate(best.chromosome, debug=True))
```
9. Đóng kết nối LLM (nên có)
```python
llm_model.close()
```

