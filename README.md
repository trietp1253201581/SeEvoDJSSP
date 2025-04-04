# SeEvoDJSSP
## Cấu trúc mã nguồn
* [model.py](./model.py): Chứa các mô hình đại diện cho các HDR  hiện đang dùng CodeSegmentHDR.
* [problem.py](./problem.py): Chứa các mô hình về bài toán lập lịch động, bao gồm Operation, Job, Machine và các Terminal được sử dụng.
* [simulate.py](./simulate.py): Chứa Simulator để chạy mô phỏng với 1 HDR và 1 Problem.
* [llm.py](./llm.py): Chứa các lớp và phương thức để gọi và lấy response từ LLM (đang dùng LLM trên OpenRouter, có các bản free)
* [base_evo.py](./base_evo.py): Chứa các lớp cơ bản của tính toán tiến hóa như Individual, Population và định
nghĩa interface Operator (các toán tử tiến hóa được sử dụng)
* [se_evo.py](./se_evo.py): Triển khai các toán tử Se-Evo, phần chính, chứa các LLM-Base Operator và các Reflection Operator.
* [prompt_template.py](./prompt_template.py): Chứa các string template của các prompt được sử dụng trong các LLM-Base Operator
* [template.txt](./template.txt): Chứa template của 1 code segment nên được trả về từ LLM.
## Chạy chương trình
Do mã nguồn chưa hoàn thiện, nên có thể thử chạy 1 vài module trước.

Hiện cũng đã cung cấp 1 vài hàm test:
1. Hàm test_simulator() trong [simulate.py](./simulate.py) để test bộ mô phỏng
2. Hàm test_llm_init() và test_co_evo() trong [se_evo.py](./se_evo.py) để test 2 toán tử khởi tạo LLM và Co-Evo Reflection.

**Lưu ý**:
Khi chạy thử, cần lấy Provision key từ OpenRouter, có thể đăng nhập và lấy từ link [Provision key Open Router](https://openrouter.ai/settings/provisioning-keys).

Sau đó, hãy thêm key này vào file `config.json` (tạo trong thư mục gốc dự án), thêm trường như sau:
```json
{
    "OPEN_ROUTER_PROVISION_KEY": "Provison key lấy được"
}
```