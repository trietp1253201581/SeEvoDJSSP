# SeEvoDJSSP
## Cấu trúc mã nguồn
* [model.py](./model.py): Chứa các mô hình đại diện cho các HDR  hiện đang dùng CodeSegmentHDR.
* [problem.py](./problem.py): Chứa các mô hình về bài toán lập lịch động, bao gồm Operation, Job, Machine và các Terminal được sử dụng.
* [simulate.py](./simulate.py): Chứa Simulator để chạy mô phỏng với 1 HDR và 1 Problem.
* [llm.py](./llm.py): Chứa các lớp và phương thức để gọi và lấy response từ LLM (đang dùng LLM trên OpenRouter, có các bản free)
* [basic_evo.py](./basic_evo.py): Chứa các lớp cơ bản của tính toán tiến hóa như Individual, Population và định
nghĩa interface Operator (các toán tử tiến hóa được sử dụng)
* [se_evo.py](./se_evo.py): Triển khai các toán tử Se-Evo, phần chính, chứa các LLM-Base Operator và các Reflection Operator.
* [prompt_template.py](./prompt_template.py): Chứa các string template của các prompt được sử dụng trong các LLM-Base Operator
* [template.txt](./template.txt): Chứa template của 1 code segment nên được trả về từ LLM.
## Chạy chương trình
Chạy file `main.py`.
```bash
python -u main.py
```

Hiện nay chỉ mới đang test với 1 problem tạo ngẫu nhiên với 5 máy, 10 job như sau:
```txt
Job(id=0, status=Status.ARRIVED, time_arr=327, next_opr=0, oprs=[Operation(deadline=26, available_machines=[1:105.0, 0:559.0, 2:90.0, ])])
Job(id=1, status=Status.ARRIVED, time_arr=302, next_opr=0, oprs=[Operation(deadline=33, available_machines=[0:224.0, ]), Operation(deadline=272, available_machines=[1:666.0, ]), Operation(deadline=991, available_machines=[1:164.0, 3:433.0, 2:349.0, 0:285.0, ]), Operation(deadline=1151, available_machines=[2:95.0, 0:390.0, ])])
Job(id=2, status=Status.ARRIVED, time_arr=49, next_opr=0, oprs=[Operation(deadline=353, available_machines=[0:550.0, 2:128.0, 1:388.0, ]), Operation(deadline=434, available_machines=[2:72.0, 3:47.0, 0:234.0, ]), Operation(deadline=731, available_machines=[1:104.0, ])])   
Job(id=3, status=Status.ARRIVED, time_arr=194, next_opr=0, oprs=[Operation(deadline=465, available_machines=[1:215.0, 3:274.0, 2:664.0, ]), Operation(deadline=539, available_machines=[1:474.0, 0:389.0, ]), Operation(deadline=816, available_machines=[2:235.0, 0:33.0, ])]) 
Job(id=4, status=Status.ARRIVED, time_arr=412, next_opr=0, oprs=[Operation(deadline=411, available_machines=[0:218.0, 3:512.0, 1:406.0, ]), Operation(deadline=1070, available_machines=[1:575.0, 3:552.0, 0:270.0, 2:599.0, ]), Operation(deadline=1509, available_machines=[2:94.0, 0:49.0, 3:113.0, 1:157.0, ])])
Job(id=5, status=Status.ARRIVED, time_arr=321, next_opr=0, oprs=[Operation(deadline=697, available_machines=[0:542.0, 1:258.0, 2:567.0, 3:12.0, ]), Operation(deadline=1394, available_machines=[2:657.0, ])])
Job(id=6, status=Status.ARRIVED, time_arr=174, next_opr=0, oprs=[Operation(deadline=301, available_machines=[1:513.0, 3:183.0, 0:520.0, 2:109.0, ])])
Job(id=7, status=Status.ARRIVED, time_arr=445, next_opr=0, oprs=[Operation(deadline=655, available_machines=[1:166.0, 3:553.0, ]), Operation(deadline=1199, available_machines=[2:501.0, ]), Operation(deadline=1219, available_machines=[2:315.0, ])])
Job(id=8, status=Status.ARRIVED, time_arr=122, next_opr=0, oprs=[Operation(deadline=247, available_machines=[0:498.0, ])])
Job(id=9, status=Status.ARRIVED, time_arr=417, next_opr=0, oprs=[Operation(deadline=546, available_machines=[1:487.0, 2:563.0, ])])  
```

Vì chưa có nhiều đụng độ phức tạp, LLM khi init lời giải đã đạt được makespan khá tốt (2730) ngay trong lần đầu
với một HDR khá đơn giản
```python
js + jw - jwt + 0.5 * util
```

**Lưu ý**:
Khi chạy thử, cần lấy Provision key từ OpenRouter, có thể đăng nhập và lấy từ link [Provision key Open Router](https://openrouter.ai/settings/provisioning-keys).

Sau đó, hãy thêm key này vào file `config.json` (tạo trong thư mục gốc dự án), thêm trường như sau:
```json
{
    "OPEN_ROUTER_PROVISION_KEY": "Provison key lấy được"
}
```

Notes: Nên chọn model Quasar-Alpha của OpenRouter, thời gian phản hồi nhanh và free không giới hạn.