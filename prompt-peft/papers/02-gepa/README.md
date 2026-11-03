# GEPA — Reflective Prompt Evolution Can Outperform Reinforcement Learning

| | |
|---|---|
| Tên đầy đủ | *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* |
| Tác giả | Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, Omar Khattab (UC Berkeley, Stanford, BespokeLabs.ai, Notre Dame, Databricks, MIT) |
| Hội nghị | **ICLR 2026, Oral** |
| Link | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) · [arXiv 2507.19457](https://arxiv.org/abs/2507.19457) · PDF trong repo (bản ICLR): [`gepa-iclr2026.pdf`](gepa-iclr2026.pdf) |
| Code | [gepa-ai/gepa](https://github.com/gepa-ai/gepa) (`pip install gepa`, MIT) · trong DSPy: `dspy.GEPA` |
| Cần đọc trước | [`../03-dspy/`](../03-dspy/) (GEPA tối ưu chương trình DSPy, so với MIPROv2 của DSPy) |
| Dịch chi tiết | [`dich-chi-tiet.md`](dich-chi-tiet.md) |

---

## Tóm tắt 1 trang

**Vấn đề.** Cách phổ biến để thích nghi LLM cho task mới là **RL**, ví dụ **GRPO**: chạy hệ thống, nhận **một con số thưởng** (đúng/sai), rồi cập nhật trọng số. Cách này thường cần **hàng nghìn tới hàng chục nghìn lượt chạy (rollout)**. Rất tốn khi mỗi lượt gọi công cụ đắt, hoặc khi không fine-tune được model (model đóng, quá lớn).

**Ý tưởng.** Mỗi lượt chạy của một hệ thống LLM có thể viết ra thành **văn bản**: chuỗi suy luận, lời gọi công cụ, thông báo lỗi của bộ chấm điểm. LLM hiện đại **đọc hiểu** được những thứ này. Vậy thay vì nén cả lượt chạy thành một con số thưởng, hãy cho LLM **suy ngẫm (reflect) bằng ngôn ngữ** về lượt chạy để rút ra bài học và **sửa prompt**. Ngôn ngữ là "tín hiệu học" giàu thông tin hơn nhiều.

**GEPA (Genetic-Pareto)** tối ưu **chỉ prompt** (trọng số giữ nguyên), gồm 3 thành phần:
1. **Tiến hoá prompt kiểu di truyền**: giữ một *pool* ứng viên; mỗi vòng chọn một ứng viên, tạo **con** bằng đột biến (sửa prompt) hoặc lai ghép (*Merge*), giữ lại nếu tốt hơn.
2. **Đột biến bằng suy ngẫm (reflective mutation)**: chạy ứng viên trên một minibatch nhỏ (3 ví dụ), lấy **trace + điểm + phản hồi dạng chữ** từ *feedback function* μ_f (vd. thông báo lỗi biên dịch, ràng buộc nào bị vi phạm, tài liệu nào còn thiếu), đưa cho một **reflection LM** để viết lại câu lệnh của một module.
3. **Chọn ứng viên theo Pareto**: không luôn chọn ứng viên có điểm trung bình cao nhất (dễ kẹt cực trị địa phương). Thay vào đó giữ mọi ứng viên **thắng ít nhất một ví dụ**, rồi chọn ngẫu nhiên có trọng số theo số ví dụ mà nó thắng.

**Kết quả chính** (6 benchmark: HotpotQA, IFBench, HoVer, PUPA, AIME-2025, LiveBench-Math).

| Qwen3-8B | Tổng hợp | Cải thiện | Số rollout |
|---|---|---|---|
| Baseline | 45.23 | — | — |
| GRPO (LoRA) | 48.91 | +3.68 | **24.000** mỗi benchmark |
| MIPROv2 | 47.84 | +2.61 | |
| **GEPA** | **54.85** | **+9.62** | 1.839–7.051 |
| GEPA+Merge | 52.40 | +7.17 | |

| GPT-4.1 mini | Tổng hợp | Cải thiện |
|---|---|---|
| Baseline | 53.03 | — |
| MIPROv2 | 58.67 | +5.64 |
| TextGrad | 59.14 | +6.11 |
| GEPA | 65.22 | +12.19 |
| **GEPA+Merge** | **66.36** | **+13.33** |

- **Hơn GRPO** tới 19 điểm (trung bình 6), dùng **ít rollout hơn tới 35 lần**. Thắng GRPO ở 5/6 benchmark (thua AIME). Muốn đạt mức validation tốt nhất của GRPO, GEPA chỉ cần 243–1.179 rollout (hiệu quả mẫu cao hơn tới 78 lần).
- **Hơn MIPROv2** (tối ưu cả câu lệnh lẫn ví dụ few-shot) hơn 10 điểm, vd. +12 trên AIME-2025.
- Prompt GEPA **ngắn hơn tới 9.2 lần** so với MIPROv2, vì chỉ có câu lệnh, không có ví dụ.
- **Chuyển model**: prompt tối ưu trên Qwen3-8B, đem dùng cho GPT-4.1 mini vẫn +9 điểm.
- **Chi phí**: toàn bộ thí nghiệm GPT-4.1 mini dưới 500 USD; riêng GEPA khoảng 86 USD cho 6 benchmark.
- Ứng dụng thêm: tìm kiếm lúc suy luận cho **sinh kernel NPU/CUDA**, và tìm **prompt đối kháng** (AIME pass@1 từ 76% xuống 10%).

**Hạn chế/lưu ý.** Phần lớn ngân sách rollout dùng cho **validation** (tập Pareto). Merge giúp với GPT-4.1 mini nhưng có lúc làm **giảm** với Qwen3-8B. Không thay đổi trọng số, nên không "dạy" được kiến thức mới mà model không có.

---

## Thông số (để chạy demo và tái hiện)

| Mục | Giá trị trong paper |
|---|---|
| Model | Qwen3-8B (temperature 0.6, top-p 0.95, top-k 20); GPT-4.1 mini (temperature 1.0); context 16.384 token |
| Chia dữ liệu | HotpotQA, IFBench, HoVer: 150 train / 300 val / 300 test; PUPA 111/111/221; AIME: 90 câu 2022–2024 chia train/val, test là AIME-2025 (×5 lần); LiveBench-Math 368 câu chia 3 |
| GEPA | minibatch **3**; Merge tối đa **5** lần; ngân sách rollout **bằng với MIPROv2** của từng benchmark |
| MIPROv2 | `auto=heavy`: 18 câu lệnh ứng viên + 18 bộ few-shot; 2.270–6.926 rollout |
| GRPO | LoRA rank 16, α 64, dropout 0.05, bf16, lr 1e-5, 500 bước = **24.000 rollout**; 1× H100/A100 80GB + GPU riêng để sinh rollout |
| Meta-prompt suy ngẫm | Phụ lục C (đã dịch trong `dich-chi-tiet.md`) |

## Điểm cần nhấn khi thuyết trình
1. **Hình 1**: đường GEPA vọt lên sau vài trăm rollout, còn GRPO bò chậm tới 24.000. Đây là "hình kể chuyện" của cả paper.
2. **Hình 2**: prompt gốc chỉ một dòng, prompt sau GEPA thành một bản hướng dẫn chi tiết với "bài học rút ra". Đây là prompting, nhưng **do máy viết**.
3. **Hình 3 + Thuật toán 1**: vòng lặp chọn ứng viên → chạy minibatch → suy ngẫm → sửa → giữ nếu tốt hơn.
4. **Hình 4 + Bảng 3**: chọn theo Pareto so với luôn chọn tốt nhất (kẹt sau 1 bước).
5. Liên hệ đề tài **Prompting + PEFT**: GRPO trong paper dùng **LoRA** (một PEFT). Kết luận là tối ưu **prompt** bằng ngôn ngữ có thể **thắng** fine-tune **LoRA** bằng RL, với ít dữ liệu và compute hơn.
6. Câu hỏi phản biện nên chuẩn bị:
   - Reflection LM có cần mạnh hơn task LM không? (Paper dùng cùng model; FAQ của GEPA khuyên reflection LM càng mạnh càng tốt.)
   - GEPA có thua ở đâu? (AIME với Qwen3-8B, GRPO 38.00 so với GEPA 32.00.)
   - Khi model không biết kiến thức cần thiết thì sửa prompt có đủ không?
