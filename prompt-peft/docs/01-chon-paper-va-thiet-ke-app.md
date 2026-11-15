# Chọn paper và thiết kế app — Prompting + PEFT nâng cao

> Phần lý thuyết cơ bản (prompting, LoRA, prompt tuning) do một team khác trình bày; tài liệu này chỉ tập trung vào 2 paper nâng cao và app demo.
> Lịch sử: nhóm từng chọn LoPA + BetterTogether (tài liệu còn ở `papers/reference/previous/`), sau đó đổi sang 2 paper mới hơn bên dưới. Bảng so sánh các ứng viên: [`papers/candidates/README.md`](../papers/candidates/README.md).

---

## 1. Hai paper đã chọn

| | **Paper 1 — Text-to-LoRA (T2L)** | **Paper 2 — GEPA** |
|---|---|---|
| Tên đầy đủ | *Text-to-LoRA: Instant Transformer Adaption* | *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* |
| Tác giả | Charakorn, Cetin, Tang, Lange (Sakana AI) | Agrawal, …, Potts, Zaharia, Khattab (UC Berkeley, Stanford, Databricks, MIT…) |
| Hội nghị | **ICML 2025** | **ICLR 2026, Oral** |
| Tài liệu | [`papers/01-text-to-lora/`](../papers/01-text-to-lora/) | [`papers/02-gepa/`](../papers/02-gepa/) (+ nền tảng [`papers/03-dspy/`](../papers/03-dspy/)) |
| Code | [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora), **có checkpoint đã train** | `pip install gepa`, có trong DSPy (`dspy.GEPA`) |
| Đầu vào | **Mô tả task bằng chữ** | **Trace + phản hồi bằng chữ** từ các lần chạy |
| Đầu ra | **LoRA adapter** (trọng số) | **Prompt** mới (câu lệnh) |
| Câu hỏi chính | Có thể sinh LoRA cho task mới chỉ từ một câu mô tả, không cần train? | Tối ưu prompt bằng suy ngẫm ngôn ngữ có thắng được fine-tune LoRA bằng RL không? |

**Vì sao ghép 2 paper này.** Cả hai đều dùng **ngôn ngữ tự nhiên để thích nghi LLM**, nhưng theo hai hướng ngược nhau:
- **T2L: Prompt → PEFT.** Viết một câu mô tả, hypernetwork biến nó thành **trọng số LoRA**.
- **GEPA: PEFT → Prompt.** Thay vì fine-tune trọng số (GRPO + LoRA), **tối ưu prompt** bằng cách cho LLM tự đọc lỗi và sửa câu lệnh, và thắng.

Câu hỏi xuyên suốt buổi thuyết trình: **"Khi nào nên đưa kiến thức vào trọng số, khi nào nên đưa vào prompt?"**

Cả hai đều mới (2025–2026), ở hội nghị top, có code chính thức, và **không cần tự train mô hình lớn** để demo: T2L có checkpoint sẵn; GEPA chỉ gọi LLM.

---

## 2. Paper 1 — Text-to-LoRA trong 1 trang

**Vấn đề.** Mỗi task cần một LoRA riêng: gom dữ liệu, chọn siêu tham số, train lâu.

**Ý tưởng.** Train một **hypernetwork** `h_θ` sinh trọng số LoRA:
```
ΔW_{m,l} = h_θ( concat[ f(mô tả task), E[module m], E[layer l] ] )
```
- `f`: embedding của mô tả (gte-large-en-v1.5). `E`: embedding học được của loại module (q_proj, v_proj) và chỉ số layer.
- Gộp mọi (m, l) vào một batch → **toàn bộ LoRA sinh ra trong một lần forward** (vài giây).
- Train bằng **SFT trên 479 task** (Super-NaturalInstructions), mỗi task 128 mô tả do GPT-4o-mini viết. Train bằng reconstruction (bắt chước LoRA có sẵn) thì **không khái quát** được.

**Kết quả** (Mistral-7B, 10 benchmark chưa thấy): gốc 55.8 → thêm mô tả vào prompt 60.6 → 3-shot ICL 61.0 → multi-task LoRA 66.3 → **T2L 67.7**. Phép so sánh cho thấy giá trị của mô tả là **T2L 67,7 so với multi-task LoRA 66,3** (cùng 479 task, không dùng mô tả); không so 67,7 với 60,6 vì T2L còn được train trên 479 task. Chạy được với Llama-3.1-8B (73.2 → 76.9) và **Gemma-2-2B** (60.7 → 66.4). Rẻ hơn 3-shot ICL hơn 4 lần về FLOPs.

**Điểm "wow" cho demo**: mô tả khác nhau → LoRA khác nhau → **phong cách trả lời khác nhau** (Hình 4); mô tả kém ("solve this task please") → không giúp.

## 3. Paper 2 — GEPA trong 1 trang

**Vấn đề.** RL (GRPO) để thích nghi LLM cần **hàng chục nghìn rollout**, mỗi rollout chỉ cho **một con số thưởng**.

**Ý tưởng.** Trace của hệ LLM là **văn bản**; hãy để LLM **đọc và suy ngẫm** về nó. GEPA tối ưu **chỉ prompt** bằng:
1. **Vòng tiến hoá**: pool ứng viên → chọn → đột biến (hoặc Merge) → thử trên minibatch 3 ví dụ → giữ nếu tốt hơn.
2. **Đột biến bằng suy ngẫm**: hàm phản hồi μ_f trả về **điểm + phản hồi chữ** (vd. "ràng buộc X chưa đạt", "còn thiếu tài liệu Y"); reflection LM đọc rồi **viết lại câu lệnh** của một module.
3. **Chọn theo Pareto**: giữ mọi ứng viên **thắng ít nhất một ví dụ**, tránh kẹt cực trị địa phương.

**Kết quả** (Qwen3-8B, 6 benchmark): baseline 45.23 → GRPO (LoRA, 24.000 rollout) 48.91 → MIPROv2 47.84 → **GEPA 54.85**, chỉ dùng 1.839–7.051 rollout. Với GPT-4.1 mini: 53.03 → **GEPA+Merge 66.36**. Chi phí GEPA cho 6 benchmark khoảng 86 USD.

---

## 4. DSPy là gì? (kiến thức cần trước GEPA)

> Paper: *DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines* — Khattab et al., **ICLR 2024**. Thư viện: `pip install dspy`, tài liệu tại https://dspy.ai.
> GEPA là một **optimizer của DSPy** (`dspy.GEPA`), và baseline chính của GEPA (MIPROv2) cũng là optimizer DSPy. Hiểu DSPy trước thì phần GEPA dễ theo hơn nhiều.

### 4.1 Vấn đề DSPy giải quyết
Ứng dụng LLM thật thường là **pipeline nhiều bước** (ví dụ: sinh truy vấn → tìm tài liệu → tóm tắt → trả lời). Cách làm truyền thống:
- Mỗi bước có một **prompt viết tay dài**, phải chỉnh bằng cách thử–sai.
- Đổi model (GPT → Llama) hoặc đổi một bước thì các prompt khác có thể hỏng, phải chỉnh lại từ đầu.
- Ví dụ few-shot phải tự chọn bằng tay cho từng bước.

**Ý tưởng DSPy**: *"Programming — not prompting — LMs"*. Ta **viết chương trình** mô tả *cần làm gì*. Còn *prompt cụ thể trông thế nào* (câu lệnh, ví dụ few-shot, thậm chí trọng số model) thì để **optimizer tự tìm** dựa trên dữ liệu và một metric.

Có thể so sánh với PyTorch:

| PyTorch | DSPy |
|---|---|
| Layer (`nn.Linear`) | **Module** (`dspy.Predict`, `dspy.ChainOfThought`) |
| Kiểu input/output của layer | **Signature** (`"question -> answer"`) |
| Model = ghép nhiều layer trong `forward()` | **Program** = ghép nhiều module trong `forward()` |
| Loss function | **Metric** (vd. đáp án có khớp không) |
| Optimizer (Adam) cập nhật **trọng số** | **Optimizer** (trước đây gọi là *teleprompter*) cập nhật **prompt / ví dụ few-shot / trọng số** |
| `model.fit()` | `optimizer.compile(program, trainset=...)` |

### 4.2 Năm khái niệm chính

**1. Signature** — khai báo input → output, không viết prompt:
```python
import dspy
lm = dspy.LM("openai/Qwen2.5-7B-Instruct", api_base="http://localhost:8000/v1", api_key="local")  # vLLM local
dspy.configure(lm=lm)

class SolveMath(dspy.Signature):
    """Giải bài toán có lời văn."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="chỉ một con số")
```

**2. Module** — chiến lược gọi LLM với một signature:
```python
solver = dspy.ChainOfThought(SolveMath)     # tự thêm bước "suy luận" trước đáp án
pred = solver(question="Roger có 5 quả bóng, mua thêm 2 hộp, mỗi hộp 3 quả...")
print(pred.reasoning, pred.answer)           # "... 5 + 6 = 11", "11"
```
Các module hay dùng: `dspy.Predict` (gọi thẳng), `dspy.ChainOfThought` (CoT), `dspy.ReAct` (agent dùng tool).

**3. Program** — ghép nhiều module bằng code Python bình thường:
```python
class MultiHopQA(dspy.Module):
    def __init__(self):
        self.gen_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(2)]
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        for gq in self.gen_query:                       # 2 hop
            query = gq(context=context, question=question).search_query
            context += search(query, k=3)              # retriever (BM25)
        return self.answer(context=context, question=question)
```

**4. Metric** — hàm chấm điểm, chỉ cần nhãn **cuối cùng**:
```python
def exact_match(example, pred, trace=None):
    return example.answer.strip().lower() == pred.answer.strip().lower()
```

**5. Optimizer + compile** — tự tìm prompt tốt nhất:
```python
opt = dspy.BootstrapFewShotWithRandomSearch(metric=exact_match,
                                            max_bootstrapped_demos=3, num_candidate_programs=6)
compiled = opt.compile(MultiHopQA(), trainset=train)
compiled.save("hotpot_prompt_opt.json")        # lưu lại, khi demo chỉ cần load
```

### 4.3 Bên trong optimizer làm gì? (cầu nối sang GEPA)
`BootstrapFewShot`:
1. Chạy program trên các câu hỏi trong tập train.
2. Ghi lại **trace**: input/output của *từng* module ở mỗi lần chạy.
3. Lần chạy nào cho **đáp án cuối đúng** (theo metric) → coi các bước trung gian của nó là "tốt" → dùng làm **ví dụ few-shot** cho từng module.
4. Biến thể `...WithRandomSearch` thử nhiều tổ hợp ví dụ và chọn tổ hợp tốt nhất trên tập dev.

→ Đây là cách DSPy **tự tạo nhãn cho các bước trung gian** mà không cần con người gán nhãn.

Các optimizer khác: `MIPROv2` (tối ưu cả câu lệnh lẫn ví dụ), `BootstrapFinetune` (dùng chính các trace đúng đó để **fine-tune trọng số** thay vì làm ví dụ few-shot).

**GEPA làm khác**: thay vì chọn *ví dụ* từ trace đúng, GEPA đưa trace **kèm phản hồi dạng chữ** (vì sao sai) cho một LLM để **viết lại câu lệnh** của từng module:
```python
optimizer = dspy.GEPA(metric=metric_with_feedback,      # trả về điểm + feedback (chữ)
                      max_metric_calls=150,              # ngân sách rollout
                      reflection_lm=dspy.LM("openai/gpt-4.1-mini"))
optimized = optimizer.compile(MultiHopQA(), trainset=train, valset=val)
```

### 4.4 Gợi ý cách trình bày (khoảng 10 phút)
1. Một pipeline viết prompt tay, đổi model thì hỏng (1 slide).
2. Bảng so sánh PyTorch ↔ DSPy (1 slide).
3. Live code: signature + `ChainOfThought` cho GSM8K, dùng `dspy.inspect_history()` để **in ra prompt thật** mà DSPy sinh ra.
4. Live code: program multi-hop, chỉ ra "không có nhãn cho bước sinh truy vấn".
5. Hình minh hoạ: bootstrap trace → ví dụ few-shot (DSPy gốc, MIPROv2) và trace + phản hồi → viết lại câu lệnh (GEPA) → dẫn sang paper.

Lưu ý: API của DSPy thay đổi khá nhanh giữa các phiên bản (vd. `dspy.OpenAI` cũ → `dspy.LM`, `rationale` → `reasoning`). Cần cố định phiên bản trong `requirements.txt` và kiểm tra lại code mẫu ở trên với phiên bản đó.

---

## 5. App, hạ tầng và thí nghiệm so sánh

Đã chuyển sang tài liệu riêng: [`04-thiet-ke-thi-nghiem-va-app.md`](04-thiet-ke-thi-nghiem-va-app.md). Tóm tắt:
- Chạy **cả hai pipeline trên cùng một model** (Llama-3.1-8B-Instruct, vì T2L có checkpoint) và **cùng 6 dataset**: ARC-Challenge, HellaSwag, BoolQ, HumanEval, AIME 2025, GSM8K.
- Một harness đánh giá chung; 6 phương án bắt buộc (gốc, mô tả trong prompt, 3-shot ICL, T2L, GEPA, T2L + GEPA) và 2 phương án mở rộng.
- App demo 3 tab (LoRA Studio, Prompt Evolution, Đấu trường) đọc kết quả từ harness.
- Không tự train T2L: dùng checkpoint có sẵn. Paper không ghi thời gian train SFT (Phụ lục H chỉ nói mọi model vừa một GPU H100 80GB).
