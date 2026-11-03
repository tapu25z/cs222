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

**Kết quả** (Mistral-7B, 10 benchmark chưa thấy): gốc 55.8 → thêm mô tả vào prompt 60.6 → 3-shot ICL 61.0 → multi-task LoRA 66.3 → **T2L 67.7**. Chạy được với Llama-3.1-8B (73.2 → 76.9) và **Gemma-2-2B** (60.7 → 66.4). Rẻ hơn 3-shot ICL hơn 4 lần về FLOPs.

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

## 5. App: "Language-Driven Adaptation Lab"

Một **FastAPI server** + **web client**, 2 tab tương ứng 2 paper, cùng một câu hỏi: *dùng chữ để thích nghi LLM*.

### Tab 1 — "LoRA Studio" (Text-to-LoRA)
Người dùng **gõ mô tả task** → server sinh LoRA bằng T2L (vài giây) → gắn vào model gốc → **chat song song**: model gốc và model + LoRA vừa sinh.

| Tính năng | Mô tả | Liên hệ paper |
|---|---|---|
| Sinh LoRA từ mô tả | Hiện thời gian sinh, số tham số, dung lượng file adapter | Mục 3, Hình 1–2 |
| So sánh 4 cách | Model gốc · thêm mô tả vào prompt · 3-shot ICL · **T2L LoRA** trên tập con benchmark (GSM8K, ARC-Easy, BoolQ, PIQA, khoảng 100 câu mỗi bộ) | Bảng 2, Bảng 8 |
| Mô tả tốt và kém | Cùng câu hỏi, 3–4 mô tả (chi tiết, "solve this task please", chuỗi ngẫu nhiên, mô tả task khác) → so đáp án | Hình 4, Bảng 5 |
| Bản đồ LoRA | t-SNE các LoRA sinh từ mô tả của nhiều task → task giống nhau nằm gần nhau | Hình 5 |
| (Thử nghiệm của nhóm) Mô tả tiếng Việt | T2L train hoàn toàn bằng tiếng Anh; thử mô tả tiếng Việt xem LoRA còn hữu ích không | Ngoài phạm vi paper; báo cáo trung thực |

### Tab 2 — "Prompt Evolution" (GEPA)
Chọn một task → xem **prompt tiến hoá** qua từng vòng GEPA: phản hồi chữ, prompt mới, điểm tăng dần.

| Task | Dữ liệu | Chương trình DSPy | Hàm phản hồi μ_f (chữ) | Liên hệ paper |
|---|---|---|---|---|
| **Làm theo ràng buộc** | IFBench / IF-RLVR (như paper) | 2 bước: trả lời → viết lại theo ràng buộc | "Ràng buộc nào đạt, ràng buộc nào chưa đạt" | IFBench |
| **Hỏi đáp nhiều bước** | HotpotQA | 2 hop: truy vấn → tìm → truy vấn → tìm → trả lời | "Tài liệu nào còn thiếu ở hop này" | HotpotQA, Hình 2 |
| (Tuỳ chọn) Task tiếng Việt | Vd. trích xuất thông tin hoặc phân loại phản hồi sinh viên (UIT-VSFC) | 1–2 bước | "Sai nhãn nào, vì sao" | Cho thấy GEPA không phụ thuộc ngôn ngữ |

| Tính năng | Mô tả |
|---|---|
| So sánh optimizer | Baseline · MIPROv2 · **GEPA** (cùng ngân sách rollout). GRPO: **không tái hiện** (quá nặng), chiếu số của Bảng 1 |
| Đường cong điểm theo rollout | Như Hình 1 |
| Cây tiến hoá | Vẽ cây ứng viên (cha → con), tô các ứng viên trên biên Pareto, như Hình 4 |
| Xem prompt | Diff giữa prompt gốc và prompt sau mỗi vòng, kèm phản hồi đã dẫn tới thay đổi |
| Chạy prompt tối ưu | Nhập câu hỏi → chạy chương trình với prompt gốc và prompt GEPA |

### Model

| Phần | Model chính (GPU 16–24GB) | Bản nhẹ | Ghi chú |
|---|---|---|---|
| T2L | **Gemma-2-2B-it** + checkpoint `trained_t2l/gemma_2b_t2l` | — | Paper có kết quả cho Gemma (Bảng 8) |
| T2L (nâng cao) | Mistral-7B-Instruct-v0.2 + checkpoint Mistral | Mistral 4-bit | README của code: cần GPU > 16GB để chạy cùng T2L |
| GEPA, task LM | **Qwen3-8B** (như paper) qua vLLM | Qwen3-4B, hoặc Qwen3-8B lượng tử hoá AWQ | Qwen3-8B bf16 khoảng 16GB trọng số, cần khoảng 20–24GB kể cả KV cache |
| GEPA, reflection LM | Model mạnh qua API (vd. GPT-4.1 mini) | Dùng chính task LM | FAQ của GEPA: reflection LM càng mạnh càng tốt |

### Kiến trúc
```
                ┌──────────────────────── Web client ─────────────────────────┐
                │ Tab "LoRA Studio"                │ Tab "Prompt Evolution"    │
                └───────────────┬──────────────────┴─────────────┬────────────┘
                                │ HTTP                           │
                ┌───────────────▼────────────────────────────────▼────────────┐
                │ FastAPI                                                      │
                │  /t2l/generate  mô tả → T2L hypernetwork → LoRA (safetensors)│
                │  /t2l/chat      base | base + LoRA (peft, đổi adapter nóng)  │
                │  /t2l/eval      bảng so sánh đã tính sẵn + chạy nhanh        │
                │  /gepa/runs     danh sách lần chạy GEPA/MIPROv2 (JSON log)   │
                │  /gepa/tree     cây ứng viên + biên Pareto                   │
                │  /gepa/predict  chạy chương trình với prompt đã tối ưu       │
                └───────┬──────────────────────────────────────┬───────────────┘
                        │                                      │ OpenAI-compatible API
          HF transformers + peft                       vLLM (Qwen3) + API reflection LM
          Gemma-2-2B + T2L (khoảng 7–9GB)              (khoảng 10–22GB tuỳ model)
```
Hai phần **không cần chạy cùng lúc** trên một GPU: đổi model trong giờ nghỉ, hoặc chạy trên hai máy.

### Cấu trúc thư mục
Xem [`app/README.md`](../app/README.md).

---

## 6. Hạ tầng: train trước hay chạy lúc demo?

**Không cần thuê GPU cloud, và không cần tự train mô hình lớn.**

| Việc | Khi nào | Phần cứng | Ghi chú |
|---|---|---|---|
| Train T2L | **Không làm** | — | Paper: khoảng 5 ngày trên 1 H100 mỗi model. Dùng checkpoint có sẵn |
| (Mở rộng) Train vài LoRA riêng bằng `peft` trên Gemma-2-2B làm "oracle" | Trước | GPU 16GB | Để có dòng "LoRA riêng" trong bảng so sánh |
| (Mở rộng) Train T2L **reconstruction** trên 3–9 LoRA nhỏ | Trước hoặc live | GPU phổ thông | Paper: 9 LoRA mất khoảng 10 phút (trên H100); minh hoạ "hypernetwork học nén LoRA" |
| Sinh LoRA + so sánh trên benchmark con | Trước (tính bảng) + live (vài câu) | GPU 10GB | Sinh LoRA chỉ vài giây |
| Chạy GEPA và MIPROv2 đầy đủ | **Trước** (vài giờ) | GPU cho task LM + API cho reflection | Lưu log mọi ứng viên, prompt, điểm ra JSON |
| GEPA "live" | Trên sân khấu | như trên | Ngân sách nhỏ (khoảng 30–50 lần chấm), đủ cho 1–2 vòng suy ngẫm |

**VRAM ước lượng** (phải đo lại ở tuần 2):

| Thành phần | VRAM |
|---|---|
| Gemma-2-2B bf16 + gte-large + T2L (M hoặc L) | khoảng 7–9GB |
| Mistral-7B bf16 + T2L | > 16GB (theo README) |
| Qwen3-4B qua vLLM | khoảng 10–12GB |
| Qwen3-8B qua vLLM (bf16) | khoảng 20–24GB |
| Qwen3-8B AWQ 4-bit qua vLLM | khoảng 8–10GB |

**Chi phí API cho reflection LM:** paper chạy GEPA trên 6 benchmark hết khoảng 86 USD với GPT-4.1 mini (ngân sách 1.8K–7K rollout mỗi benchmark). Demo của nhóm dùng ngân sách nhỏ hơn nhiều (vài trăm lần chấm), nên chi phí ước tính vài USD. Cần đặt giới hạn chi tiêu trên tài khoản API.

**Phương án dự phòng khi thuyết trình:**
1. Máy GPU của nhóm chạy server (phương án chính).
2. Kết quả đã tính sẵn (bảng, log GEPA, cây tiến hoá) được client hiển thị **không cần GPU**.
3. Video quay sẵn từng bước.
