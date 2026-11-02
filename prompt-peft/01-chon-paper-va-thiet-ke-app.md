# Chọn paper và thiết kế app — Prompting + PEFT nâng cao

> Phần lý thuyết cơ bản (prompting, LoRA, prompt tuning) do một team khác trình bày; tài liệu này chỉ tập trung vào 2 paper nâng cao và app demo.

---

## 1. Hai paper đã chọn

| | **Paper 1 — LoPA** | **Paper 2 — BetterTogether** |
|---|---|---|
| Tên đầy đủ | *Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation* | *Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together* |
| Tác giả | Jain, Chaudhuri, Reps, Jermaine | Soylu, Potts, Khattab (Stanford, nhóm DSPy) |
| Hội nghị | **NeurIPS 2024** | **EMNLP 2024** (main, 2024.emnlp-main.597) |
| File | `paper-advance/lopa-neurips2024.pdf` | *chưa có trong repo* → tải từ https://aclanthology.org/2024.emnlp-main.597 |
| Loại prompt | **Soft prompt** (vector do máy học) | **Hard prompt** (prompt chữ + ví dụ few-shot, do máy tối ưu) |
| Loại PEFT | Prompt tuning nâng cấp, so với LoRA | **LoRA** fine-tune trọng số |
| Câu hỏi chính | Prompt tuning có thể mạnh ngang LoRA mà server không phải lưu adapter không? | Nên tối ưu prompt **hay** fine-tune trọng số? → Làm **cả hai**, xen kẽ |

**Vì sao ghép 2 paper này**: đề tài là *Prompting + PEFT*.
- LoPA: *prompt học dưới dạng vector*, PEFT ở phía **input**.
- BetterTogether: *prompt chữ được tối ưu tự động* + *LoRA trên trọng số*, áp dụng cho **pipeline nhiều bước** (CoT, hỏi đáp nhiều bước).

Hai paper phủ đủ hai nửa của đề tài và đi từ task đơn giản (phân loại, một bước) tới task phức tạp (suy luận nhiều bước).

Các paper đã cân nhắc nhưng không chọn: POST (ICML 2025, khó demo vì phải distill), PARA (sửa mọi tầng ở server, ngược thông điệp của LoPA), S-LoRA (MLSys 2024, thiên về hệ thống), Gist Tokens (NeurIPS 2023), DePT (ICLR 2024, quá giống LoPA — dùng làm baseline), DoRA (ICML 2024, thuần PEFT).

---

## 2. Paper 1 — LoPA trong 1 trang

**Vấn đề**: LoRA mạnh nhưng mỗi task cần một adapter lưu và chạy **trên server**. Prompt tuning thì không cần (chỉ là prefix của input) nhưng **yếu hơn**.

**Ý tưởng**: soft prompt `Z` (m vector × d chiều) gồm 2 phần:
```
Z  =  Z_S  ⊙  sigmoid( u · vᵀ )
      ^^^      ^^^^^^^^^^^^^^^^
  phần chung   phần riêng cho từng input (Z_I), dạng low-rank hạng r:
  của task     u = MLP_u(X'),  v = MLP_v(X'),  X' = biểu diễn input từ một encoder nhỏ
```
`concat(Z, embedding(input))` → LLM **freeze**. Chỉ train `Z_S`, `MLP_u`, `MLP_v`.
- Cổng sigmoid + nhân từng phần tử → hai phần học phụ thuộc nhau (tốt hơn cộng `Z_S + Z_I` như IDPG).
- Trong paper: m = 10, encoder RoBERTa (NLU) / CodeSage (code) + MLP; ngang FFT và LoRA trên GLUE, thắng LoRA 11/24 trường hợp.

## 3. Paper 2 — BetterTogether trong 1 trang

**Bối cảnh**: một "LM program" (DSPy) gồm nhiều module gọi LLM, ví dụ: *sinh truy vấn → tìm tài liệu → sinh truy vấn 2 → tìm → trả lời* (HotPotQA). Chỉ có nhãn cho **đáp án cuối**, **không có nhãn cho các bước trung gian**, và không lan truyền gradient qua cả pipeline được.

**Hai cách tối ưu**:
- *Prompt optimization* (DSPy `BootstrapFewShotWithRandomSearch`): chạy chương trình trên tập train, giữ lại các "vết" (trace) cho ra đáp án đúng, dùng chúng làm ví dụ few-shot cho từng module; thử nhiều tổ hợp và chọn cái tốt nhất trên tập validation.
- *Weight fine-tuning* (DSPy `BootstrapFinetune`): dùng chính các trace đúng đó làm dữ liệu train LoRA cho LLM → **model tự dạy chính nó**, không cần nhãn trung gian.

**BetterTogether** = xen kẽ hai bước: **tối ưu prompt → fine-tune trọng số (LoRA) trên trace của prompt đã tối ưu → tối ưu prompt lần nữa**.

**Kết quả trong paper**: HotPotQA, GSM8K, Iris với Mistral-7B, Llama-2-7B, Llama-3-8B; kết hợp tốt nhất ở 7/9 cặp dataset × model, hơn "chỉ fine-tune" tới 60% và hơn "chỉ tối ưu prompt" tới 6% (trung bình theo model và task).

*Cần đọc kỹ trong paper trước khi làm: thứ tự và số vòng chính xác, cấu hình LoRA (rank, lr, số epoch), kích thước train/dev, retriever dùng cho HotPotQA.*

---

## 4. App: "One Model, Many Skills"

Một **FastAPI server** + **web client**. Cùng một app, hai phần tương ứng hai paper.

### Phần 1 — LoPA skills (task một bước)
| Skill | Ngôn ngữ | Dataset | Metric | Encoder cho Z_I |
|---|---|---|---|---|
| Cảm xúc review | Anh | SST-2 | Accuracy | `xlm-roberta-base` |
| Cảm xúc phản hồi sinh viên | Việt | UIT-VSFC | Accuracy, macro-F1 | `xlm-roberta-base` |
| Dự đoán output code | Python | CruxEval-O (800 mẫu, tự chia train/test, seed cố định) | Exact match | CodeSage-small |

So sánh: Zero-shot · Few-shot · Prompt Tuning · LoRA · **LoPA** (accuracy, số tham số train, dung lượng file, server có phải load adapter không).

### Phần 2 — BetterTogether skills (pipeline nhiều bước)
| Skill | Dataset | DSPy program | Metric |
|---|---|---|---|
| Giải toán có lời văn | GSM8K | `ChainOfThought("question -> answer")` | Exact match |
| Hỏi đáp nhiều bước | HotPotQA | 2 hop: `GenerateQuery → Retrieve → GenerateQuery → Retrieve → Answer` | Exact match |
| (Tuỳ chọn) Toán tiếng Việt | Bản dịch một phần GSM8K, hoặc ViMMRC | như GSM8K | Exact match |

So sánh (giống bảng chính của paper): Zero-shot · Chỉ tối ưu prompt · Chỉ LoRA · **Prompt → LoRA → Prompt**.
Client hiển thị thêm **trace từng bước** (truy vấn sinh ra, tài liệu tìm được, lời giải CoT) để khán giả thấy prompt đã được tối ưu trông như thế nào.

Retriever cho HotPotQA: để chạy offline và gọn, dùng BM25 trên các đoạn văn có sẵn trong HotPotQA (distractor setting) thay vì index toàn bộ Wikipedia. Đây là điểm khác so với paper; cần ghi rõ trong slide.

### Backbone
| Phần | Model chính (GPU riêng) | Bản nhẹ (Colab/Kaggle T4) |
|---|---|---|
| LoPA | `Qwen2.5-1.5B` | `Qwen2.5-0.5B` |
| BetterTogether | `Qwen2.5-7B-Instruct` hoặc `Llama-3.1-8B-Instruct` (QLoRA 4-bit) | `Qwen2.5-1.5B/3B-Instruct` |

Qwen hỗ trợ tiếng Việt; Llama-3 gần với setup của paper hơn. Chọn trong `config.yaml`.

### Kiến trúc
```
             ┌──────────── Web client ─────────────┐
             │ chọn skill · nhập câu · bảng so sánh │
             │ · xem trace từng bước                │
             └───────────────┬──────────────────────┘
                             │ HTTP
             ┌───────────────▼──────────────────────────────────────────────┐
             │ FastAPI                                                       │
             │  /lopa/predict   : encoder + MLP → Z → LLM nhỏ (freeze)       │
             │  /bt/run         : chạy DSPy program đã compile (JSON)        │
             │                    → gọi LLM endpoint (base hoặc + LoRA)      │
             └───────────────┬───────────────────────────┬──────────────────┘
                             │                           │ OpenAI-compatible API
                      LLM nhỏ (HF, freeze)         vLLM (GPU) / Ollama (CPU)
                                                   backbone 7–8B + LoRA adapter
```

### Cấu trúc thư mục dự kiến
```
prompt-peft/app/
  config.yaml
  data/                    # script tải và chia dữ liệu
  lopa/       lopa.py, train.py, evaluate.py
  bettertogether/
              programs.py   # DSPy program cho GSM8K, HotPotQA
              optimize.py   # chỉ prompt / chỉ LoRA / prompt→LoRA→prompt
              evaluate.py
  artifacts/  # checkpoint LoPA, LoRA adapter, program đã compile (.json)
  server/main.py
  client/
  notebooks/  part1_lopa.ipynb, part2_bettertogether.ipynb
```

---

## 5. Hạ tầng demo: train trước, khi demo có cần GPU không?

**Không cần thuê hay host GPU trên cloud.** Phần tốn GPU là *train/tối ưu*, làm **trước** buổi thuyết trình. Khi demo chủ yếu chỉ chạy *inference*.

| Việc | Khi nào | Phần cứng |
|---|---|---|
| Train PT, LoRA, LoPA (Phần 1) | Trước | GPU riêng / Kaggle |
| DSPy tối ưu prompt (hàng nghìn lần gọi LLM) + fine-tune LoRA (Phần 2) | Trước | GPU riêng (7–8B) |
| Lưu kết quả: checkpoint LoPA, LoRA adapter, program `.json`, bảng kết quả | Trước | — |
| Demo inference LoPA (model 1.5B, phân loại chỉ sinh 1 token) | Trên sân khấu | **Laptop CPU là đủ**; GPU thì nhanh hơn |
| Demo inference BetterTogether (7–8B, sinh lời giải dài) | Trên sân khấu | **GPU riêng** của nhóm, chạy local và nối laptop qua mạng LAN hoặc SSH tunnel |
| "Live training" | Trên sân khấu | Chỉ chạy 50–100 bước cho thấy loss giảm, rồi load checkpoint đã train sẵn |

Phương án dự phòng (theo thứ tự):
1. Máy GPU riêng của nhóm chạy server (phương án chính).
2. Laptop CPU: LoPA chạy bình thường; BetterTogether dùng bản đã merge LoRA và lượng tử hoá GGUF Q4 (khoảng 5GB) chạy bằng Ollama/llama.cpp. Chậm nhưng đủ cho 1–2 câu hỏi demo.
3. Kết quả đã tính sẵn (bảng, trace) + video quay sẵn.

Không nên phụ thuộc Colab/Kaggle khi đang thuyết trình (phiên có thể bị ngắt, phải chờ cấp GPU) — chỉ dùng chúng để train.
