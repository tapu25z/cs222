# Chọn paper và thiết kế app — Prompting + PEFT nâng cao

> Bản đề xuất (draft), sẽ chỉnh lại theo lựa chọn của nhóm. Đọc `01-prompting-peft-co-ban.md` trước.

---

## 1. So sánh 3 paper ứng viên (đã có trong `paper-advance/`)

| Tiêu chí | **LoPA** (NeurIPS 2024) | PARA (ACL 2024*) | POST (ICML 2025) |
|---|---|---|---|
| Hội nghị | NeurIPS — top-tier ML | ACL — top-tier NLP | ICML — top-tier ML |
| Dễ hiểu để thuyết trình | ★★★ (một công thức, một hình) | ★★ | ★ (distillation + DP + prompt transfer) |
| Dễ cài đặt lại | ★★★ (chỉ thêm module ở **input**, có code chính thức) | ★★ (phải sửa **mọi tầng** Transformer) | ★ (cần distill model, train DP, transfer) |
| Chạy được trên Colab/1 GPU | Có (model 0.5B–1.5B) | Có | Khó |
| Liên hệ với *Prompting* | Mạnh — là soft prompt | Yếu — giống adapter hơn | Mạnh — soft prompt |
| Liên hệ với *PEFT* | Mạnh — so sánh trực tiếp với LoRA | Mạnh | Trung bình |
| Câu chuyện cho app | "Một model, nhiều task, không cần adapter ở server" | "Multi-tenant nhanh hơn LoRA" | "Tuỳ biến LLM mà không lộ dữ liệu" |

\* Bản PDF trong repo là bản arXiv (2502.01033), không ghi venue — cần kiểm tra lại trước khi trích dẫn là ACL 2024.

### ✅ Đề xuất: **LoPA — "Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation"** (Jain et al., NeurIPS 2024)

Lý do:
1. **Đúng chủ đề "Prompting + PEFT"**: nó là một phương pháp *prompt tuning* (nhánh prompting) được nâng cấp để cạnh tranh với *LoRA* (nhánh PEFT) → kể được câu chuyện nối tiếp bài giảng CS224N Lecture 9.
2. **Ý tưởng gọn, dễ trình bày** trong 1–2 slide (xem mục 2).
3. **Dễ code lại từng bước trên sân khấu**: toàn bộ phương pháp chỉ khoảng 50 dòng PyTorch bọc quanh một model HuggingFace đã freeze.
4. **Có lợi ích thực tế rõ cho app**: soft prompt chỉ là phần đầu của input → một backbone duy nhất phục vụ nhiều task, đổi task chỉ cần đổi một file vài MB.
5. Kết quả mạnh: ngang FFT/LoRA trên GLUE, thắng LoRA 11/24 trường hợp, ít tham số hơn.

POST là lựa chọn dự phòng nếu nhóm muốn góc nhìn **privacy** (ấn tượng hơn nhưng rủi ro khi demo cao hơn).

---

## 2. LoPA trong 1 trang

**Vấn đề**: LoRA mạnh nhưng mỗi task cần một adapter lưu và chạy **trên server**. Prompt tuning thì không cần (chỉ là prefix của input) nhưng **yếu hơn**.

**Ý tưởng**: soft prompt `Z` (m vector, mỗi vector d chiều) được tạo từ 2 phần:

```
Z  =  Z_S  ⊙  sigmoid( Z_I )
      ^^^             ^^^
  phần chung       phần riêng cho từng câu input,
  của task         dạng low-rank:  Z_I = u · vᵀ
  (m × d,          u = MLP_u(X'),  v = MLP_v(X'),  hạng r nhỏ (vd. 4)
   học được)       X' = biểu diễn câu input từ một encoder nhỏ (vd. RoBERTa / CodeSage)
```
Sau đó `concat(Z, embedding(input))` → đưa vào LLM **đã freeze**.

- `Z_S` (task-specific) = "hướng dẫn chung" giống prompt tuning.
- `Z_I` (instance-specific) = điều chỉnh theo từng câu hỏi (giống IDPG) nhưng nén low-rank → ít tham số.
- Cổng `sigmoid` + nhân từng phần tử (⊙) → hai phần học **phụ thuộc lẫn nhau** (paper chỉ ra là tốt hơn cộng đơn giản `Z_S + Z_I`).
- Chỉ train: `Z_S`, `MLP_u`, `MLP_v`. LLM và encoder đều freeze.

**Thiết lập trong paper**: m = 10 virtual tokens; NLU dùng RoBERTa-355M; code dùng CodeGen-350M, DeepSeek-Coder 1.3B/7B, Phi-2, Phi-3, Llama-3-8B; encoder là RoBERTa hoặc CodeSage-125M + MLP.

---

## 3. Ý tưởng app (đề xuất)

### "One Model, Many Skills" — trợ lý đa nhiệm dùng một LLM duy nhất

Một web app (Gradio/Streamlit) với **một LLM nhỏ được freeze** (vd. Qwen2.5-0.5B/1.5B) và một danh sách **"skill"**. Mỗi skill là một file LoPA vài MB.
Người dùng chọn skill → app sinh soft prompt cho câu input → gửi vào cùng một model.

Skill gợi ý (chọn 2–3):
- **Phân loại cảm xúc review** (SST-2 tiếng Anh, hoặc dữ liệu review tiếng Việt như UIT-VSFC).
- **Kiểm tra hai câu có cùng nghĩa** (MRPC/QQP).
- **Dự đoán output của code Python** (CruxEval-O) — giống thí nghiệm trong paper.

Màn hình **"So sánh"** (điểm nhấn của bài thuyết trình): cùng một input, chạy song song
1. Zero-shot prompting
2. Few-shot / CoT prompting
3. Prompt Tuning (Lester)
4. LoRA
5. **LoPA**

và hiển thị: đáp án, độ chính xác trên tập test, **số tham số train**, dung lượng file, và **"server có phải load adapter không?"**.

### Kiến trúc

```
              ┌──────────────────────── phía người dùng / client ───────────────────┐
 input text ─►│ Encoder nhỏ (freeze) ─► MLP_u, MLP_v ─► Z_I = u·vᵀ                    │
              │                         Z = Z_S ⊙ sigmoid(Z_I)   (file skill.pt)      │
              └───────────────────────────────┬────────────────────────────────────────┘
                                              │ soft prompt Z (m × d) + input
                                              ▼
                               ┌──────── server ────────┐
                               │  LLM freeze (dùng chung │
                               │  cho mọi skill)         │──► đáp án
                               └─────────────────────────┘
```

### Các bước build khi thuyết trình (live, mỗi bước một commit / một cell notebook)
1. **Baseline prompting**: load LLM, chạy zero-shot và few-shot trên skill 1 → ghi lại accuracy.
2. **Prompt Tuning**: thêm m = 10 vector học được ở đầu input, train vài phút → so sánh.
3. **LoRA** bằng thư viện `peft` → so sánh (mạnh nhưng phải gắn adapter vào model).
4. **LoPA**: viết module `Z_S ⊙ sigmoid(u·vᵀ)` (khoảng 50 dòng) → train → so sánh.
5. **Đóng gói skill**: lưu mỗi skill thành một file nhỏ; chứng minh đổi skill không cần reload model.
6. **Giao diện**: Gradio app với dropdown chọn skill + bảng so sánh.
7. (Tuỳ chọn) Thí nghiệm hạng r = 1, 2, 4, 8 → vẽ biểu đồ giống Figure 4 của paper.

Train thật sẽ làm **trước** buổi thuyết trình (checkpoint có sẵn); trên sân khấu chỉ chạy vài bước train ngắn cho thấy loss giảm, rồi load checkpoint.

---

## 4. Quyết định đã chốt

| Hạng mục | Lựa chọn |
|---|---|
| Paper | **LoPA** (NeurIPS 2024) |
| Dữ liệu | **Cả hai**: skill tiếng Anh (SST-2, có số liệu paper để đối chiếu) + skill tiếng Việt (UIT-VSFC, review sinh viên) |
| Phần cứng | Colab/Kaggle (T4) **hoặc** GPU riêng → code chạy được cả hai, chọn cỡ model bằng cấu hình |
| Dạng app | **Web + API (FastAPI)** — tách client/server đúng như kiến trúc trong paper |

### Hệ quả cho thiết kế
- **LLM backbone**: `Qwen2.5-1.5B` (đa ngôn ngữ, hỗ trợ tiếng Việt). Trên T4 có thể giảm xuống `Qwen2.5-0.5B`; với GPU riêng có thể thử `Qwen2.5-3B/7B`. Đặt trong file config.
- **Encoder cho Z_I**: `xlm-roberta-base` (đa ngôn ngữ; paper dùng RoBERTa cho tiếng Anh) + MLP (h = 256), rank r = 4, m = 10 virtual tokens như paper.
- **FastAPI server** chỉ giữ LLM đã freeze, có endpoint nhận `soft_prompt + input` → dù có bao nhiêu skill thì server cũng không phải load adapter nào. Để so sánh, LoRA được phục vụ bằng endpoint riêng, phải gắn adapter của từng skill.
- **Client** (web nhỏ, HTML/JS hoặc một service "skill") giữ các file skill LoPA, tạo `Z` rồi gọi server. Màn hình so sánh hiển thị 5 phương pháp như mục 3.

### Cấu trúc thư mục dự kiến
```
prompt-peft/app/
  config.yaml          # tên model, m, r, đường dẫn skill
  lopa.py              # module LoPA (Z_S ⊙ sigmoid(u·vᵀ))
  train.py             # train PT / LoRA / LoPA cho một skill
  evaluate.py          # zero-shot, few-shot, PT, LoRA, LoPA → bảng kết quả
  server/main.py       # FastAPI: LLM freeze + /generate, /generate_lora
  client/              # web UI chọn skill + bảng so sánh
  notebooks/demo.ipynb # chạy từng bước trên Colab/Kaggle khi thuyết trình
```

### Còn cần nhóm cho biết
- Thời lượng thuyết trình và số người trong nhóm (để chia các bước demo).
- Có muốn thêm skill code (CruxEval-O) giống paper không?
