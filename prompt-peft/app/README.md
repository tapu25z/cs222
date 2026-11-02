# App: "One Model, Many Skills"

> **Trạng thái: chưa implement.** Thư mục này chỉ có khung. Thiết kế đầy đủ: [`../docs/01-chon-paper-va-thiet-ke-app.md`](../docs/01-chon-paper-va-thiet-ke-app.md), mục 5–6.

App gồm 2 phần, tương ứng 2 paper:
- **Phần 1 — LoPA**: skill một bước (SST-2, UIT-VSFC, CruxEval-O). So sánh Zero-shot, Few-shot, Prompt Tuning, LoRA, LoPA.
- **Phần 2 — BetterTogether**: pipeline DSPy (GSM8K, HotPotQA). So sánh Zero-shot, Chỉ prompt, Chỉ LoRA, Prompt → LoRA → Prompt.

## Cấu trúc thư mục

```
app/
├── README.md
├── config.yaml              # tên model, m, r, đường dẫn artifact, endpoint LLM
├── requirements.txt         # cố định phiên bản (đặc biệt dspy, peft, transformers)
│
├── data/                    # [Data]  script tải và chia dữ liệu (không commit dữ liệu thô)
│   ├── sst2.py
│   ├── uit_vsfc.py
│   ├── cruxeval.py          # chia train/test 50-50 như paper, seed cố định
│   ├── gsm8k.py
│   └── hotpotqa.py          # kèm BM25 index trên các đoạn văn của dataset
│
├── lopa/                    # [Team LoPA]
│   ├── lopa.py              # module LoPA: Z = Z_S ∘ sigmoid(u·vᵀ)
│   ├── baselines.py         # zero/few-shot, Prompt Tuning, LoRA (peft)
│   ├── train.py             # python -m lopa.train --skill sst2 --method lopa
│   └── evaluate.py
│
├── bettertogether/          # [Team BetterTogether]
│   ├── programs.py          # DSPy program: CoT (GSM8K), MultiHop (HotPotQA)
│   ├── retriever.py         # BM25
│   ├── optimize.py          # --strategy {prompt, weights, prompt_weights_prompt}
│   └── evaluate.py
│
├── artifacts/               # kết quả train (không commit file lớn, xem .gitignore)
│   ├── lopa/<skill>.pt      # Z_S + MLP_u + MLP_v, vài MB
│   ├── lora/<skill>/        # LoRA adapter
│   ├── dspy/<skill>.json    # program đã compile
│   └── results/*.csv        # bảng kết quả dùng cho slide và client
│
├── server/                  # [App] FastAPI
│   ├── main.py              # /lopa/predict, /bt/run, /results
│   └── llm.py               # load backbone LoPA (HF), client tới vLLM/Ollama cho BetterTogether
│
├── client/                  # [App] web UI: chọn skill, nhập câu, bảng so sánh, xem trace
│
└── notebooks/               # dùng khi live build trên sân khấu
    ├── part1_lopa.ipynb
    └── part2_bettertogether.ipynb
```

Nhãn `[...]` là nhóm phụ trách (xem [`../docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md`](../docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md), mục 4).

## Quy ước
- Mọi tham số (model, m, r, lr, đường dẫn) đặt trong `config.yaml`, không hard-code. Có 2 profile: `gpu` (model chính) và `light` (Colab/Kaggle, CPU).
- Seed cố định cho việc chia dữ liệu và train.
- Mỗi lần chạy `evaluate.py` ghi thêm một dòng vào `artifacts/results/*.csv`: skill, method, model, số tham số train, metric, thời gian.
- Notebook chỉ **gọi** code trong `lopa/`, `bettertogether/`, không chứa logic riêng, để bản demo và bản thật là một.
- Artifact lớn (checkpoint, adapter) lưu ở Google Drive / Hugging Face Hub; trong repo chỉ giữ script tải về.
