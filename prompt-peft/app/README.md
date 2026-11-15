# App và thí nghiệm: Text-to-LoRA vs GEPA

> **Trạng thái: chưa implement.** Kế hoạch đầy đủ: [`../docs/04-thiet-ke-thi-nghiem-va-app.md`](../docs/04-thiet-ke-thi-nghiem-va-app.md).

Chạy cả hai pipeline trên **cùng một model** (đề xuất Llama-3.1-8B-Instruct) và **cùng 5 dataset**: ARC-Challenge, HellaSwag, BoolQ, HumanEval, AIME 2025. Kết quả nạp vào một app demo 3 tab (LoRA Studio, Prompt Evolution, Đấu trường).

## Cấu trúc thư mục

```
app/
├── README.md
├── config.yaml              # model, checkpoint T2L, endpoint vLLM, LLM ngoài, ngân sách GEPA, seed
├── requirements.txt         # cố định phiên bản: vllm, dspy, gepa, peft, transformers, datasets
│
├── data/                    # [So sánh + App]
│   ├── load.py              # ARC-C, HellaSwag, BoolQ, HumanEval, AIME 2025 (+ MBPP, AIME 2022–2024)
│   └── splits.py            # chia train/val/test cố định seed → artifacts/splits/
│
├── harness/                 # [So sánh + App] dùng chung cho MỌI phương án
│   ├── prompts.py           # template từng dataset (trắc nghiệm, yes/no, code, AIME)
│   ├── extract.py           # tách đáp án, một hàm cho mỗi dataset
│   ├── sandbox.py           # chạy unit test HumanEval/MBPP có timeout, không mạng
│   ├── evaluate.py          # chạy một phương án trên một dataset → CSV
│   └── stats.py             # bootstrap CI
│
├── t2l/                     # [Team T2L]
│   ├── setup_t2l.md         # cài repo SakanaAI/text-to-lora, tải checkpoint, commit đã dùng
│   ├── descriptions.py      # lấy mô tả có sẵn / sinh bằng LLM ngoài (prompt Phụ lục L)
│   ├── generate.py          # mô tả → adapter PEFT
│   └── register_vllm.py     # đăng ký LoRA vào vLLM
│
├── gepa_lab/                # [Team GEPA]
│   ├── programs.py          # chương trình DSPy cho 5 dataset
│   ├── feedback.py          # metric + hàm phản hồi μ_f
│   ├── optimize.py          # chạy GEPA (model gốc hoặc model + LoRA)
│   └── tree.py              # đọc log → cây ứng viên, biên Pareto, đường cong
│
├── experiments/             # [So sánh + App]
│   └── run_arms.py          # A0–A7 trên 5 dataset → artifacts/results/*.csv
│
├── artifacts/               # splits, descriptions, lora, gepa, results (file lớn không commit)
├── server/                  # [App] FastAPI: /t2l/*, /gepa/*, /arena/*
├── client/                  # [App] web UI 3 tab
└── notebooks/               # part1_t2l.ipynb, part2_gepa.ipynb, part3_arena.ipynb
```

Nhãn `[...]` là team phụ trách (xem `../docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md`).

## Quy ước
- Mọi tham số trong `config.yaml`; API key trong `.env` (**không commit**).
- Mọi phương án chạy qua `harness/`; không trộn số từ script đánh giá khác.
- Seed cố định cho split, sinh mô tả, GEPA; lưu id của từng split.
- Mỗi lần đánh giá ghi một dòng CSV: dataset, phương án, model, mô tả hoặc prompt id, metric, CI, số dữ liệu có nhãn dùng, GPU-giờ, rollout, tiền API.
- Notebook chỉ gọi code trong các thư mục trên, không chứa logic riêng.
