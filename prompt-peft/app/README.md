# App: "Language-Driven Adaptation Lab"

> **Trạng thái: chưa implement.** Thư mục này chỉ có khung. Thiết kế đầy đủ: [`../docs/01-chon-paper-va-thiet-ke-app.md`](../docs/01-chon-paper-va-thiet-ke-app.md), mục 5–6.

App gồm 2 tab, tương ứng 2 paper:
- **Tab 1 — LoRA Studio (Text-to-LoRA)**: gõ mô tả task → sinh LoRA bằng hypernetwork → chat song song model gốc và model + LoRA; bảng so sánh gốc / mô tả trong prompt / 3-shot ICL / T2L; thí nghiệm mô tả tốt và kém; bản đồ t-SNE các LoRA.
- **Tab 2 — Prompt Evolution (GEPA)**: chọn task (IFBench, HotpotQA) → xem prompt tiến hoá qua từng vòng, phản hồi chữ, điểm theo rollout, cây ứng viên và biên Pareto; so sánh Baseline / MIPROv2 / GEPA.

## Cấu trúc thư mục

```
app/
├── README.md
├── config.yaml              # model, đường dẫn checkpoint, endpoint LLM, ngân sách GEPA
├── requirements.txt         # cố định phiên bản (dspy, gepa, peft, transformers, vllm)
│
├── data/                    # [Cả hai team] script tải và chia dữ liệu (không commit dữ liệu thô)
│   ├── benchmarks_t2l.py    # tập con GSM8K, ARC-Easy, BoolQ, PIQA (seed cố định)
│   ├── ifbench.py           # IF-RLVR train/val + IFBench test (như paper GEPA)
│   └── hotpotqa.py          # kèm BM25 index trên các đoạn văn của dataset
│
├── t2l/                     # [Team T2L]
│   ├── setup_t2l.md         # cách cài repo SakanaAI/text-to-lora, tải checkpoint, commit đã dùng
│   ├── generate.py          # mô tả → LoRA (gọi hypernetwork), lưu adapter
│   ├── evaluate.py          # gốc / mô tả / ICL / T2L (+ LoRA oracle nếu có) → CSV
│   ├── descriptions.py      # bộ mô tả: chi tiết, kém, ngẫu nhiên, tiếng Việt
│   └── tsne.py              # embedding LoRA → t-SNE
│
├── gepa_lab/                # [Team GEPA]
│   ├── programs.py          # chương trình DSPy: IFBench 2 bước, HotpotQA 2 hop
│   ├── feedback.py          # metric + hàm phản hồi μ_f (trả điểm + chữ)
│   ├── optimize.py          # --optimizer {baseline, miprov2, gepa} --budget N
│   ├── tree.py              # đọc log → cây ứng viên, biên Pareto
│   └── evaluate.py
│
├── artifacts/               # kết quả (không commit file lớn, xem .gitignore)
│   ├── lora/<mô tả>/        # LoRA do T2L sinh
│   ├── gepa/<task>/         # log GEPA/MIPROv2, chương trình đã tối ưu (.json)
│   └── results/*.csv        # bảng kết quả dùng cho slide và client
│
├── server/                  # [App] FastAPI
│   ├── main.py              # /t2l/generate, /t2l/chat, /t2l/eval, /gepa/runs, /gepa/tree, /gepa/predict
│   └── llm.py               # HF + peft cho Gemma (đổi adapter nóng); client OpenAI-compatible tới vLLM/API
│
├── client/                  # [App] web UI 2 tab
│
└── notebooks/               # dùng khi live build trên sân khấu
    ├── part1_text_to_lora.ipynb
    └── part2_gepa.ipynb
```

Nhãn `[...]` là nhóm phụ trách (xem [`../docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md`](../docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md), mục 4).

## Quy ước
- Mọi tham số (model, checkpoint, endpoint, ngân sách) đặt trong `config.yaml`, không hard-code. Có 2 profile: `gpu` (Gemma-2-2B + Qwen3-8B) và `light` (Qwen3-4B hoặc bản lượng tử hoá).
- API key (cho reflection LM) đặt trong `.env`, **không commit**.
- Seed cố định cho việc chia dữ liệu và chạy optimizer.
- Mỗi lần đánh giá ghi thêm một dòng vào `artifacts/results/*.csv`: task, phương pháp, model, ngân sách hoặc số tham số, metric, thời gian.
- Notebook chỉ **gọi** code trong `t2l/`, `gepa_lab/`, không chứa logic riêng.
- Artifact lớn lưu ở Google Drive hoặc Hugging Face Hub; trong repo chỉ giữ script tải về.
