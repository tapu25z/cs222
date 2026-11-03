# Prompting + PEFT nâng cao — Text-to-LoRA + GEPA

Đề tài thuyết trình (khoảng 3 tiếng, nhóm 15 người): trình bày 2 paper và build từng bước một app áp dụng chúng.
Tài liệu tham khảo khoá học: [Stanford CS224N](https://web.stanford.edu/class/cs224n/) (Lecture 9: Efficient Adaptation).

| | Paper | Hội nghị |
|---|---|---|
| 1 | **Text-to-LoRA** — Instant Transformer Adaption (Sakana AI) | ICML 2025 |
| 2 | **GEPA** — Reflective Prompt Evolution Can Outperform Reinforcement Learning | ICLR 2026, Oral |
| nền | **DSPy** — Compiling Declarative Language Model Calls into State-of-the-Art Pipelines (GEPA là optimizer của DSPy) | ICLR 2024 |

## Cấu trúc thư mục

```
prompt-peft/
├── README.md                  ← file này
├── docs/                      ← thiết kế và kế hoạch
│   ├── 01-chon-paper-va-thiet-ke-app.md
│   └── 02-ke-hoach-thuyet-trinh-va-phan-cong.md
├── papers/                    ← tài liệu về paper: link, tóm tắt, dịch/diễn giải
│   ├── README.md              ← danh sách tất cả paper + link
│   ├── 01-text-to-lora/       ← paper 1 (PDF + tóm tắt + dịch chi tiết)
│   ├── 02-gepa/               ← paper 2
│   ├── 03-dspy/               ← nền tảng cho paper 2
│   ├── candidates/            ← bảng so sánh 21 paper ứng viên 2025+ (đã chọn từ đây)
│   └── reference/             ← paper tham khảo
│       ├── basic/             ← CoT, GPT-3 few-shot, Adapter, LoRA, Lottery Ticket, slide CS224N
│       ├── advanced/          ← ứng viên không chọn: PARA, POST
│       └── previous/          ← cặp chọn trước đó: LoPA, BetterTogether (có dịch đầy đủ)
└── app/                       ← code app demo (chưa implement)
    └── README.md              ← cấu trúc code và quy ước
```

## Đọc theo thứ tự nào?
1. [`docs/01-chon-paper-va-thiet-ke-app.md`](docs/01-chon-paper-va-thiet-ke-app.md): vì sao chọn 2 paper, thiết kế app, hạ tầng demo.
2. [`docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md`](docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md): kịch bản 180 phút, phân công, lộ trình 6 tuần.
3. Team Text-to-LoRA: [`papers/01-text-to-lora/`](papers/01-text-to-lora/).
4. Team GEPA: [`papers/03-dspy/`](papers/03-dspy/) trước, rồi [`papers/02-gepa/`](papers/02-gepa/).
5. Người làm code: [`app/README.md`](app/README.md).
