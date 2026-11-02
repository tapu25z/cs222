# Prompting + PEFT nâng cao — LoPA + BetterTogether

Đề tài thuyết trình (khoảng 3 tiếng, nhóm 15 người): trình bày 2 paper và build từng bước một app áp dụng chúng.
Tài liệu tham khảo khoá học: [Stanford CS224N](https://web.stanford.edu/class/cs224n/) (Lecture 9: Efficient Adaptation).

| | Paper | Hội nghị |
|---|---|---|
| 1 | **LoPA** — Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation | NeurIPS 2024 |
| 2 | **BetterTogether** — Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together | EMNLP 2024 |
| nền | **DSPy** — Compiling Declarative Language Model Calls into State-of-the-Art Pipelines (cần cho paper 2) | ICLR 2024 |

## Cấu trúc thư mục

```
prompt-peft/
├── README.md                  ← file này
├── docs/                      ← thiết kế và kế hoạch
│   ├── 01-chon-paper-va-thiet-ke-app.md
│   └── 02-ke-hoach-thuyet-trinh-va-phan-cong.md
├── papers/                    ← tài liệu về paper: link, tóm tắt, dịch/diễn giải
│   ├── README.md              ← danh sách tất cả paper + link
│   ├── 01-lopa/               ← paper 1 (PDF + tóm tắt + dịch chi tiết)
│   ├── 02-bettertogether/     ← paper 2
│   ├── 03-dspy/               ← nền tảng cho paper 2
│   └── reference/             ← paper tham khảo
│       ├── basic/             ← CoT, GPT-3 few-shot, Adapter, LoRA, Lottery Ticket, slide CS224N
│       └── advanced/          ← ứng viên không chọn: PARA, POST
└── app/                       ← code app demo (chưa implement)
    └── README.md              ← cấu trúc code và quy ước
```

## Đọc theo thứ tự nào?
1. [`docs/01-chon-paper-va-thiet-ke-app.md`](docs/01-chon-paper-va-thiet-ke-app.md): vì sao chọn 2 paper, thiết kế app, hạ tầng demo.
2. [`docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md`](docs/02-ke-hoach-thuyet-trinh-va-phan-cong.md): kịch bản 180 phút, phân công, lộ trình 6 tuần.
3. Team LoPA: [`papers/01-lopa/`](papers/01-lopa/).
4. Team BetterTogether: [`papers/03-dspy/`](papers/03-dspy/) trước, rồi [`papers/02-bettertogether/`](papers/02-bettertogether/).
5. Người làm code: [`app/README.md`](app/README.md).
