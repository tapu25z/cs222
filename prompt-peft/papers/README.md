# Danh sách paper

## Paper chính

| Thư mục | Paper | Hội nghị | Link | Code | PDF trong repo |
|---|---|---|---|---|---|
| [`01-lopa/`](01-lopa/) | Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation — Jain, Chaudhuri, Reps, Jermaine | NeurIPS 2024 | [arXiv 2405.15282](https://arxiv.org/abs/2405.15282) | [jabhinav/Prompt-Tuning-Strikes-Back-with-LOPA](https://github.com/jabhinav/Prompt-Tuning-Strikes-Back-with-LOPA) | ✅ |
| [`02-bettertogether/`](02-bettertogether/) | Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together — Soylu, Potts, Khattab | EMNLP 2024 (main) | [ACL Anthology 2024.emnlp-main.597](https://aclanthology.org/2024.emnlp-main.597) · [arXiv 2407.10930](https://arxiv.org/abs/2407.10930) | Trong thư viện [DSPy](https://github.com/stanfordnlp/dspy) | ❌ cần tải |
| [`03-dspy/`](03-dspy/) | DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines — Khattab et al. | ICLR 2024 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html) · [arXiv 2310.03714](https://arxiv.org/abs/2310.03714) | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) · [dspy.ai](https://dspy.ai) | ❌ cần tải |

Mỗi thư mục gồm:
- `README.md`: link, thông tin, **tóm tắt 1 trang**, các con số chính, điểm cần nhớ khi thuyết trình.
- `dich-chi-tiet.md`: **dịch và diễn giải từng mục** của paper sang tiếng Việt (mục nào khó thì có giải thích thêm).
- File PDF.

## Paper tham khảo

### `reference/basic/` — nền tảng (team lý thuyết đã trình bày)
| File | Paper | Hội nghị |
|---|---|---|
| `few-shot.pdf` | Language Models are Few-Shot Learners (GPT-3) — Brown et al. | NeurIPS 2020 |
| `CoT.pdf` | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — Wei et al. | NeurIPS 2022 |
| `PEFT.pdf` | Parameter-Efficient Transfer Learning for NLP (Adapter) — Houlsby et al. | ICML 2019 |
| `loRa.pdf` | LoRA: Low-Rank Adaptation of Large Language Models — Hu et al. | ICLR 2022 |
| `finding-sparse.pdf` | The Lottery Ticket Hypothesis — Frankle & Carbin | ICLR 2019 |
| `slide.pdf` | CS224N Lecture 9: Efficient Adaptation — Diyi Yang | Stanford |

### `reference/advanced/` — ứng viên đã cân nhắc nhưng không chọn
| File | Paper | Hội nghị | Lý do không chọn |
|---|---|---|---|
| `post-icml2025.pdf` | Efficient and Privacy-Preserving Soft Prompt Transfer for LLMs (POST) | ICML 2025 | Phải distill model, khó demo |
| `para-acl2024.pdf` | PARA: Parameter-Efficient Fine-tuning with Prompt Aware Representation Adjustment | ACL 2024 (theo tên file; bản PDF là arXiv 2502.01033, chưa xác nhận venue) | Sửa mọi tầng ở server, ngược thông điệp của LoPA |

Các paper liên quan khác (không có PDF): DePT (ICLR 2024, baseline trong LoPA), S-LoRA (MLSys 2024), Gist Tokens (NeurIPS 2023), DoRA (ICML 2024).
