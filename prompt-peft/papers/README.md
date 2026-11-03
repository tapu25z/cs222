# Danh sách paper

## Paper chính

| Thư mục | Paper | Hội nghị | Link | Code | PDF trong repo |
|---|---|---|---|---|---|
| [`01-text-to-lora/`](01-text-to-lora/) | Text-to-LoRA: Instant Transformer Adaption — Charakorn, Cetin, Tang, Lange (Sakana AI) | **ICML 2025** | [PMLR](https://proceedings.mlr.press/v267/charakorn25a.html) · [arXiv 2506.06105](https://arxiv.org/abs/2506.06105) | [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) (có checkpoint) | ✅ (bản ICML) |
| [`02-gepa/`](02-gepa/) | GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning — Agrawal et al. | **ICLR 2026, Oral** | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) · [arXiv 2507.19457](https://arxiv.org/abs/2507.19457) | [gepa-ai/gepa](https://github.com/gepa-ai/gepa) · `dspy.GEPA` | ✅ (bản ICLR) |
| [`03-dspy/`](03-dspy/) | DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines — Khattab et al. (**nền tảng cho GEPA**) | ICLR 2024 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html) · [arXiv 2310.03714](https://arxiv.org/abs/2310.03714) | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) · [dspy.ai](https://dspy.ai) | ✅ (bản arXiv v1) |

Mỗi thư mục gồm:
- `README.md`: link, thông tin, **tóm tắt 1 trang**, các con số chính, điểm cần nhớ khi thuyết trình.
- `dich-chi-tiet.md`: **dịch và diễn giải từng mục** của paper sang tiếng Việt (mục nào khó thì có giải thích thêm).
- File PDF.

## Paper ứng viên mới (2025+)

Bảng so sánh 21 paper từ 2025 trở đi (hội nghị, ý chính, code, độ dễ demo) và các cặp gợi ý, dùng khi chọn paper: [`candidates/README.md`](candidates/README.md). Nhóm đã chọn cặp **Text-to-LoRA + GEPA** từ bảng này.

## Paper tham khảo

### `reference/previous/` — cặp paper chọn trước đó (đã thay)
Đã có tóm tắt và dịch chi tiết đầy đủ, giữ lại để tham khảo hoặc làm phần "bối cảnh" khi thuyết trình.

| Thư mục | Paper | Hội nghị | Liên hệ với cặp mới |
|---|---|---|---|
| [`reference/previous/lopa/`](reference/previous/lopa/) | LoPA — Prompt Tuning Strikes Back | NeurIPS 2024 | LoPA sinh *soft prompt* theo từng input; Text-to-LoRA sinh cả *LoRA* từ mô tả |
| [`reference/previous/bettertogether/`](reference/previous/bettertogether/) | BetterTogether — Fine-Tuning and Prompt Optimization | EMNLP 2024 | Cùng nhóm tác giả DSPy với GEPA; xen kẽ tối ưu prompt và LoRA |

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
