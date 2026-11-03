# Paper ứng viên từ 2025 trở đi (để so sánh)

> **Quyết định hiện tại: vẫn giữ LoPA (NeurIPS 2024) + BetterTogether (EMNLP 2024)** làm 2 paper chính.
> Bảng dưới đây là các ứng viên mới hơn đã tìm (11/2026), giữ lại để tham khảo, dùng cho phần "hướng mở rộng" khi thuyết trình, hoặc nếu sau này đổi paper.

- Hội nghị đã được xác nhận qua trang chính thức hoặc arXiv.
- Cột "Code" ghi "?" là chưa kiểm tra được.
- ★ = mức dễ demo trong 10–20GB VRAM (★★★ là dễ nhất).
- PDF có sẵn trong thư mục này: `text-to-lora-icml2025.pdf`, `gepa-iclr2026.pdf`.

## A. Prompt → PEFT: sinh adapter từ chữ hoặc tài liệu, không cần train

| # | Paper | Hội nghị | Ý chính | Code | Demo |
|---|---|---|---|---|---|
| 1 | **Doc-to-LoRA: Learning to Instantly Internalize Contexts** | **ICML 2026** | Đọc một tài liệu → sinh LoRA trong một lần forward, model "nhớ" tài liệu mà không cần đưa vào context. Gần hoàn hảo trên needle-in-a-haystack 32K token | Có, kèm checkpoint (Gemma-2-2B) và demo online | ★★★ |
| 2 | **Text-to-LoRA: Instant Transformer Adaption** | ICML 2025 | Mô tả task bằng chữ → hypernetwork sinh LoRA ngay | Có, kèm checkpoint (Gemma-2-2B, Mistral-7B, Llama-8B) và web UI | ★★★ (có PDF) |
| 3 | Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights | NeurIPS 2025 | Vài prompt không nhãn → sinh LoRA, theo tác giả tốn ít hơn tới 12.000 lần so với full fine-tune | Có (trang dự án) | ★★ |
| 4 | Generative Adapter | ICLR 2025 | Context lúc test → sinh adapter cho model freeze. Cá nhân hoá rẻ hơn 4 lần so với nhét cả lịch sử vào prompt | ? | ★★ |
| 5 | Transformer²: Self-adaptive LLMs | ICLR 2025 | Chỉ chỉnh singular value của trọng số (SVF, học bằng RL). Chạy 2 lượt: nhận diện task rồi trộn các "expert vector" tương ứng | Có | ★★ |

## B. Tối ưu prompt chữ

| # | Paper | Hội nghị | Ý chính | Code | Demo |
|---|---|---|---|---|---|
| 6 | **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** | **ICLR 2026, Oral** | LLM tự phân tích các lần chạy rồi tiến hoá prompt. Hơn GRPO (fine-tune bằng RL) trung bình 6 điểm, ít lượt chạy hơn tới 35 lần | `pip install gepa`, có trong DSPy (`dspy.GEPA`) | ★★★ (có PDF) |
| 7 | ACE: Agentic Context Engineering | ICLR 2026 | Context là một "playbook" được cập nhật dần qua sinh, suy ngẫm, chọn lọc. +10.6% với agent | Có | ★★ |
| 8 | MetaSPO: System Prompt Optimization with Meta-Learning | NeurIPS 2025 | Tối ưu **system prompt** dùng chung cho nhiều task, theo kiểu meta-learning hai tầng | ? | ★★ |
| 9 | TextGrad: Automatic "Differentiation" via Text | **Nature** 2025 (tạp chí) | "Lan truyền ngược" bằng phản hồi chữ của LLM, API giống PyTorch | `pip install textgrad` | ★★★ |
| 10 | PromptWizard | Findings ACL 2025 | Tự tối ưu cả câu lệnh lẫn ví dụ qua vòng phản hồi (Microsoft) | Có | ★★ |
| 11 | SPO: Self-Supervised Prompt Optimization | Findings EMNLP 2025 | Tối ưu prompt **không cần đáp án mẫu**, chỉ so sánh từng cặp output | ? | ★★ |

## C. Soft prompt và nén prompt

| # | Paper | Hội nghị | Ý chính | Code | Demo |
|---|---|---|---|---|---|
| 12 | Improving Complex Reasoning with Dynamic Prompt Corruption | ICLR 2025 | Soft prompt có thể làm hỏng suy luận nhiều bước; phát hiện và che các token gây nhiễu. +4–8% so với prompt tuning thường | ? | ★★ |
| 13 | ID-SPAM | ACL 2025 (short) | Soft prompt sinh theo từng input, dùng attention để đánh trọng số token | ? | ★★ |
| 14 | 500xCompressor | ACL 2025 (main) | Nén prompt dài thành vài token, tối đa 500 lần | Có | ★★ |

## D. Biến thể LoRA

| # | Paper | Hội nghị | Ý chính | Code | Demo |
|---|---|---|---|---|---|
| 15 | **Tina: Tiny Reasoning Models via LoRA** | ICLR 2026 | LoRA + RL trên model 1.5B, đạt 43% AIME24, chi phí khoảng 9 USD | Có, kèm checkpoint | ★★★ |
| 16 | HiRA: Hadamard High-Rank Adaptation | ICLR 2025, Oral | Update = W ⊙ (low-rank), hạng cao hơn LoRA mà không thêm tham số | Có | ★★ |
| 17 | LoRA-One | ICML 2025, Spotlight | Khởi tạo LoRA từ gradient một bước của full fine-tune | Có | ★★ |
| 18 | RandLoRA | ICLR 2025 | Update full-rank từ tổ hợp các ma trận ngẫu nhiên hạng thấp | ? | ★★ |
| 19 | LoRA-Pro | ICLR 2025 | Chỉnh gradient của LoRA cho gần với full fine-tune | ? | ★ |

## E. Prompt ↔ trọng số, tự thích nghi

| # | Paper | Hội nghị | Ý chính | Code | Demo |
|---|---|---|---|---|---|
| 20 | SEAL: Self-Adapting Language Models | NeurIPS 2025 | Model tự viết dữ liệu fine-tune cho chính mình, học bằng RL | ? | ★ (rất tốn tính toán) |
| 21 | Comparing the learning dynamics of in-context learning and fine-tuning | ICLR 2026 | Phân tích vì sao in-context learning và fine-tune khái quát hoá khác nhau | — | Hợp làm lý thuyết, không có demo |

## Cặp paper gợi ý (nếu đổi)

1. **Text-to-LoRA (hoặc Doc-to-LoRA) + GEPA**: "biến prompt thành trọng số" đối lập với "tối ưu prompt thay vì trọng số". Hai paper bổ sung cho nhau, demo đều nhẹ.
2. **GEPA + Tina**: cùng bài toán suy luận, so trực tiếp **tối ưu prompt** với **LoRA + RL**. Hợp nếu muốn dựng thành một cuộc tranh luận.
3. **Text-to-LoRA + Doc-to-LoRA**: cùng nhóm tác giả (Sakana), cùng code, nên dễ làm nhất. Nhưng cả hai đều thiên về PEFT; thiếu phần tối ưu prompt chữ.

## Liên hệ với cặp paper đang chọn (gợi ý cho phần "hướng mở rộng")
- **LoPA → Text-to-LoRA / Doc-to-LoRA**: LoPA sinh *soft prompt* theo từng input; T2L/D2L đi xa hơn, sinh cả *LoRA adapter* từ chữ.
- **BetterTogether → GEPA**: cùng nhóm tác giả DSPy. BetterTogether xen kẽ tối ưu prompt và LoRA; GEPA cho thấy chỉ tối ưu prompt (bằng suy ngẫm) đã có thể thắng fine-tune bằng RL.

## Nguồn
- Doc-to-LoRA: [PMLR v306](https://proceedings.mlr.press/v306/charakorn26a.html) · [arXiv 2602.15902](https://arxiv.org/abs/2602.15902) · [Sakana](https://sakana.ai/doc-to-lora/)
- Text-to-LoRA: [PMLR v267](https://proceedings.mlr.press/v267/charakorn25a.html) · [arXiv 2506.06105](https://arxiv.org/abs/2506.06105) · [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora)
- Drag-and-Drop LLMs: [NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/e15b6539e6d61aab5dee4593222ba40e-Abstract-Conference.html) · [arXiv 2506.16406](https://arxiv.org/abs/2506.16406)
- Generative Adapter: [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/447708674d7f30492ca5338b36b07d0c-Abstract-Conference.html)
- Transformer²: [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/28974) · [arXiv 2501.06252](https://arxiv.org/abs/2501.06252)
- GEPA: [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) · [arXiv 2507.19457](https://arxiv.org/abs/2507.19457) · [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- ACE: [ICLR 2026](https://iclr.cc/virtual/2026/poster/10008343) · [arXiv 2510.04618](https://arxiv.org/abs/2510.04618)
- MetaSPO: [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/118792) · [arXiv 2505.09666](https://arxiv.org/abs/2505.09666)
- TextGrad: [arXiv 2406.07496](https://arxiv.org/abs/2406.07496) · [zou-group/textgrad](https://github.com/zou-group/textgrad)
- PromptWizard: [Findings ACL 2025](https://aclanthology.org/2025.findings-acl.1025)
- SPO: [arXiv 2502.06855](https://arxiv.org/abs/2502.06855)
- Dynamic Prompt Corruption: [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/28777) · [arXiv 2503.13208](https://arxiv.org/abs/2503.13208)
- ID-SPAM: [ACL 2025](https://aclanthology.org/2025.acl-short.74)
- 500xCompressor: [ACL 2025](https://aclanthology.org/2025.acl-long.1219)
- Tina: [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009721) · [arXiv 2504.15777](https://arxiv.org/abs/2504.15777)
- HiRA: [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html)
- LoRA-One: [arXiv 2502.01235](https://arxiv.org/abs/2502.01235)
- RandLoRA: [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30212)
- LoRA-Pro: [arXiv 2407.18242](https://arxiv.org/abs/2407.18242)
- SEAL: [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6b41e04c41726e2a60e456d0a2b961ab-Abstract-Conference.html)
- ICL vs fine-tuning: [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008523)
