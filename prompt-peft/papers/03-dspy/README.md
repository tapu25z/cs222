# DSPy — nền tảng cho Paper 2

| | |
|---|---|
| Tên đầy đủ | *DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines* |
| Tác giả | Omar Khattab, …, Christopher Potts và cộng sự (Stanford) |
| Hội nghị | **ICLR 2024** (bản arXiv 2023 trước đó có tên *…into Self-Improving Pipelines*) |
| Link | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html) · [arXiv 2310.03714](https://arxiv.org/abs/2310.03714) |
| Code / tài liệu | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) · [dspy.ai](https://dspy.ai) |
| Dịch chi tiết | [`dich-chi-tiet.md`](dich-chi-tiet.md) (*chờ PDF*) |

> ⚠️ **PDF chưa có trong repo.** Tải từ arXiv và lưu thành `dspy-iclr2024.pdf` trong thư mục này.

---

## Tóm tắt (theo abstract)

DSPy là một **mô hình lập trình** biểu diễn pipeline LM dưới dạng **đồ thị tính toán dạng chương trình (imperative)**. Trong đó LM được gọi qua các **module khai báo (declarative)**.
- Các module được **tham số hoá**: chúng có thể *học* (bằng cách tạo và thu thập ví dụ minh hoạ) cách áp dụng tổ hợp các kỹ thuật prompting, fine-tuning, augmentation và suy luận.
- Một **compiler** tối ưu bất kỳ pipeline DSPy nào để tối đa một metric cho trước.
- Kết quả trong paper: các pipeline đã compile **vượt few-shot prompting thông thường** và vượt **ví dụ do chuyên gia viết** với GPT-3.5 và Llama2-13b-chat. Chương trình compile cho model nhỏ, mở như T5-770M, **cạnh tranh được** với chuỗi prompt do chuyên gia viết cho GPT-3.5.

## Giải thích dễ hiểu và code mẫu

Đã viết trong [`../../docs/01-chon-paper-va-thiet-ke-app.md`](../../docs/01-chon-paper-va-thiet-ke-app.md), **mục 3 "DSPy là gì?"**:
- bảng so sánh PyTorch ↔ DSPy;
- 5 khái niệm: Signature, Module, Program, Metric, Optimizer, có code mẫu;
- bên trong optimizer bootstrap làm gì, và cầu nối sang BetterTogether;
- gợi ý trình bày trong 12 phút.

## Cần lấy từ PDF
- [ ] Định nghĩa chính thức của signature, module, teleprompter (tên cũ của optimizer).
- [ ] Các giai đoạn của compiler.
- [ ] Hai case study (GSM8K, HotPotQA): setup và bảng kết quả.

## Tài liệu học thêm
- Tutorial chính thức trên https://dspy.ai (mục Tutorials: RAG, multi-hop, math reasoning).
- API thay đổi theo phiên bản; cố định phiên bản khi viết code (xem `app/README.md`).
