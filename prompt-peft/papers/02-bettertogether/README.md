# BetterTogether — Fine-Tuning and Prompt Optimization

| | |
|---|---|
| Tên đầy đủ | *Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together* |
| Tác giả | Dilara Soylu, Christopher Potts, Omar Khattab (Stanford) |
| Hội nghị | **EMNLP 2024** (main conference), trang 10696–10710 |
| Link | [ACL Anthology 2024.emnlp-main.597](https://aclanthology.org/2024.emnlp-main.597) · [PDF](https://aclanthology.org/2024.emnlp-main.597.pdf) · [arXiv 2407.10930](https://arxiv.org/abs/2407.10930) |
| Code | Optimizer `BetterTogether` nằm trong thư viện [DSPy](https://github.com/stanfordnlp/dspy) |
| Cần đọc trước | [`../03-dspy/`](../03-dspy/) |
| Dịch chi tiết | [`dich-chi-tiet.md`](dich-chi-tiet.md) (*chờ PDF*) |

> ⚠️ **PDF chưa có trong repo.** Phần dưới được tổng hợp từ abstract và trang ACL Anthology. Các chỗ đánh dấu *(cần kiểm tra)* phải đối chiếu lại với bản PDF trước khi đưa vào slide.
> Việc cần làm: tải PDF từ link trên, lưu thành `bettertogether-emnlp2024.pdf` trong thư mục này.

---

## Tóm tắt 1 trang

**Bối cảnh.** Hệ thống NLP hiện đại thường là **pipeline nhiều module**, mỗi module gọi LLM một lần. Ví dụ RAG hỏi đáp nhiều bước: sinh truy vấn → tìm tài liệu → sinh truy vấn tiếp → tìm → trả lời. Tối ưu cả pipeline rất khó vì:
- chỉ có **nhãn cho đáp án cuối**, không có nhãn cho bước trung gian (truy vấn tìm kiếm "đúng" là gì?);
- **không lan truyền gradient** được qua cả pipeline (có bước tìm kiếm, có sinh văn bản rời rạc).

**Hai hướng tối ưu đã có (trong DSPy):**
1. **Tối ưu prompt**: tự tìm ví dụ few-shot hoặc câu lệnh cho từng module (vd. `BootstrapFewShotWithRandomSearch`).
2. **Fine-tune trọng số**: dùng các trace "đúng" do chính chương trình sinh ra làm dữ liệu train LM (vd. `BootstrapFinetune`).

**Ý tưởng.** Hai bước này **bổ trợ nhau**: cho **chính LM tự dạy mình** bằng cách **xen kẽ** tối ưu prompt và fine-tune trọng số.
- Prompt tốt → nhiều trace đúng hơn → dữ liệu fine-tune tốt hơn.
- Model đã fine-tune → tối ưu prompt lại trên model mới để tận dụng hết.
- Thứ tự gợi ý: **tối ưu prompt → fine-tune trọng số (trên trace của prompt đã tối ưu) → tối ưu prompt lần nữa** *(cần kiểm tra các biến thể thứ tự khác được thử trong paper)*.

**Thí nghiệm.**
- Task: **HotPotQA** (hỏi đáp nhiều bước), **GSM8K** (toán có lời văn), **Iris** (phân loại dựa trên đặc trưng).
- LM: **mistral-7b**, **llama-2-7b**, **llama-3-8b**.

**Kết quả** (theo abstract).
- Kết hợp tối ưu prompt và trọng số **tốt nhất ở 7/9 cặp** (dataset × LM).
- Trung bình theo LM và task, hơn "chỉ fine-tune trọng số" **tới 60%** và hơn "chỉ tối ưu prompt" **tới 6%** (bản arXiv v2; bản v1 ghi 65% và 5%).
- Theo task: khoảng **5–78%** trên HotPotQA, **2.5–10%** trên GSM8K, **3.5–88%** trên Iris *(cần kiểm tra là so với baseline nào)*.

---

## Cần lấy từ PDF trước khi làm (team BetterTogether)
- [ ] Ký hiệu và cách phát biểu bài toán (program Φ, module, metric μ, tập train).
- [ ] Thuật toán chính xác (pseudo-code), các thứ tự đã thử (vd. chỉ p, chỉ w, w→p, p→w, p→w→p).
- [ ] Cấu hình fine-tune: LoRA rank, learning rate, epoch, lượng tử hoá; phần cứng.
- [ ] Cấu hình tối ưu prompt: số ví dụ bootstrap, số candidate program.
- [ ] Kích thước train/dev/test của mỗi dataset, retriever dùng cho HotPotQA.
- [ ] Bảng kết quả chính (đầy đủ số) và phần phân tích.
- [ ] Phần Limitations.

## Điểm cần nhấn khi thuyết trình
1. Vấn đề "không có nhãn trung gian": dùng ví dụ HotPotQA, hỏi khán giả *"truy vấn tìm kiếm đúng là gì?"*.
2. Bootstrap: giữ lại trace dẫn tới đáp án đúng → nhãn trung gian "miễn phí".
3. Cùng một nguồn dữ liệu (trace) có thể dùng cho **prompt** (few-shot) **hoặc** **trọng số** (LoRA) → paper hỏi: tại sao không dùng cả hai?
4. Liên hệ với LoPA: LoPA = soft prompt, không sửa model; BetterTogether = hard prompt + LoRA, có sửa model nhưng tối ưu được cả pipeline.

## Nguồn
- [ACL Anthology 2024.emnlp-main.597](https://aclanthology.org/2024.emnlp-main.597)
- [arXiv 2407.10930](https://arxiv.org/abs/2407.10930)
