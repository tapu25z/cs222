# BetterTogether — Fine-Tuning and Prompt Optimization

| | |
|---|---|
| Tên đầy đủ | *Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together* |
| Tác giả | Dilara Soylu, Christopher Potts, Omar Khattab (Stanford) |
| Hội nghị | **EMNLP 2024** (main conference), trang 10696–10710 |
| Link | [ACL Anthology 2024.emnlp-main.597](https://aclanthology.org/2024.emnlp-main.597) · [arXiv 2407.10930](https://arxiv.org/abs/2407.10930) · PDF trong repo: [`bettertogether-emnlp2024.pdf`](bettertogether-emnlp2024.pdf) |
| Code | Optimizer `BetterTogether` trong thư viện [DSPy](https://github.com/stanfordnlp/dspy) (http://dspy.ai) |
| Cần đọc trước | [`../03-dspy/`](../03-dspy/) |
| Dịch chi tiết | [`dich-chi-tiet.md`](dich-chi-tiet.md) |

---

## Tóm tắt 1 trang

**Bối cảnh.** Hệ thống NLP hiện đại là **LM program**: pipeline nhiều module, mỗi module `M_i` có một **prompt π_i** và dùng một LM với **trọng số θ_i**. Ví dụ HotPotQA: sinh truy vấn → tìm → sinh truy vấn → tìm → trả lời.

**Bài toán.** Tìm Π (mọi prompt) và Θ (mọi trọng số) để tối đa metric µ trên tập train nhỏ:
```
arg max_{Θ,Π} (1/|X|) Σ µ(Φ⟨Θ,Π⟩(x), m)
```
Không giải chính xác được: Φ không khả vi, **không có nhãn cho bước trung gian**, tập train nhỏ, số lần gọi LM có hạn.

**Hai công cụ có sẵn trong DSPy**, cùng dựa trên **bootstrap**: chạy chương trình và giữ lại **trace** (input/output từng module) của những lần cho **đáp án cuối đúng**.
- **BootstrapFewShotRS (BFRS)**, tối ưu Π: chọn vài trace làm **ví dụ few-shot** cho mỗi module, thử nhiều tổ hợp, chọn tổ hợp tốt nhất trên tập validation.
- **BootstrapFinetune (BFT)**, tối ưu Θ: gộp **mọi** trace đúng (với prompt thay bằng bản vanilla) thành dữ liệu, **fine-tune LoRA** cho LM.

**Ý tưởng — BetterTogether.** Cho **cùng một LM tự dạy chính nó** bằng cách xen kẽ:
```
Π′ ← tối ưu prompt                         (prompt tốt → nhiều trace đúng hơn)
Θ′ ← fine-tune trên trace của Φ⟨Θ,Π′⟩       (dữ liệu fine-tune tốt hơn)
Π″ ← tối ưu prompt lại trên model Θ′        (điều chỉnh hành vi sau fine-tune)
```
Paper so sánh 8 chiến lược: Zero-shot, Π, Θ, Π→Π, Θ→Θ (đối chứng tốn hai vòng như nhau), và Π→Θ, Θ→Π, Π→Θ→Π.

**Thí nghiệm.**
- **HotPotQA**: 3 module CoT + ColBERTv2, Wikipedia 5 triệu đoạn.
- **GSM8K**: 1 module CoT.
- **Iris**: 1 module CoT, phân loại dựa trên đặc trưng số.
- Model: Mistral-7B-Instruct-v0.2, Llama-2-7b-chat, Llama-3-8B-Instruct.
- Kết quả là trung bình 3 seed trên tập test.

**Kết quả chính (Bảng 1).**
- Ở **7/9** cặp dataset × LM, chiến lược tốt nhất dùng **cả Π và Θ**; nhưng **không có thứ tự nào thắng tuyệt đối** giữa Π→Θ, Θ→Π, Π→Θ→Π.
- **Tối ưu prompt là thiết yếu ở mọi task.** Chỉ fine-tune (Θ) từ zero-shot gần như không giúp, vì dữ liệu tự sinh từ prompt yếu cũng yếu.
- Trung bình theo LM và task: hơn chỉ trọng số tới **60%**, hơn chỉ prompt tới **6%**. Theo task: **5–78%** (HotPotQA), **2.5–10%** (GSM8K), **3.5–88%** (Iris).

| Ví dụ (Mistral-7B) | HotPotQA | GSM8K | Iris |
|---|---|---|---|
| Zero-shot | 17.2 | 40.3 | 26.0 |
| Chỉ prompt (Π) | 33.8 | 46.4 | 57.3 |
| Chỉ trọng số (Θ) | 22.9 | 40.7 | 29.3 |
| Tốt nhất của BetterTogether | **37.6** (Π→Θ→Π) | **48.3** (Θ→Π) | **66.7** (Θ→Π) |

**Hạn chế (tác giả tự nêu).** Chỉ 9 trường hợp; chỉ thử LoRA; **chưa hiểu vì sao** cả hai bước đều cần.

---

## Thông số cài đặt (Phụ lục C — để tái hiện)

| Mục | Giá trị trong paper |
|---|---|
| Chia dữ liệu | HotPotQA: 1000 train / 500 dev / 1500 test. GSM8K: 1000 / 500 / 1319. Iris: 150 mẫu chia đều 3 phần |
| Dữ liệu cho tối ưu prompt | HotPotQA và GSM8K: 100 train + 250 validation; Iris: 15 + 35 (không chồng nhau) |
| BFRS | 6 candidate program, tối đa 3 ví dụ few-shot mỗi module |
| LoRA | layer query và key; rank 32, alpha 64, dropout 0 |
| Train | 5 epoch, bfloat16, lr 1e-5, batch hiệu dụng 8 |
| Suy luận | HuggingFace text-generation-inference; temperature 0.1; tối đa 1024 token |
| Chi phí | Khoảng 75 giờ GPU A100 cho cả Bảng 1 |

## Điểm cần nhấn khi thuyết trình
1. "Không có nhãn trung gian": dùng ví dụ HotPotQA, hỏi khán giả *"truy vấn tìm kiếm đúng là gì?"*.
2. Bootstrap = giữ trace dẫn tới đáp án đúng, tức có nhãn trung gian "miễn phí".
3. Cùng một nguồn trace có hai cách dùng: **few-shot (prompt)** hoặc **LoRA (trọng số)**. Paper hỏi: sao không dùng cả hai?
4. Bảng 1: chỉ ra dòng Θ gần như không tăng so với zero-shot, nhưng Π→Θ thì tăng. Đây là bằng chứng "prompt trước, fine-tune sau".
5. Ví dụ trong Bảng A.1: suy luận ở hop 1 **sai** nhưng đáp án cuối **đúng**, vẫn được giữ làm dữ liệu. Đây là điểm yếu của bootstrap, dùng làm câu hỏi thảo luận.
6. Liên hệ với LoPA: LoPA là soft prompt, không sửa model; BetterTogether là prompt chữ + LoRA, sửa model nhưng tối ưu được cả pipeline nhiều bước.
