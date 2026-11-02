# DSPy — nền tảng cho Paper 2

| | |
|---|---|
| Tên đầy đủ | *DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines* (ICLR 2024). Bản arXiv v1 có tên *…into Self-Improving Pipelines* |
| Tác giả | Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts |
| Hội nghị | **ICLR 2024** |
| Link | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html) · [arXiv 2310.03714](https://arxiv.org/abs/2310.03714) |
| PDF trong repo | [`dspy-arxiv-2310.03714v1.pdf`](dspy-arxiv-2310.03714v1.pdf). Đây là **bản arXiv v1 (10/2023)**, chưa phải bản ICLR; số liệu có thể khác chút |
| Code / tài liệu | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) · [dspy.ai](https://dspy.ai) |
| Dịch chi tiết | [`dich-chi-tiet.md`](dich-chi-tiet.md) |
| Giải thích dễ hiểu + code mẫu | [`../../docs/01-chon-paper-va-thiet-ke-app.md`](../../docs/01-chon-paper-va-thiet-ke-app.md), mục 3 |

---

## Tóm tắt 1 trang

**Vấn đề.** Pipeline LM hiện nay được cài bằng **prompt template viết cứng**: chuỗi dài, tìm ra bằng thử–sai. Cách này giòn (đổi LM, đổi pipeline là hỏng) và giống như **chỉnh tay trọng số** của một bộ phân loại.

**Ý tưởng.** Lập trình thay vì viết prompt, lấy cảm hứng từ PyTorch:

| Khái niệm DSPy | Vai trò | Tương đương PyTorch |
|---|---|---|
| **Signature** | Khai báo input → output bằng ngôn ngữ tự nhiên (`"question -> answer"`) | Kiểu input/output của layer |
| **Module** | Kỹ thuật prompting có tham số: `Predict`, `ChainOfThought`, `ReAct`, `MultiChainComparison`… | Layer |
| **Program** | Ghép module trong `forward()` bằng Python bình thường (define-by-run) | `nn.Module` |
| **Teleprompter** (nay gọi là *optimizer*) | Nhận chương trình + vài ví dụ train + metric, trả về chương trình tối ưu | Optimizer + vòng train |

**Tham số của một module** gồm: LM nào, câu hướng dẫn, và **quan trọng nhất là các ví dụ minh hoạ**, dùng làm few-shot (LM freeze) hoặc dữ liệu train (fine-tune).

**Compiler, 3 giai đoạn.**
1. **Sinh ứng viên**: chạy chương trình (hoặc một *teacher*) trên ví dụ train, ghi **trace**, giữ trace qua được metric. Nhờ vậy có nhãn cho mọi bước dù chỉ có nhãn cuối (`BootstrapFewShot`).
2. **Tối ưu tham số**: chọn tổ hợp ví dụ tốt nhất (random search, Optuna), hoặc **fine-tune** trọng số bằng chính các ví dụ đó (`BootstrapFinetune`).
3. **Tối ưu bậc cao**: đổi luồng chương trình, vd. **ensemble** + bỏ phiếu.

**Kết quả chính.**

| | GPT-3.5 | Llama2-13b-chat |
|---|---|---|
| GSM8K: chương trình đơn giản → sau compile tốt nhất | 33% → **82%** | 9% → **47%** |
| HotPotQA: chương trình đơn giản → sau compile tốt nhất | 32% → **46%** | 22% → **41%** |

- GSM8K: bootstrap **bằng hoặc vượt** chuỗi suy luận do người viết. Ghép đúng module nâng LM từ 4–20% lên **49–88%**.
- HotPotQA: chương trình `multihop` + bootstrap tốt nhất; compile giúp **Llama2-13b cạnh tranh với GPT-3.5**. Fine-tune được cả **T5-Large 770M** đạt 39.3% EM chỉ với 200 câu có nhãn + 800 câu không nhãn.
- Compile mất **vài phút tới vài chục phút**.
- Mã nguồn LangChain (09/2023) có 50 prompt dài hơn 1000 ký tự; **DSPy không có prompt viết tay nào**.

## Liên hệ với BetterTogether
- BetterTogether dùng **chính** `BootstrapFewShotWithRandomSearch` (tối ưu prompt) và `BootstrapFinetune` (fine-tune) của DSPy.
- DSPy gốc fine-tune một model **khác, nhỏ hơn** (T5) bằng trace của model lớn (Llama2-13b).
- BetterTogether: **cùng một model** tự dạy mình, và **xen kẽ** hai bước.

## Lưu ý về phiên bản API
Code trong paper là API DSPy năm 2023 (`dspy.Retrieve`, field `rationale`, `dspy.settings.trace`…). Thư viện hiện tại đổi nhiều: `dspy.LM`, field `reasoning`, *teleprompter* đổi tên thành *optimizer*… Khi viết code cho app, theo tài liệu tại https://dspy.ai và cố định phiên bản trong `app/requirements.txt`.
