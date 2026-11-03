# LoPA — Prompt Tuning Strikes Back

| | |
|---|---|
| Tên đầy đủ | *Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation* |
| Tác giả | Abhinav Jain (Rice), Swarat Chaudhuri (UT Austin), Thomas Reps (UW–Madison), Chris Jermaine (Rice) |
| Hội nghị | **NeurIPS 2024** (38th Conference on Neural Information Processing Systems) |
| Link | [arXiv 2405.15282](https://arxiv.org/abs/2405.15282) · PDF trong repo: [`lopa-neurips2024.pdf`](lopa-neurips2024.pdf) |
| Code | https://github.com/jabhinav/Prompt-Tuning-Strikes-Back-with-LOPA |
| Dịch chi tiết | [`dich-chi-tiet.md`](dich-chi-tiet.md) |

---

## Tóm tắt 1 trang

**Vấn đề.** Để tuỳ biến một foundation model (FM) cho nhiều người dùng, các phương pháp PEFT phổ biến như LoRA cần **lưu một adapter riêng cho mỗi task trên server**. Mỗi lần xử lý một batch request, server phải chọn và ghép đúng adapter, nên khó mở rộng. *Prompt tuning* không gặp vấn đề này: soft prompt chỉ là phần thêm vào **đầu input**, server xử lý mọi task giống nhau. Nhưng prompt tuning **kém hơn** LoRA khá xa.

**Câu hỏi.** Có thể làm prompt tuning mạnh ngang LoRA mà vẫn giữ ưu điểm "không đụng vào server" không?

**Phương pháp — LoPA (Low-rank Prompt Adaptation).** Soft prompt `Z` (m vector, mỗi vector d chiều) được ghép từ hai phần:

```
Z = Z_S ∘ g(Z_I),     g = sigmoid,     ∘ = nhân từng phần tử
Z_I = u × vᵀ,         u = MLP_u(X′) ∈ R^{d×r},  v = MLP_v(X′) ∈ R^{m×r}
X′ = biểu diễn của input lấy từ một Encoder model nhỏ (RoBERTa / CodeSage)
```
- `Z_S` (*task-specific*): chung cho mọi input của task, giống prompt tuning thường.
- `Z_I` (*instance-specific*): khác nhau theo từng input, được **phân rã low-rank** (hạng r ≪ d, m) để ít tham số.
- `Z` được nối vào trước embedding của input rồi đưa vào FM đã **freeze**. Chỉ train `Z_S`, `MLP_u`, `MLP_v`.
- Vì `Z` tính được ở bất kỳ đâu, kể cả phía client, **server không cần lưu gì riêng cho từng task**.

**Vì sao tốt hơn.**
1. Với prompt tuning thường, các vector `W^V z_k` cố định, chỉ trọng số attention thay đổi theo input. Với LoPA, chính các vector đó cũng thay đổi theo input, nên ảnh hưởng tới attention ở các tầng sâu linh hoạt hơn.
2. Cổng sigmoid làm gradient của `Z_S` và `Z_I` **phụ thuộc lẫn nhau**. Ngược lại, IDPG tương đương `Z_S + Z_I` nên hai phần học độc lập.

**Kết quả chính.**
- GLUE (RoBERTa-355M): LoPA trung bình **90.53**, gần FFT (91.35) và LoRA (91.26); hơn prompt tuning 28.62 điểm, hơn DePT 25.39 điểm, hơn S-IDPG 5.36 điểm. Dùng 1.60M tham số so với 2.36M của LoRA.
- Code (CruxEval-I/O, MBPP; 6 backbone 350M–8B): thường cách LoRA 1–4 điểm pass@1 với khoảng 2/3 số tham số; **thắng LoRA ở CruxEval-I** với mọi model trừ Llama-3.
- Tổng hợp mọi task và model: hơn PT **28.52%**, hơn IDPG **20.16%** (tương đối), chỉ thua LoRA **0.54%**; **thắng LoRA 11/24 trường hợp**.

**Ablation.**
- Độ dài prompt m: LoPA ổn định khi đổi m (Z_S đóng vai trò regularizer); LoPA với m = 5 đã hơn PT và IDPG dùng m lớn hơn.
- Rank r: NLU tăng r từ 1 → 4 thì tốt hơn (MRPC +8%); CruxEval ít dữ liệu nên tăng r không giúp (dễ overfit).
- Encoder: CodeSage hơn CodeBERT khoảng 2 điểm; fine-tune encoder không giúp đáng kể.
- Hàm ghép: `concat` kém nhất (83.44), `max` 88.68, **gating** tốt nhất (90.53).

**Hạn chế (tác giả tự nêu).** Mới chứng minh trên benchmark, chưa chắc đúng với task thực tế "lạ"; mới thử đặt prompt ở đầu input.

---

## Thông số cài đặt (để tái hiện)

| Mục | Giá trị trong paper |
|---|---|
| Số virtual token m | 10 (mọi baseline soft prompt) |
| Rank r | chọn tốt nhất trong {1, 2, 4} |
| Encoder | NLU: RoBERTa + MLP (h = 256); Code: CodeSage-125M + MLP (h = 1024) |
| Learning rate LoPA | 1e-5 (NLU), 1e-3 (code) |
| LR baseline | FFT 1e-5; LoRA và các soft prompt khác 1e-4 |
| Epoch | NLU: FFT/LoRA 10, soft prompt 20. MBPP 10. CruxEval: < 7B 20, ≥ 7B 10, FFT 5 |
| Chia dữ liệu | GLUE: split chuẩn; MBPP, CruxEval: 50-50 |
| Đánh giá | Accuracy/F1 (GLUE); pass@1, temperature 0.6 (MBPP), 0.2 (CruxEval) |
| Phần cứng | 2 × A100 40GB |

## Điểm cần nhấn khi thuyết trình
1. Hình 1 (vấn đề adapter ở server) → Hình 2 (kiến trúc LoPA): hai hình này kể được cả câu chuyện.
2. Một công thức duy nhất: `Z = Z_S ∘ σ(u·vᵀ)`.
3. Bảng 1: chỉ cho khán giả xem nhóm "dưới đường chấm" (không sửa server). LoPA là phương pháp duy nhất trong nhóm này đạt ngang LoRA.
4. Câu hỏi phản biện nên chuẩn bị:
   - Encoder chạy ở client thì client cũng phải có GPU? (Encoder nhỏ, 125M–355M.)
   - Gửi `Z` (m × d số thực) qua mạng có tốn không?
   - Vì sao không so với LoRA phục vụ nhiều adapter hiệu quả như S-LoRA?
