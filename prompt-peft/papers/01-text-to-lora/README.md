# Text-to-LoRA (T2L) — Instant Transformer Adaption

| | |
|---|---|
| Tên đầy đủ | *Text-to-LoRA: Instant Transformer Adaption* |
| Tác giả | Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert T. Lange (Sakana AI) |
| Hội nghị | **ICML 2025** (Proceedings of the 42nd ICML, PMLR 267, trang 7485–7514) |
| Link | [PMLR](https://proceedings.mlr.press/v267/charakorn25a.html) · [arXiv 2506.06105](https://arxiv.org/abs/2506.06105) · [Blog Sakana](https://sakana.ai/text-to-lora/) · PDF trong repo: [`text-to-lora-icml2025.pdf`](text-to-lora-icml2025.pdf) |
| Code | [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) (Apache 2.0) · checkpoint: [HF SakanaAI/text-to-lora](https://huggingface.co/SakanaAI/text-to-lora) |
| Dịch chi tiết | [`dich-chi-tiet.md`](dich-chi-tiet.md) |

---

## Tóm tắt 1 trang

**Vấn đề.** Muốn LLM làm tốt một task riêng, thường phải train một **LoRA adapter cho mỗi task**: cần gom dữ liệu, chọn siêu tham số, train lâu, và mỗi task một adapter riêng không chia sẻ kiến thức.

**Câu hỏi.**
1. Có thể train end-to-end một mạng neural để **nén hàng trăm LoRA** có sẵn không?
2. Có thể **sinh LoRA mới cho task chưa từng thấy** chỉ từ **mô tả task bằng ngôn ngữ tự nhiên** không?

**Phương pháp — T2L là một hypernetwork** (mạng sinh trọng số cho mạng khác):
```
ΔW_{m,l} = h_θ( concat[ f(z), E[m], E[l] ] )
  z    : mô tả task bằng chữ (vd. "This task challenges your mathematical reasoning ...")
  f(z) : embedding của mô tả (gte-large-en-v1.5)
  E[m] : embedding học được của loại module (q_proj, v_proj)
  E[l] : embedding học được của chỉ số layer
  h_θ  : MLP → sinh ma trận A, B của LoRA cho module m, layer l
```
- Gộp mọi (m, l) vào một batch → sinh **toàn bộ LoRA trong một lần forward**, mất vài giây.
- Ba kích thước: **L** (sinh A và B cùng lúc, 55M tham số), **M** (head dùng chung cho A hoặc B, 34M), **S** (sinh từng rank một, 5M). Một LoRA rank 8 trên q, v của Mistral-7B có 3.4M tham số.

**Hai cách train T2L.**
- **Reconstruction**: học tái tạo các LoRA có sẵn (loss L1 giữa LoRA sinh ra và LoRA đích). Rẻ: 9 LoRA khoảng 10 phút, 479 LoRA khoảng 10 giờ.
- **SFT**: train thẳng trên dữ liệu của 479 task SNI (gradient lan qua LLM gốc tới hypernetwork). **Khái quát sang task mới tốt hơn hẳn.**

**Kết quả chính** (Mistral-7B-Instruct, 10 benchmark: ARC-C/E, BoolQ, GSM8K, HellaSwag, OpenBookQA, PIQA, Winogrande, HumanEval, MBPP).
- **Nén LoRA**: T2L tái tạo 9 LoRA benchmark đạt **73.3–73.5** điểm trung bình, ngang LoRA riêng từng task (73.3). Có chỗ còn hơn, nhờ nén có mất mát hoạt động như regularization.
- **Zero-shot trên task chưa thấy** (Bảng 2, TB 10 task):

| Phương pháp | TB 10 task |
|---|---|
| Mistral-7B gốc | 55.8 |
| + mô tả task trong prompt | 60.6 |
| + 3-shot ICL | 61.0 |
| Multi-task LoRA | 66.3 |
| Hyperdecoders (theo từng input) | 67.3 |
| **T2L (SFT) L** | **67.7** |

- Tổng quát sang model khác: **Llama-3.1-8B** 73.2 → **76.9**; **Gemma-2-2B** (kèm ICL) 60.7 → **66.4**.
- Rẻ: sinh LoRA + trả lời tốn **0.86 TFLOPs**/câu, so với **4.18 TFLOPs** khi dùng 3-shot ICL (rẻ hơn hơn 4 lần).
- **Điều khiển bằng mô tả**: mô tả khớp với task → LoRA tốt; mô tả kém hoặc ngẫu nhiên → kém hơn. Hai mô tả khác nhau cho ra hai cách suy luận khác nhau mà đều đúng (Hình 4).

**Hạn chế (tác giả tự nêu).** Chưa đạt tới LoRA train riêng cho từng task; phụ thuộc chất lượng mô tả; chỉ sinh LoRA (có thể có cách điều chỉnh LLM hiệu quả hơn).

---

## Thông số (để chạy demo và tái hiện)

| Mục | Giá trị |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 (chính); Llama-3.1-8B-Instruct, Gemma-2-2B-Instruct (Phụ lục A) |
| LoRA đích | rank 8, alpha 16, q_proj + v_proj, dropout 0.05, rsLoRA |
| Encoder mô tả | gte-large-en-v1.5 (1024 chiều); thử thêm Mistral (4096 chiều), kết quả tương đương |
| Dữ liệu train | 479 task Super-NaturalInstructions (SNI), mỗi task 128 mô tả do GPT-4o-mini sinh |
| LR | LoRA riêng 8e-5; T2L SFT 2.5e-5; T2L recon 1e-3 |
| Phần cứng | Mọi model vừa **1 GPU H100 80GB**. SFT tốn bộ nhớ hơn nhiều vì phải lan gradient qua LLM; **reconstruction chạy được trên GPU phổ thông** |
| Train T2L SFT đầy đủ | README của code: khoảng **5 ngày trên 1 H100** mỗi model → **nhóm không tự train, dùng checkpoint có sẵn** |
| Chạy demo | README: cần GPU **> 16GB** để chạy cùng lúc Mistral-7B và T2L; với **Gemma-2-2B** dùng GPU nhỏ hơn được |

## Điểm cần nhấn khi thuyết trình
1. **Hình 1 + Hình 2**: "gõ một câu mô tả → có ngay LoRA". Đây là cầu nối trực tiếp giữa **Prompting** (đầu vào là chữ) và **PEFT** (đầu ra là LoRA).
2. **Bảng 2**: T2L hơn multi-task LoRA và ICL mà không cần dữ liệu của task mới.
3. **Hình 4 (GSM8K)**: mô tả "solve this task please" → sai; mô tả chi tiết → đúng. Đây là bằng chứng "viết mô tả tốt = LoRA tốt", tức prompting vẫn quan trọng.
4. **Hình 5 (t-SNE)**: LoRA của các task giống nhau nằm gần nhau (MBPP cạnh HumanEval).
5. **Reconstruction vs SFT** (Bảng 6 + Phụ lục D): các LoRA làm cùng việc **không nằm gần nhau** trong không gian trọng số, nên học "bắt chước LoRA" không khái quát được. Đây là một kết luận thú vị để thảo luận.
6. Câu hỏi phản biện nên chuẩn bị:
   - Benchmark có giống dữ liệu train không? (Tác giả đã loại 10 task SNI bị trùng; ARC tương tự SNI task #47.)
   - Mô tả do GPT-4o-mini viết; người dùng thật viết kém hơn thì sao?
   - Mô tả tiếng Việt có dùng được không? (Ngoài phạm vi paper, mọi dữ liệu train là tiếng Anh. Có thể thử như một thí nghiệm của nhóm.)
