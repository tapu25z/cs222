# Text-to-LoRA — Dịch và diễn giải chi tiết

> Bản dịch có diễn giải của *Text-to-LoRA: Instant Transformer Adaption* (Charakorn, Cetin, Tang, Lange — Sakana AI, ICML 2025). PDF: [`text-to-lora-icml2025.pdf`](text-to-lora-icml2025.pdf).
> Ô **💡 Giải thích** là phần nhóm thêm vào, không có trong paper. Số liệu bảng được đối chiếu với bản PDF ICML.

---

## Abstract

Foundation Model là công cụ chung để tạo nội dung nhanh, nhưng thường vẫn cần **thích nghi theo từng task**. Cách truyền thống là gom dữ liệu cẩn thận rồi fine-tune model nhiều lần. Fine-tuning giúp áp dụng foundation model cho nhiều ứng dụng mới, nhưng **tốn kém, lâu** và **nhạy với siêu tham số**.

Để khắc phục, tác giả giới thiệu **Text-to-LoRA (T2L)**, một model có thể thích nghi LLM **ngay lập tức chỉ dựa trên mô tả task bằng ngôn ngữ tự nhiên**. T2L là một **hypernetwork** được train để tạo LoRA chỉ bằng **một lần forward rẻ**.
- Sau khi train T2L trên 9 LoRA đã có (GSM8K, Arc…), các LoRA được T2L tái tạo đạt hiệu năng **ngang adapter riêng của từng task** trên tập test tương ứng.
- T2L có thể **nén hàng trăm LoRA** và **khái quát zero-shot** sang task hoàn toàn mới.

Đây là bước tiến tới việc **phổ cập hoá** việc chuyên biệt hoá foundation model, cho phép **thích nghi bằng ngôn ngữ** với chi phí tính toán tối thiểu. Code: https://github.com/SakanaAI/text-to-lora.

---

## 1. Introduction (Giới thiệu)

Hệ sinh học có thể thích nghi nhanh chỉ với tín hiệu cảm giác ít ỏi; ví dụ mắt người điều chỉnh độ nhạy sáng và tiêu điểm qua điều biến thần kinh. LLM hiện nay có nhiều khả năng và kiến thức nhưng **cứng nhắc** khi cần thêm năng lực riêng cho task. Khi đó người ta thường train lại một phần model bằng PEFT, ví dụ **LoRA**.

Thông thường **mỗi task phải tối ưu một LoRA riêng**, cần dữ liệu và siêu tham số riêng. Cách này hạn chế việc chia sẻ kiến thức giữa các task và tốn công kỹ thuật. Gần đây người ta thấy ma trận low-rank của LoRA còn có thể **nén thêm** (train bản có mất mát mà vẫn giữ hiệu năng), và có thể **kết hợp nhiều LoRA** cho task mới lúc suy luận. Cốt lõi các cách này là dùng rõ ràng kỹ thuật phân rã hoặc giảm chiều (SVD, routing). Từ đó nảy ra hai câu hỏi:
1. Có thể **train end-to-end** một mạng neural để **nén nhiều LoRA** đã train sẵn không?
2. Có thể **giải mã LoRA mới cho task chưa thấy** lúc test, **chỉ dựa vào hướng dẫn bằng ngôn ngữ tự nhiên** không?

Giả thuyết: các LoRA khác nhau **chia sẻ cùng một cơ chế thích nghi** bên dưới và có thể được tối ưu đồng thời mà không cần cấu trúc hay công thức kết hợp tường minh. Để kiểm chứng, tác giả đề xuất **T2L**, một hypernetwork **nén LoRA riêng của task** và **sinh LoRA mới zero-shot** lúc suy luận.
- T2L được train để nén LoRA trên một phân phối task đa dạng từ **Super Natural Instructions (SNI)**.
- Quan trọng: đầu vào của T2L là **mô tả task bằng ngôn ngữ tự nhiên**, cho phép sinh LoRA zero-shot cho task chưa thấy.

Thực nghiệm cho thấy T2L train được bằng **tái tạo (reconstruction)** adapter có sẵn hoặc bằng **SFT** trên phân phối task. Sau khi train, T2L **vượt LoRA đa nhiệm (multi-task LoRA)** và **Arrow Routing** (phương pháp định tuyến LoRA zero-shot tốt nhất) trên nhiều benchmark, và sinh được LoRA cho task chưa thấy chỉ từ mô tả.

> **Hình 1.** *Trái*: tổng quan cách train T2L. Từ embedding của mô tả task, hypernetwork sinh ma trận ΔW cho từng task; trọng số T2L được tối ưu bằng chưng cất LoRA có sẵn hoặc SFT đa nhiệm. *Phải trên*: hiệu năng tương đối so với LoRA "oracle" trên task SNI khi tăng tỷ lệ nén. *Phải dưới*: hiệu năng sinh LoRA zero-shot trên 10 benchmark tăng khi tăng số dataset train, với cả 3 kiến trúc T2L.

**Đóng góp.**
1. Kiến trúc hypernetwork **sinh LoRA trong một lần forward** từ mô tả chữ; train được bằng chưng cất adapter có sẵn lẫn SFT đa nhiệm.
2. T2L **mã hoá hiệu quả hàng trăm LoRA**. Nén có mất mát nhưng vẫn giữ hiệu năng của adapter riêng; và khái quát sang task chưa thấy khi có mô tả phù hợp.
3. Ablation kỹ: T2L tốt lên khi tăng số dataset, ảnh hưởng của embedding mô tả, cách train, và nội dung mô tả.
4. Phân tích bản chất LoRA do T2L sinh: các cụm có nghĩa khi giảm chiều (5.5); bằng chứng vì sao T2L train bằng reconstruction không khái quát được (Phụ lục D).

---

## 2. Preliminaries (Kiến thức nền)

Có nhiều dataset fine-tune `D = {D_1, …, D_T}` tương ứng các task `T = {t_1, …, t_T}`. Để train T2L, giả sử mỗi dataset có một tập **mô tả task bằng ngôn ngữ tự nhiên** `Z^i = {z_1^i, …, z_m^i}`, tức `D_i = {X^i, Y^i, Z^i}`. Mô tả **không cần riêng cho từng mẫu**, chỉ là mô tả chung cho dataset.

Với một task t_i, mục tiêu fine-tune LLM có trọng số Ψ:
```
ΔW^i = argmin_{ΔW^i} L_SFT(D_i, Ψ, ΔW^i)                       (1)
```
Với **multi-task**, train **một** adapter cho hợp của mọi dataset:
```
ΔW = argmin_{ΔW} E_{D_i ~ D} L_SFT(D_i, Ψ, ΔW)                  (2)
```

**LoRA.** Freeze trọng số gốc, chỉ học ma trận low-rank làm adapter. Với một phép biến đổi tuyến tính `h = W_0 x`, bản fine-tune là `h = W_0 x + ΔW x = W_0 x + Bᵀ A x`, với `A, B ∈ R^{r×d}`, r < d. Ký hiệu `ΔW_{m,l}` cho module loại m (vd. q_proj) ở layer l.

**Hypernetwork.** Mạng neural **sinh tham số cho một mạng "gốc"** khác (Ha et al., 2016). Nó là một **mã hoá gián tiếp** của mạng gốc vì có ít tham số hơn nhiều, nhờ học cách chia sẻ tham số gián tiếp. Cho vector mô tả layer φ_l, hypernetwork tham số θ sinh tham số layer l: `W_l = h_θ(φ_l)`. Truyền thống, φ_l là one-hot hoặc vector học được, và θ được train end-to-end trên task đích.

---

## 3. Text-to-LoRA: học nén và sinh LoRA

Với mỗi module đích (m) và layer (l), hypernetwork sinh hai ma trận A, B dựa trên mô tả `z^i` của task `t_i`:
```
ΔW^i_{m,l} = h_θ(φ^i_{m,l}),   φ^i_{m,l} = concat[ f(z^i), E[m], E[l] ]         (3), (4)
```
- f: biểu diễn vector của mô tả, thường là token CLS của transformer hai chiều hoặc activation của token cuối của một LLM.
- E: từ điển embedding học được, tra theo loại module m hoặc chỉ số layer l.

Viết gọn `ΔW^i := h_θ(φ^i)`. Mục tiêu **SFT** cho T2L:
```
θ = argmin_θ E_{D_i ~ D, z^i ~ Z^i} L_SFT(D_i, Ψ, h_θ(φ^i))                     (5)
```
Các giá trị m, l có thể **gộp batch**, nên T2L sinh ΔW cho **mọi module và layer trong một lần forward**.

> 💡 **Giải thích.** Thay vì train trực tiếp A, B cho mỗi task (như LoRA thường), ta train **một máy sinh A, B**. Đầu vào của máy là "task này là gì" (mô tả chữ) và "đang sinh cho chỗ nào" (module, layer). Train xong, gặp task mới chỉ cần viết mô tả là máy sinh ra LoRA.

### 3.1 Các kiến trúc T2L

Phần lớn tham số của hypernetwork nằm ở **layer output**, vì nó tỷ lệ với kích thước trọng số đích. Để khảo sát đánh đổi độ phức tạp và hiệu năng, tác giả đề xuất 3 biến thể dùng chung backbone, chỉ khác head và embedding:
- **L** (lớn nhất): layer cuối sinh **cả A và B cùng lúc**; `|θ_head| = d_out × 2 × r × d`.
- **M** (vừa): head **dùng chung** cho A và B, sinh A hoặc B tuỳ một embedding học được; `|θ_head| = d_out × r × d`.
- **S** (nhỏ nhất, inductive bias mạnh nhất): mỗi lần chỉ sinh **một rank** của ma trận; `|θ_head| = d_emb × d`.

Tham khảo: một LoRA có `r × d × 2 × L × |M|` tham số (L layer, |M| module đích). Mặc định `d_out = 512`. Mọi kiến trúc đều sinh được toàn bộ A, B trong một lần forward nhờ gộp batch các embedding. Chi tiết ở Phụ lục F; cách khởi tạo giúp train ổn định ở Phụ lục G.

> **Hình 2.** Các biến thể kiến trúc T2L. Embedding mô tả task qua *task encoder*, nối với *module emb* và *depth (layer) emb*, qua các khối MLP; M và S thêm *A/B emb*, S thêm *rank emb*. Output một lần forward: L → `[2, r, d]`, M → `[r, d]`, S → `[d]`.

### 3.2 Train T2L bằng tái tạo LoRA (reconstruction)

Cách đơn giản nhất: **tái tạo các LoRA đã train sẵn**. Cách này tận dụng được thư viện LoRA công khai, hoặc làm hai giai đoạn (train thư viện LoRA trước, rồi train T2L tái tạo chúng).
- Nếu chỉ để nén, có thể dùng one-hot hoặc vector học được làm embedding task; nhưng như vậy **không sinh được LoRA cho task mới**.
- Để sinh zero-shot, T2L được điều kiện thêm bằng **embedding của mô tả chữ**.

Với thư viện LoRA Ω, loss tái tạo:
```
L(Ω, θ) = E_{ΔW^i ~ Ω} | ΔW^i − h_θ(φ^i) |                                      (6)
```

### 3.3 Train T2L bằng SFT

T2L cũng có thể tối ưu **trực tiếp trên dataset fine-tune**. Cách này **không cần LoRA đích trung gian** và train end-to-end. Nên dùng khi các LoRA đã train **không tụ lại theo chức năng**: ví dụ t_1 và t_2 là hai task liên quan, cần cùng một năng lực, nhưng ΔW^1 và ΔW^2 có thể nằm ở **hai cực tiểu khác nhau**. Khi đó T2L tái tạo phải nén hai ma trận khác nhau về số, nên khó khái quát. Thực tế T2L train bằng reconstruction **không khái quát được sang task mới** (mục 5.4), còn T2L train bằng SFT **tự học cách gom cụm task**, điều đã được chứng minh giúp định tuyến LoRA zero-shot. Loss SFT là phương trình (5).

---

## 4. Experiments (Thí nghiệm)

**Mục tiêu.** Đánh giá các kiến trúc và cách train ở hai khía cạnh: **nén adapter** (4.1) và **sinh LoRA zero-shot cho task chưa thấy** (4.2).

**Baseline.**
- LoRA riêng của từng task (oracle), LoRA trung bình (trung bình từng phần tử), **multi-task LoRA** (một LoRA train trên mọi task train).
- **Hyperdecoders**: hypernetwork sinh LoRA **theo từng chuỗi input**, cài theo kiến trúc của tác giả.
- Tăng sức cho base model không fine-tune: **few-shot ICL** và **thêm mô tả task vào đầu câu hỏi**.
- Kết quả **Arrow Routing** lấy từ Ostapenko et al. (2024); chỉ so sánh gián tiếp vì dùng bộ LoRA, task train và có thể cả prompt đánh giá khác.

**Thiết lập.**
- Base LLM chính: **Mistral-7B-Instruct**; Bảng 7 dùng **Llama-3.1-8B-Instruct**, Bảng 8 dùng **Gemma-2-2b-Instruct**.
- Embedding mô tả: **gte-large-en-v1.5**.
- Mọi LoRA có **rank 8**, chỉ nhắm **query và value projection** ở mọi khối attention (tổng **3.4M** tham số). Khi đó T2L **L, M, S** có lần lượt **55M, 34M, 5M** tham số train.
- Dữ liệu train LoRA: SNI, 500 task (theo Brüel-Gabrielsson et al.). 11 task để validation, bỏ 10 dataset bị trùng với benchmark đánh giá, còn **479 dataset** để train. Mọi mẫu là tiếng Anh.
- **10 benchmark đánh giá**: Arc-challenge (ArcC), Arc-easy (ArcE), BoolQ (BQ), GSM8K, HellaSwag (HS), OpenBookQA (OQA), PIQA, Winogrande (WG), HumanEval (HE), MBPP.
  - Benchmark có phần giống task train (chủ yếu là trắc nghiệm, có miền trùng nhau; vd. ARC giống SNI task #47). Nhưng MBPP và HumanEval rất khác, vì task train **không có task sinh code nào**.
- Mô tả task cho dataset train và benchmark được **sinh hoàn toàn bằng LLM** (Phụ lục L). Khi dùng embedding mô tả, kết quả là **trung bình trên 3 mô tả** cho mỗi benchmark.

### 4.1 Nén LoRA (LoRA Compression)

Câu hỏi: T2L có **khôi phục hiệu năng** của LoRA đã train không, khi train bằng reconstruction? Để kiểm soát chất lượng, tác giả train một LoRA riêng (**oracle**) trên tập train của mỗi benchmark, tạo thành thư viện LoRA. Lưu ý các benchmark được T2L "thấy gián tiếp" khi train, vì T2L học chưng cất LoRA của chúng.

**Bảng 1.** T2L train bằng reconstruction trên 9 LoRA benchmark (ô xanh trong PDF = T2L hơn LoRA riêng).

| | ArcC | ArcE | BQ | GSM8K | HS | OQA | PIQA | WG | MBPP | TB (9 task) |
|---|---|---|---|---|---|---|---|---|---|---|
| Base model | 65.4 | 77.8 | 71.6 | 40.9 | 49.7 | 54.2 | 72.8 | 45.0 | 43.1 | 55.8 |
| *Embedding one-hot* | | | | | | | | | | |
| T2L (Recon) L | 76.4 | 89.9 | 89.4 | **53.8** | 92.6 | 85.0 | 69.7 | **51.2** | **52.6** | **73.4** |
| T2L (Recon) M | **76.7** | 89.9 | 89.4 | 53.2 | 92.6 | 85.0 | 69.9 | **51.4** | **52.9** | **73.4** |
| T2L (Recon) S | 75.2 | 88.8 | 87.4 | 50.9 | 89.1 | 75.6 | **83.9** | **58.1** | 48.1 | 73.0 |
| *Embedding mô tả task* | | | | | | | | | | |
| T2L (Recon) L | 76.6 | 89.8 | 89.4 | **53.9** | 92.6 | 85.0 | 69.6 | **51.2** | 51.8 | 73.3 |
| T2L (Recon) M | 76.5 | 89.9 | 89.4 | **53.9** | 92.5 | 84.9 | **70.4** | **51.6** | **52.8** | **73.5** |
| T2L (Recon) S | 75.4 | 88.8 | 87.8 | 49.1 | 89.7 | 76.7 | **84.2** | **56.9** | 48.0 | 73.0 |
| LoRA riêng từng task | 76.6 | 89.9 | 89.4 | 53.5 | 92.6 | 85.0 | 69.9 | 51.1 | 52.1 | 73.3 |

(Số in đậm ở đây là các ô được tô xanh trong PDF.)

T2L **khôi phục hoàn toàn** hiệu năng của adapter oracle với cả hai loại embedding, và **còn vượt** LoRA riêng ở một số benchmark. Giả thuyết: lợi ích đến từ **nén có mất mát**, hoạt động như **regularization** cho trọng số LoRA đã train. Theo tác giả, hiệu ứng rõ nhất ở PIQA và WG, nơi LoRA oracle bị overfit và kém hơn base model.

> 💡 Theo bảng, LoRA oracle trên PIQA (69.9) thấp hơn base model (72.8). Với WG, oracle (51.1) vẫn cao hơn base (45.0), nhưng thấp hơn nhiều so với T2L S (56.9–58.1).

**Tăng số task cần nén (Hình 3).** Với T2L dùng embedding one-hot, tác giả train trên {16, 32, 64, 128, 256, 479} task và đo trên 10 task SNI train. Số task tăng thì sai số tái tạo L1 tăng.
- Khi loss tái tạo < 10⁻⁴, T2L khôi phục đầy đủ hiệu năng oracle; sai số tăng thì hiệu năng giảm. Vậy T2L học một **nén có mất mát**.
- Dù vậy, mọi kiến trúc vẫn giữ được khoảng **65%** hiệu năng oracle, và không giảm thêm kể cả khi sai số L1 mỗi phần tử > 8×10⁻⁴.
- Mục 5.1 cho thấy tăng số task train **có lợi** trong thiết lập SFT.

### 4.2 Sinh LoRA zero-shot (Zero-Shot LoRA Generation)

T2L được train bằng **SFT trên 479 task SNI**, mỗi task **128 mô tả**. Với mỗi mẫu trong minibatch, lấy ngẫu nhiên một mô tả của dataset tương ứng (online).

**Bảng 2.** Zero-shot trên benchmark chưa thấy. Kết quả T2L là trung bình 3 LoRA sinh từ 3 mô tả khác nhau. (In đậm: hơn multi-task LoRA.)

| Phương pháp | ArcC | ArcE | BQ | HS | OQA | PIQA | WG | MBPP | TB (8) | GSM8K | HE | TB (10) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Không thích nghi lúc test* | | | | | | | | | | | | |
| Mistral-7B-Instruct | 65.4 | 77.8 | 71.6 | 49.7 | 54.2 | 72.8 | 45.0 | 43.1 | 60.0 | 40.9 | 37.2 | 55.8 |
| Thêm mô tả task vào prompt | 72.0 | 85.8 | 67.6 | 58.9 | 63.4 | 77.9 | 59.0 | 41.6 | 65.8 | 40.9 | 39.0 | 60.6 |
| 3-shot ICL | 72.1 | 85.9 | 71.7 | 59.0 | 66.2 | 76.2 | 58.0 | 42.6 | 66.5 | 40.9 | 37.2 | 61.0 |
| LoRA trung bình | 70.7 | 84.4 | 75.4 | 59.9 | 59.0 | 78.0 | 54.3 | 47.1 | 66.1 | 42.4 | 37.8 | 60.9 |
| Multi-task LoRA | 76.2 | 88.3 | 85.5 | 65.2 | 68.0 | 81.8 | 62.4 | 48.1 | 71.9 | 47.5 | 39.6 | 66.3 |
| *Thích nghi zero-shot* | | | | | | | | | | | | |
| Arrow Routing | 60.9 | 86.2 | **87.6** | **80.8** | 48.6 | 83.0 | 68.5 | 50.2 | 70.7 | N/A | 28.7 | N/A |
| Hyperdecoders (theo input) | **76.6** | **88.5** | 83.9 | 65.2 | **76.6** | 81.3 | 64.9 | 51.6 | **73.6** | 43.6 | **40.9** | **67.3** |
| T2L (SFT) S | 76.0 | **88.7** | 83.8 | **68.0** | **71.6** | 82.3 | 61.0 | 41.2 | 71.6 | 47.3 | 39.0 | 65.9 |
| T2L (SFT) M | **77.2** | **89.0** | 84.3 | 65.1 | **76.1** | 81.8 | 64.0 | 50.5 | **73.5** | 45.2 | **41.3** | **67.5** |
| T2L (SFT) L | **77.5** | **88.9** | 85.0 | **66.5** | **75.5** | 82.1 | 64.2 | 51.9 | **73.9** | 45.8 | 39.2 | **67.7** |
| *Oracle* | | | | | | | | | | | | |
| LoRA riêng từng task | 76.6 | 89.9 | 89.4 | 92.6 | 85.0 | 69.9 | 51.1 | 52.1 | 75.8 | 53.5 | N/A | N/A |

Kết quả:
- Multi-task LoRA làm tốt trên benchmark dù không fine-tune thêm, nhưng vẫn còn khoảng cách với LoRA riêng.
- T2L train bằng SFT **sinh ra LoRA hữu ích**, cải thiện so với multi-task LoRA **một cách nhất quán** trên nhiều benchmark.
- Dù chưa xoá được hoàn toàn khoảng cách với LoRA riêng, T2L **còn vượt oracle trên một số task** (ô xanh trong PDF: ArcC, PIQA, WG).
- Phụ lục A thử thêm Llama và Gemma. Một ưu điểm chính của T2L là **hiệu quả**; phân tích FLOPs ở Phụ lục I.

---

## 5. Ablations and Analyses (Phân tích thành phần)

### 5.1 Tăng compute theo số task train

Tăng số task train và tăng ngân sách train tương ứng. **Bảng 3**: T2L nói chung **tốt lên** khi có thêm task. Riêng **S không hưởng lợi** khi train 479 task, có lẽ do dung lượng model hạn chế.

| Kiến trúc | Số task | Số bước SGD tối đa | TB |
|---|---|---|---|
| T2L (SFT) L | 479 / 256 / 128 / 64 | 1M / 640K / 320K / 160K | **67.7** / 67.4 / 65.8 / 66.0 |
| T2L (SFT) M | 479 / 256 / 128 / 64 | như trên | **67.5** / 66.9 / 66.0 / 65.2 |
| T2L (SFT) S | 479 / 256 / 128 / 64 | như trên | 65.2 / **65.9** / 65.8 / 64.7 |

Phụ lục C thử tăng số task mà **giữ cố định** compute. Kết luận: **phải tăng compute theo số task**. Ví dụ M với compute tăng theo số task tốt hơn hẳn so với compute cố định khi dùng ≥ 256 task.

### 5.2 Model embedding mô tả

**Bảng 4** (T2L SFT, 128 task): hai model embedding là **gte-large-en-v1.5** (lấy token CLS ở layer cuối, model hai chiều) và **Mistral-7B-Instruct** (lấy activation của token cuối). Trung bình **65.9** và **65.6**: gần như nhau, chứng tỏ T2L **không phụ thuộc nhiều** vào cách embedding mô tả.

### 5.3 Thay đổi mô tả task

Bốn loại mô tả:
- **Train**: mô tả dùng khi train cho đúng task đó.
- **Eval**: mô tả chưa thấy, nhưng đúng task.
- **Random strings**: chuỗi ký tự ngẫu nhiên.
- **Train (random)**: mô tả train của **task khác**, lấy ngẫu nhiên.

Hai nhóm: **khớp** (Train, Eval) và **không khớp** (Random strings, Train (random)). Thí nghiệm dùng T2L train bằng reconstruction (đã thấy mô tả của benchmark khi train).

**Bảng 5.**

| | Train | Eval | Train (random) | Random strings |
|---|---|---|---|---|
| T2L L | 73.3 | 73.6 | 49.1 | 68.2 |
| T2L M | 73.5 | 70.2 | 49.5 | 68.5 |
| T2L S | 73.0 | 72.9 | 55.7 | 53.9 |
| **TB** | **73.3** | **72.2** | **51.4** | **63.5** |

- Mô tả **khớp** cho LoRA tốt nhất, ngang LoRA oracle, kể cả mô tả Eval chưa thấy. Vậy T2L **chịu được thay đổi cách diễn đạt**, miễn là mô tả **đúng task**.
- Mô tả **không khớp** cho LoRA kém hơn rõ rệt.
- Tác giả cho rằng có thể dùng một LLM để **chỉnh mô tả cho khớp**, tránh trường hợp hỏng này.

**Hình 4 (ví dụ định tính, GSM8K, Mistral-7B-Instruct).** Câu hỏi: "Toulouse có số cừu gấp đôi Charleston. Charleston gấp 4 lần Seattle. Seattle có 20 con. Tổng ba thành phố có bao nhiêu?" (đáp án 260).
- (i) Base model: **240** (sai; tính Toulouse = 160, Charleston = 80 nhưng cộng sai).
- (ii) LoRA từ mô tả **kém** "solve this task please": vẫn **240**.
- (iii) LoRA từ mô tả "This task challenges your problem-solving abilities through mathematical reasoning…": **260** (đúng).
- (iv) LoRA từ mô tả "Use your programming skill to think algorithmically, logically and systematically…": **260** (đúng), nhưng **cách suy luận khác** (liệt kê quan hệ trước rồi tính).

Hai ví dụ cuối đều đúng nhưng có **phong cách trả lời khác nhau**, cho thấy T2L có thể **điều khiển (steer)** model bằng mô tả. Hyperdecoders không làm được vì nó dùng chính câu hỏi làm input cho hypernetwork.

### 5.4 Cách train

**Bảng 6.** Mọi model được train với thời gian xấp xỉ nhau (khoảng **10 giờ**).

| | Recon S | Recon M | Recon L | SFT S | SFT M | SFT L |
|---|---|---|---|---|---|---|
| Benchmark | 61.8 | 61.7 | 62.0 | 64.8 | 66.5 | 67.5 |

Trung bình theo kiến trúc: **Recon 61.8**, **SFT 66.3**.

SFT cho T2L **tốt hơn hẳn** reconstruction (66.3 so với 61.83 trung bình). Lý do: để T2L reconstruction khái quát được, các LoRA của task giống nhau phải **tụ lại** trong một đa tạp ẩn nào đó. SFT không cần LoRA đích nên tránh được vấn đề này nhờ học end-to-end. Phụ lục D cho thấy adapter của các task giống nhau **không nằm gần nhau** trong không gian trọng số.

### 5.5 Trực quan hoá activation của T2L

Kiểm tra xem T2L có sinh adapter riêng cho task chưa thấy không: dùng **SFT T2L M** train trên 256 task, đưa vào mọi benchmark, mỗi benchmark 3 mô tả chưa thấy. **Hình 5**: chiếu t-SNE 2D activation của **task encoder** (trái) và output **khối MLP cuối** (phải).
- Có **phân cụm rõ** theo task ở cả hai hình, nên T2L thực sự sinh adapter khác nhau cho task khác nhau, "ngay lập tức".
- Task giống nhau như **MBPP và HumanEval** nằm **cùng cụm**: SFT T2L sinh adapter giống nhau cho task giống nhau về ngữ nghĩa.

---

## 6. Related Work (Công trình liên quan)

- **Hypernetwork để thích nghi.** Hypernetwork là cách mã hoá gián tiếp trọng số, đã dùng cho nhiều kiến trúc (vd. attention) và nhiều bối cảnh (vd. continual learning). Paper này tập trung vào **sinh adapter low-rank từ hướng dẫn ngôn ngữ**. Công trình trước dùng hypernetwork cho thích nghi đa nhiệm nhưng chỉ với **định danh task học được** thay vì ngôn ngữ, nên không khái quát zero-shot sang task mới.
- **Hypernetwork cho thích nghi LLM zero-shot.**
  - Xiao et al. (2023): chỉ 5 phương ngữ tiếng Anh, và cần biến đổi phương ngữ dựa trên chuyên gia.
  - **Gisting** (Mu et al., 2024): nén mô tả task trong context thành **token prefix**. Nhưng prefix chỉ tác động tới ma trận attention, kém linh hoạt hơn LoRA (sửa được nhiều phần của LLM).
  - **Hyperdecoders**: sinh adapter **theo từng chuỗi input**. Tác giả lập luận: thích nghi theo **mô tả** cho người dùng nhiều quyền điều khiển hơn (Hình 4); và adapter theo input **không gộp được** vào model gốc, gây chi phí cho mỗi câu hỏi.
  - HyperTuning, HNET-LM, HINT: chủ yếu cho model encoder-decoder (T5, BART), thường khởi tạo một phần hypernetwork từ model gốc. T2L dùng model instruction-tuned hiện đại (Mistral, Llama, Gemma), **không phụ thuộc** model embedding, và dùng mô tả do LLM sinh thay vì mô tả của SNI (Phụ lục B cho thấy cách này tốt hơn đáng kể).
  - Lv et al. (2024), đồng thời: hypernetwork sinh LoRA lúc suy luận nhưng cần **ví dụ few-shot** trong context; T2L chỉ cần **mô tả task**, người dùng viết trong vài giây.

---

## 7. Discussion and Limitations (Thảo luận và hạn chế)

**Thảo luận.**
- Mô tả task được sinh bằng **GPT-4o mini** để đảm bảo chất lượng và nhất quán. Khi dùng thật, người dùng có thể viết mô tả **kém chất lượng**, làm giảm hiệu năng.
- Kết quả chủ yếu cho LLM, nhưng T2L có thể dùng trực tiếp cho LLM khác hoặc **model ngôn ngữ–thị giác**.
- Câu hỏi mở: T2L train trên model nhỏ có chuyển được sang model lớn hơn **cùng kiến trúc** không?

**Hạn chế.**
- Chỉ xét **LoRA** làm không gian output. Có thể có cách điều biến LLM từ mô tả hiệu quả hơn, ví dụ **điều chỉnh trực tiếp activation**.
- Mức nén của T2L còn cải thiện được bằng inductive bias tốt hơn.
- T2L có dấu hiệu mở rộng tốt, nhưng **chưa đạt** hiệu năng của LoRA riêng từng task khi zero-shot. Đạt được thích nghi zero-shot mạnh như vậy vẫn là **một trong những thách thức lớn nhất** của T2L.

---

## Phụ lục (chọn lọc)

### A. Tổng quát sang Llama và Gemma

**Bảng 7 — Llama-3.1-8B-Instruct**

| | ArcC | ArcE | BQ | GSM8K | HS | OQA | PIQA | WG | HE | MBPP | TB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 73.3 | 90.6 | 80.4 | 75.7 | 66.6 | 75.4 | 79.8 | 55.3 | 66.5 | 68.7 | 73.2 |
| 3-shot ICL | 80.7 | 91.9 | 80.0 | 75.7 | 59.3 | 77.6 | 80.9 | 61.3 | 66.5 | 70.4 | 74.4 |
| Thêm mô tả task | 80.2 | 92.5 | 79.9 | 75.7 | 69.8 | 78.4 | 81.7 | 62.4 | 68.3 | 70.2 | 75.9 |
| Multi-task LoRA | 82.0 | 92.8 | 83.3 | 77.6 | 70.8 | **81.8** | **83.8** | **60.3** | 63.4 | 69.4 | 76.5 |
| **T2L (SFT) L** | **82.4** | **92.9** | **84.4** | **79.1** | **72.8** | **81.8** | 81.2 | 60.0 | **64.6** | **69.9** | **76.9** |

**Bảng 8 — Gemma-2-2B-Instruct** (mọi phương pháp dùng thêm ICL vì ICL cải thiện mạnh MBPP với Gemma)

| | ArcC | ArcE | BQ | GSM8K | HS | OQA | PIQA | WG | HE | MBPP | TB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-2B-Instruct | 73.7 | **89.9** | 81.0 | 55.6 | 55.2 | 71.0 | 71.0 | 53.8 | **43.9** | 12.3 | 60.7 |
| 3-shot ICL | 72.4 | 88.9 | **82.5** | 55.6 | 55.7 | 72.6 | 67.6 | 53.7 | **43.9** | 43.1 | 63.6 |
| Thêm mô tả + ICL | 72.4 | 88.9 | **82.5** | 55.6 | 55.7 | 72.6 | 67.6 | 53.7 | **43.9** | 43.1 | 63.6 |
| Multi-task LoRA + ICL | 73.5 | 89.4 | 81.6 | **57.2** | 59.5 | **74.6** | 69.4 | 58.1 | 39.0 | 50.4 | 65.2 |
| **T2L (SFT) L + ICL** | **74.0** | 89.8 | 81.8 | 55.1 | **62.5** | 73.9 | **75.2** | **58.7** | 41.5 | **51.5** | **66.4** |

(Số in đậm giữ theo PDF.) T2L luôn vượt baseline với các model có kích thước và kiến trúc khác nhau, dù dùng **cùng bộ siêu tham số** cho mọi base model.

> 💡 **Quan trọng cho demo:** Gemma-2-2B chạy được trên GPU nhỏ, và checkpoint `gemma_2b_t2l` có sẵn.

### B. Nguồn mô tả train
Dùng **định nghĩa task gốc của SNI** (do cộng đồng viết) thay cho mô tả do ChatGPT sinh làm T2L kém hơn (67.7 → 66.5), có lẽ vì mô tả crowd-source không nhất quán về mẫu và độ chi tiết.

### C. Tăng số task với compute cố định
Kể cả khi giữ số bước gradient gần như cố định, tăng số task train vẫn cải thiện trung bình (Bảng 10). Điều này gợi ý T2L mở rộng được và có **chuyển giao tích cực** giữa các task.

### D. LoRA của các task giống nhau
- Đo cosine similarity giữa các adapter (nối A, B của mọi layer): **không có tương quan** giữa độ giống trọng số và độ giống mô tả (Pearson ≈ 0).
- Nhưng có **tương quan dương** giữa độ giống mô tả và **hiệu năng** trên benchmark: adapter của task có mô tả giống benchmark thì làm tốt benchmark đó hơn.
- Vậy adapter **cùng chức năng** lại **không nằm gần nhau** trong không gian tham số. Đây là lý do T2L reconstruction khó khái quát.

### E. Siêu tham số (Bảng 11)

| | LoRA riêng | T2L (SFT) | T2L (recon) |
|---|---|---|---|
| Batch size | 8 | 8 | = số LoRA đích |
| Gradient accumulation | 1 | 1 | 1 |
| Max learning rate | 8×10⁻⁵ | 2.5×10⁻⁵ | 10⁻³ |
| Max gradient norm | 1.0 | 1.0 | 1.0 |
| NEFTune noise alpha | 5.0 | 5.0 | 5.0 |
| Warmup | 0.1 | 0.1 | 0.1 |
| LR scheduler | Linear + warmup | Linear + warmup | Linear + warmup |

Cấu hình LoRA (Listing 1): `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `target_modules=["q_proj","v_proj"]`, `use_rslora=true`. Với reconstruction, mỗi mẫu train là **cả một LoRA** (479 mẫu cho 479 task), nên tăng lên **100.000 epoch** để hội tụ.

### F. Chi tiết kiến trúc
- Embedding module và layer: **32 chiều**, từ điển 34 embedding (32 layer + 2 module đích).
- Task encoder: một layer tuyến tính nhận embedding mô tả (1024 chiều với gte, 4096 với Mistral), ra vector **64 chiều**.
- Ghép embedding task, module, layer → `mlp0` → khối MLP residual `mlp1`. M và S thêm **A/B embedding 128 chiều**; S thêm **rank embedding 128 chiều** sau `mlp2`. Cuối cùng là khối MLP và head tuyến tính (output L: `2×r×d`; M: `r×d`; S: `d`).
- Thực tế q_proj có d_in = d_out = 4096; v_proj có d_in = 4096, d_out = 1024. r = 8.
- Số tham số: **L 55.252.992**, **M 34.282.240**, **S 4.923.392**; LoRA: **3.407.872**.

### G. Khởi tạo
Dùng **Bias-HyperInit**: khởi tạo trọng số head bằng 0 và bias khớp với khởi tạo của layer bên dưới (với L: bias của head A ~ U(−1/d, 1/d), head B = 0, giống LoRA thường; M, S chỉnh để khớp độ lớn gradient). **Không có khởi tạo này, train không ổn định và thường thất bại.**

### H. Chi tiết train
- Mọi model vừa **1 GPU H100 (80GB)**. **SFT tốn bộ nhớ hơn nhiều** vì phải lan gradient qua LLM gốc; **reconstruction có thể chạy trên GPU phổ thông**.
- Reconstruction: cố định 100K epoch, batch = số LoRA đích. Tái tạo **9 LoRA benchmark mất khoảng 10 phút**; tái tạo **479 LoRA SNI mất khoảng 10 giờ**.
- Với reconstruction, T2L dự đoán **z-score** của từng phần tử A, B; lúc test nhân với độ lệch chuẩn rồi cộng trung bình để về đúng thang đo.

### I. Phân tích FLOPs
Giả định: 3-shot ICL khoảng 256 token, câu hỏi khoảng 64 token, mô tả khoảng 48 token; base Mistral-7B (H = 4096, L = 32); encoder gte-large (H = 1024, L = 24); kiến trúc M.

| Thành phần | TFLOPs mỗi câu |
|---|---|
| gte-large-en-v1.5 (mô tả) | 0.029 |
| Hypernetwork M | 0.000005 |
| LLM không ICL (64 token) | 0.827 |
| **Tổng T2L** | **0.856** |
| **LLM với 3-shot ICL** (320 token) | **4.177** |

T2L rẻ hơn **hơn 4 lần** so với 3-shot ICL, tiết kiệm ngay từ câu hỏi đầu tiên.

### J. Dữ liệu
500 dataset SNI tại https://huggingface.co/Lots-of-LoRAs; 479 để train, phần còn lại để đánh giá. ICL lấy 3 ví dụ từ tập train của từng benchmark (MBPP có split riêng cho ICL; HumanEval chỉ có test nên luôn zero-shot).

### K. Dùng ΔW đầy đủ thay vì A, B
Khi so sánh trên **ΔW = BᵀA** đầy đủ (thay vì nối A, B), xuất hiện **tương quan dương** giữa độ giống mô tả và độ giống adapter. Lý do: với một ΔW có **rất nhiều cặp A, B** khác nhau cho ra nó. Vậy nếu tính loss tái tạo trên ΔW, T2L reconstruction có thể khái quát tốt hơn; nhưng thực nghiệm cho thấy **không** vượt T2L tái tạo A, B khi sinh LoRA zero-shot.

### L. Sinh mô tả task bằng LLM
Dùng **GPT-4o mini**: với mỗi task sinh **200 mô tả** (gọi 10 lần, mỗi lần 20 mô tả, từ 5 cặp hỏi–đáp ngẫu nhiên của task). Prompt có ví dụ cặp hỏi–đáp với mô tả mẫu, và yêu cầu tránh dài dòng hoặc thông tin thừa như định dạng đáp án hay hướng dẫn tường minh.

> 💡 **Áp dụng cho app:** khi người dùng viết mô tả quá ngắn, app có thể dùng một LLM viết lại mô tả theo phong cách này trước khi đưa vào T2L. Mục 5.3 gợi ý đúng điều đó.
