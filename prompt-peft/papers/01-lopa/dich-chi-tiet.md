# LoPA — Dịch và diễn giải chi tiết

> Bản dịch có diễn giải (không phải dịch từng chữ) của *Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation* (NeurIPS 2024).
> Ô **💡 Giải thích** là phần nhóm thêm vào cho dễ hiểu, không có trong paper.
> Thuật ngữ giữ nguyên tiếng Anh khi đó là tên gọi chuẩn (soft prompt, adapter, rank…).

---

## Abstract

Parameter-Efficient Fine-Tuning (PEFT) đã trở thành cách chuẩn để tuỳ biến Foundation Model (FM) cho các task riêng của người dùng. Tuy nhiên, các phương pháp PEFT thông thường phải **lưu nhiều adapter riêng cho từng task**, gây khó khăn khi mở rộng vì các adapter này phải được đặt và chạy **ngay trên server của FM**. Prompt tuning truyền thống là một lời giải tiềm năng, vì nó tuỳ biến model chỉ bằng một **tiền tố (prefix) riêng cho từng task** ở input, nhưng hiệu năng lại kém hơn các phương pháp PEFT khác như LoRA.

Để lấp khoảng cách đó, tác giả đề xuất **Low-Rank Prompt Adaptation (LoPA)**, một cách tiếp cận dựa trên prompt tuning. LoPA đạt hiệu năng ngang các phương pháp PEFT tốt nhất và ngang full fine-tuning, trong khi **ít tham số hơn** và **không cần adapter trên server**. LoPA sinh soft prompt bằng cách cân bằng giữa việc *chia sẻ thông tin của task* cho mọi mẫu và *tuỳ biến theo từng mẫu*. Nó dùng **phân rã low-rank** cho thành phần soft prompt riêng của từng mẫu để tiết kiệm tham số. Tác giả đánh giá trên nhiều task hiểu ngôn ngữ tự nhiên, sinh code và hiểu code, với nhiều foundation model ở các kích thước khác nhau.

---

## 1. Introduction (Giới thiệu)

Mô hình ngôn ngữ có khả năng few-shot rất tốt, kể cả trên task chưa thấy khi train. Dù vậy, **fine-tuning vẫn cần thiết** để đạt hiệu năng tối ưu trên một task cụ thể. Nhưng model càng lớn thì cập nhật toàn bộ tham số càng không thực tế. PEFT giải quyết điều này bằng cách chỉ chỉnh một tập nhỏ tham số và giữ nguyên phần còn lại, cho phép **cá nhân hoá một FM cho nhiều người dùng cùng lúc**.

Đã có rất nhiều phương pháp PEFT, khác nhau ở *vị trí* sửa trong FM, *hàm* dùng để sửa, và *cách* tích hợp phần sửa. Dù hiệu quả, chúng đều cần **duy trì nhiều module kiểu adapter cho từng task trên server**, và mỗi khi xử lý một batch request phải **chọn và ghép** đúng các module đó (Hình 1).

> **Hình 1.** Sơ đồ minh hoạ cách các phương pháp PEFT như LoRA cá nhân hoá một FM cho nhiều task (vd. phân loại Có/Không, hoàn thành code) khi suy luận: mỗi task có adapter riêng nằm trên server.

**Prompt tuning** là cách đơn giản với một số ưu điểm:
1. Tiết kiệm tham số: chỉ cần một tập nhỏ vector (*soft prompt*) gắn vào tầng input và được học cho từng task.
2. **Không cần xử lý riêng theo task trên server**: một prefix được thêm vào trước, và server xử lý mọi task giống hệt nhau.

Nhưng prompt tuning đã được chứng minh là **kém hơn** các phương pháp PEFT khác. Gần đây có các nỗ lực cải thiện bằng cách chèn soft prompt vào **nhiều tầng** của transformer, nhưng như vậy lại tăng số tham số và lại **cần sửa ở server**. Một hướng thú vị (IDPG) làm soft prompt **phụ thuộc vào input** bằng một bộ sinh prompt nhẹ cho từng mẫu, và đạt cải thiện đáng kể. Câu hỏi đặt ra: *có thể cải thiện prompt tuning hơn nữa mà vẫn tiết kiệm tham số không?*

Tác giả đề xuất **LoPA**, một phương pháp prompt tuning *nhận biết từng mẫu* (instance-aware). LoPA tạo soft prompt từ hai thành phần:
- một thành phần **riêng cho task**, chia sẻ thông tin chung giữa các mẫu;
- một thành phần **riêng cho từng mẫu**, mang thông tin của mẫu đó.

Phân tích cho thấy chỉ dùng *một* trong hai thành phần (như các công trình trước: prompt tuning chỉ dùng phần task, IDPG chỉ dùng phần mẫu) thì **không đủ** để vượt các baseline PEFT. LoPA đạt hiệu năng cao nhờ **cân bằng** hai phần. LoPA kết hợp hai phần bằng một **hàm cổng (gating)**, kích hoạt thông tin của task tuỳ theo từng mẫu, và dùng **phân rã low-rank** cho phần riêng của mẫu để tiết kiệm tham số (Hình 2). Sau khi train, người dùng chỉ cần đưa soft prompt làm prefix cho input gửi tới FM, **không tốn thêm chi phí tính toán ở server**.

Thí nghiệm gồm 6 task NLU của GLUE và 3 task hiểu/sinh code. LoPA vượt các phương pháp prompt tuning hiện có, thường ngang full fine-tuning và LoRA, và **vượt LoRA ở 11/24 trường hợp**.

**Đóng góp:**
- Đề xuất LoPA, một chiến lược prompt tuning tiết kiệm tham số và hiệu năng cao.
- Kiểm chứng với full fine-tuning và các phương pháp PEFT tốt nhất trên 9 task, 7 backbone transformer.

---

## 2. Related Work (Công trình liên quan)

**Dựa trên adapter.** Chèn các lớp/module adapter có thể train vào mạng transformer; rẻ hơn full fine-tuning mà vẫn giữ hiệu năng.
- **LoRA**: xấp xỉ low-rank cho phần cập nhật trọng số, rất phổ biến.
- **Compacter**: adapter tham số hoá bằng lớp nhân siêu phức (PHM) low-rank.
- **DoRA**: tách trọng số thành độ lớn và hướng, dùng LoRA cho phần hướng.

**Soft prompting.** Chèn soft prompt vào hidden state thay vì thêm module.
- **Prefix-tuning, P-tuning v2**: thêm vector prefix vào key/value ở **mọi tầng**.
- **Prompt tuning, P-tuning, DePT**: chỉ thêm embedding prompt ở **tầng đầu tiên**. DePT còn dùng LoRA để học phần cập nhật cho ma trận embedding.

Các phương pháp này *không phụ thuộc mẫu* (instance-agnostic), chỉ tối ưu một prompt cho cả task. Các phương pháp *phụ thuộc mẫu*:
- **IDPG**: sinh soft prompt cho từng mẫu, ở tầng embedding (bản S) hoặc mọi tầng (bản M).
- **LPT**: chèn prompt vào một tầng giữa, để khỏi tính gradient ở các tầng dưới và giữ thông tin task.
- **SPT**: học một cổng quyết định có truyền tiếp soft prompt của tầng trước hay học prompt mới.

Trừ prompt tuning và S-IDPG, các phương pháp PEFT đều chèn module hoặc prefix vào tầng sâu, hoặc tái tham số hoá low-rank các tầng có sẵn, nên **phải lưu tham số PEFT trên server**. Còn LoPA đưa soft prompt làm prefix của input, **không cần lưu gì theo task trên server**. Ngoài ra LoPA cân bằng giữa *riêng biệt* và *tổng quát* tốt hơn prompt tuning (chỉ có prompt chung cho task) và IDPG (chỉ nhấn vào prompt riêng cho mẫu).

---

## 3. Proposed Methodology (Phương pháp đề xuất)

### 3.1 Preliminaries (Kiến thức nền)

**Transformer.** Mỗi tầng có nhiều head (N_H head), mỗi head thực hiện self-attention. Xét một head với các ma trận query, key, value `W^Q, W^K, W^V ∈ R^{d_H × d}` (d là chiều của model). Với chuỗi input `X = {x_1, …, x_n}` và vector query `x_i`, output của head tại vị trí i là:

```
o_i = Attention(W^Q x_i, W^K X, W^V X) = softmax( W^Q x_i (W^K X)ᵀ ) W^V X        (1)
```
(bỏ hằng số √d_H cho gọn). Ở tầng đầu, X chính là ma trận embedding `X_e ∈ R^{d×n}`.

**Full Fine-Tuning (FFT).** Cập nhật toàn bộ trọng số W để tối đa log-likelihood của đáp án: `max_W E_{(X,y)~D} [log p_W(y|X)]`.

**Prompt Tuning (PT).** Học một soft prompt `Z ∈ R^{d×m}` gồm m vector `{z_1, …, z_m}`, nối vào trước embedding của input, và train bằng lan truyền ngược: `max_Z E[log p_W(y | concat(Z, X_e))]`. Có thể hiểu Z là embedding của các **token ảo** giúp model làm task mới mà không đổi tham số.

Với prompt tuning, output của một head ở tầng đầu trở thành:
```
o_i^PT = Σ_k A_ik W^V z_k   +   (1 − Σ_k A_ik) o_i                                   (2)
         └──── bias ────┘       └──── scale ────┘

A_ik = exp((W^K z_k)ᵀ W^Q x_i) / [ Σ_k exp((W^K z_k)ᵀ W^Q x_i) + Σ_j exp((W^K x_j)ᵀ W^Q x_i) ]   (3)
```
A_ik là mức attention mà query x_i dành cho vector prefix z_k. Từ (2): soft prompt **nội suy tuyến tính** output của head. Phần bias nằm trong một **không gian con dịch chuyển (offset subspace)** sinh bởi các vector `{W^V z_k}`, có số chiều ≤ m.

> 💡 **Giải thích.** Phương trình (2) nói rằng thêm soft prompt tương đương với việc "kéo" output của mỗi head về phía một hướng mới (phần *bias*), đồng thời "thu nhỏ" output gốc (phần *scale*). Hướng kéo là tổ hợp của các vector `W^V z_k`. Với prompt tuning thường, các vector này **cố định** cho mọi input; chỉ các hệ số A_ik thay đổi.

### 3.2 Low-rank Prompt Adaptation (LoPA)

LoPA tạo soft prompt dưới dạng **`Z = Z_S ∘ g(Z_I)`**, trong đó:
- `Z_S ∈ R^{d×m}`: thành phần **riêng cho task**;
- `Z_I ∈ R^{d×m}`: thành phần **riêng cho từng mẫu**;
- g là hàm cổng (sigmoid), ∘ là tích Hadamard (nhân từng phần tử).

Trực giác: Z_S được **chia sẻ** giữa các mẫu nên nắm thông tin chung, giúp model thích nghi với task. Z_I tinh chỉnh soft prompt cho từng mẫu, đóng vai trò như **tín hiệu kích hoạt**. Hai thành phần có tham số riêng. Z_S giống prompt tuning: gồm m vector học được. Z_I được tính từ input qua một hàm mã hoá `f: R^{d×n} → R^{d×m}`.

Mã hoá trực tiếp một ma trận d×m thì tốn: ví dụ IDPG dùng một MLP có hidden h cần O(hdm) tham số. Để tiết kiệm, LoPA giả định **Z_I có hạng thấp**: `Z_I = u × vᵀ` với `u ∈ R^{d×r}`, `v ∈ R^{m×r}`, mỗi ma trận do một MLP riêng sinh ra. Như vậy f chỉ cần `O(hdm(r/d + r/m))` tham số, tức giảm theo hệ số `(r/d + r/m) < 1` khi r ≪ min(d, m). Tổng hợp lại:

```
Z = Z_S ∘ g( MLP_u(X′) × MLP_v(X′)ᵀ )                                              (4)
             └─────────── Z_I = f(X′) ──────────┘
```
Ba thành phần train được: **Z_S, MLP_u, MLP_v**. Mỗi MLP gồm một lớp chiếu xuống, một hàm phi tuyến, và một lớp chiếu lên. MLP đặt trên một **mô hình ngôn ngữ nhỏ hơn (Encoder model)**, mô hình này cho biểu diễn input X′.

> **Hình 2.** Minh hoạ LoPA. Không cần lưu adapter riêng cho task trên server. Input của FM là `X = concat(Z | X_e)`.

> 💡 **Giải thích bằng số.** Với d = 1024, m = 10, r = 4: Z_I đầy đủ có 10 240 phần tử. Dạng low-rank chỉ cần u (1024 × 4) + v (10 × 4), tức 4 136 phần tử. Cái được tiết kiệm thật sự là **lớp chiếu lên của MLP**: nó chỉ cần sinh ra 4 136 số thay vì 10 240.

**Không gian offset do LoPA tạo ra.** Từ (2), vector bias là tổ hợp tuyến tính của các `W^V z_k` với hệ số A_ik. Trong LoPA, **input ảnh hưởng tới Z**, nên chính các vector `W^V z_k` cũng thay đổi theo input. Kết quả là **mỗi input có một offset subspace riêng**. Còn prompt tuning truyền thống giữ các vector cố định, chỉ để các hệ số thay đổi. Vì vậy các phương pháp nhạy với từng mẫu có thể tác động mạnh hơn tới attention ở các tầng sâu.

**Học kết hợp Z_S và Z_I.** Đạo hàm của hàm mục tiêu L theo hai thành phần (với g = sigmoid):
```
∇_{Z_S} L = ∇_Z L ∘ σ(Z_I)
∇_{Z_I} L = (∇_Z L ∘ Z_S) · σ(Z_I)(1 − σ(Z_I))
```
Nghĩa là thay đổi của Z_I (qua sigmoid) ảnh hưởng trực tiếp tới cập nhật của Z_S, và cập nhật của Z_I được nhân với cả Z_S lẫn đạo hàm của sigmoid. Ngược lại, IDPG có `Z = MLP(X′)`; bias của MLP có thể coi là phần riêng cho task, nên thực chất là `Z = Z_S + Z_I`, và cập nhật của hai phần **độc lập** nhau. Phép cộng tuyến tính như vậy có thể không nắm được quan hệ phức tạp giữa hai phần, điều mà phân rã phi tuyến của LoPA nắm được.

---

## 4. Experiments, Results, and Discussion (Thí nghiệm)

### 4.1 Thiết lập

**Task.**
- 6 task **NLU** của GLUE: SST-2 (cảm xúc), MNLI (suy luận), MRPC (diễn giải), QNLI (hỏi đáp), QQP (câu hỏi trùng), RTE (suy luận).
- **Sinh code**: MBPP, hoàn thành thân hàm.
- **Hiểu code**: CruxEval-I (đoán *input* khi biết output của hàm) và CruxEval-O (đoán *output* khi biết input).

**Baseline.**
- Tuỳ biến *phía server*: (1) FFT, (2) LoRA, (3) P-tuning v2, (4) prefix-tuning (h = 512).
- Tuỳ biến *phía người dùng* (soft prompt): (5) prompt tuning, (6) S-IDPG, (7) DePT.
- (8) model không tuning (zero-shot).

IDPG dùng lớp DNN (không phải PHM) vì tốt hơn trên bộ task này (Phụ lục 7.1); LoPA cũng dùng DNN để so sánh công bằng.

**Backbone.**
- NLU: RoBERTa-355M.
- Code: CodeGen-mono 350M, DeepSeek-Coder 1.3B, phi-2 2.7B, phi-3 3.8B, DeepSeek-Coder 7B, Llama-3 8B.

**Chi tiết cài đặt.**
- GLUE dùng split chuẩn; MBPP và CruxEval chia 50-50.
- Mọi soft prompt dùng **m = 10** token ảo.
- Encoder: NLU dùng RoBERTa + MLP (h = 256); code dùng **CodeSage-125M + MLP (h = 1024)**.
- Rank r của LoPA chọn tốt nhất trong {1, 2, 4}.

**Cấu hình train.**
- NLU: FFT, LoRA 10 epoch; các phương pháp prompt 20 epoch.
- MBPP: 10 epoch cho mọi phương pháp.
- CruxEval: backbone < 7B train 20 epoch, ≥ 7B train 10 epoch; FFT 5 epoch.
- Learning rate: LoPA 1e-5 (NLU), 1e-3 (code); FFT 1e-5; LoRA và soft prompt khác 1e-4.
- Phần cứng: 2 × A100 40GB.

**Đánh giá.** Accuracy, F1 cho GLUE; pass@1 với temperature tốt nhất: 0.6 (MBPP), 0.2 (CruxEval).

### Bảng 1 — Kết quả GLUE (RoBERTa-355M)

| Phương pháp | Tham số train | SST-2 | MNLI | MRPC | QNLI | QQP | RTE | TB |
|---|---|---|---|---|---|---|---|---|
| FFT | 355M | 95.99 | 90.40 | 90.81 | 94.60 | 90.39 | 85.92 | 91.35 |
| P-tuning v2 | 0.49M | 89.91 | 88.13 | 70.18 | 85.21 | 84.63 | 53.43 | 78.58 |
| Prefix-tuning | 25.75M | 93.80 | 89.51 | 90.86 | 94.98 | 86.87 | 82.67 | 89.78 |
| LoRA | 2.36M | 96.22 | 90.30 | 90.77 | 94.69 | 89.91 | 85.66 | 91.26 |
| *— không sửa server —* | | | | | | | | |
| Không tuning | 0 | 59.97 | 39.60 | 73.52 | 50.16 | 42.34 | 53.43 | 53.17 |
| PT | 10.2K | 84.40 | 54.67 | 72.38 | 58.74 | 48.20 | 53.07 | 61.91 |
| DePT | 10.2K | 89.68 | 71.51 | 72.97 | 57.09 | 45.81 | 53.79 | 65.14 |
| S-IDPG | 2.89M | 95.30 | 84.50 | 78.60 | 90.48 | 84.88 | 77.26 | 85.17 |
| **LoPA** | 1.60M | 95.99 | 89.22 | **91.09** | 93.74 | 89.72 | 83.39 | 90.53 |

MRPC và QQP là trung bình của accuracy và F1; các task còn lại là accuracy.

### Bảng 2 — Kết quả code (pass@1)

| Backbone | Phương pháp | Tham số | CruxEval-I | CruxEval-O | MBPP |
|---|---|---|---|---|---|
| CodeGen-350M | FFT | 350M | 33.0 | 19.5 | 17.49 |
| | LoRA | 1.3M | 31.0 | 18.2 | 21.56 |
| | Không tuning | 0 | 20.8 | 15.0 | 15.85 |
| | PT | 10.2K | 32.8 | 15.8 | 15.20 |
| | S-IDPG | 8.5M | 16.9 | 13.2 | 17.04 |
| | **LoPA** | 4.4M | 34.5 | 18.5 | 17.04 |
| DeepSeek-Coder-1.3B | FFT | 1.3B | 45.0 | 34.8 | 44.76 |
| | LoRA | 4.7M | 35.5 | 36.0 | 44.14 |
| | Không tuning | 0 | 26.8 | 29.8 | 34.08 |
| | PT | 20.5K | 41.2 | 31.2 | 34.49 |
| | S-IDPG | 16.3M | 26.0 | 28.5 | 42.50 |
| | **LoPA** | 4.2M | 43.0 | 34.5 | 44.66 |
| Phi-2 2.7B | FFT | 2.7B | 40.2 | 37.0 | 55.03 |
| | LoRA | 7.9M | 41.5 | 42.5 | 51.54 |
| | Không tuning | 0 | 33.5 | 33.0 | 45.17 |
| | PT | 25.6K | 35.0 | 34.0 | 49.69 |
| | S-IDPG | 20.3M | 35.0 | 33.0 | 53.29 |
| | **LoPA** | 4.76M | 43.0 | 37.2 | 52.15 |
| Phi-3 3.8B | FFT | 3.8B | 39.2 | 39.5 | 54.00 |
| | LoRA | 6.3M | 38.0 | 41.2 | 42.92 |
| | Không tuning | 0 | 33.8 | 39.5 | 8.82 |
| | PT | 30.7K | 33.5 | 31.5 | 34.08 |
| | S-IDPG | 24.2M | 31.0 | 39.5 | 42.29 |
| | **LoPA** | 5.3M | 42.2 | 42.5 | 44.35 |
| DeepSeek-Coder-7B | FFT | 7B | OOM | OOM | OOM |
| | LoRA | 11.8M | 47.5 | 49.8 | 53.38 |
| | Không tuning | 0 | 39.3 | 44.0 | 50.51 |
| | PT | 41.0K | 45.8 | 44.8 | 37.47 |
| | S-IDPG | 32.1M | 40.5 | 41.5 | 53.59 |
| | **LoPA** | 6.35M | 50.0 | 48.0 | 52.46 |
| Llama-3 8B | FFT | 8B | OOM | OOM | OOM |
| | LoRA | 9.4M | 45.5 | 40.5 | 44.55 |
| | Không tuning | 0 | 27.0 | 31.5 | 45.37 |
| | PT | 41.0K | 37.5 | 32.0 | 26.07 |
| | S-IDPG | 32.1M | 29.2 | 35.2 | 33.05 |
| | **LoPA** | 6.4M | 41.2 | 39.8 | 43.94 |

OOM = hết bộ nhớ GPU.

### 4.2 So sánh với baseline

**NLU.** LoPA luôn hơn prompt tuning và DePT, trung bình **28.62** và **25.39** điểm. Điều này cho thấy làm soft prompt phụ thuộc vào mẫu giúp nó tác động mạnh hơn tới output. LoPA hơn IDPG trung bình **5.36** điểm, cho thấy cách phân rã đề xuất nắm được quan hệ phức tạp giữa Z_S và Z_I. Khác biệt rõ nhất khi **ít dữ liệu**: +12.5 điểm ở MRPC, +6.13 điểm ở RTE. LoPA cách FFT và LoRA **dưới 1 điểm**, trong khi dùng ít hơn LoRA **760K tham số** nhờ phân rã low-rank.

**Hiểu code.** LoPA luôn cải thiện so với không tuning ở mọi backbone. So với prompt tuning: hơn khoảng 2–4 điểm ở model nhỏ (CodeGen, DeepSeek-Coder-1.3B), và **8–11 điểm** ở model lớn (Llama-3, Phi-3) trên CruxEval-O. Đáng chú ý, IDPG **kém hơn PT** ở mọi model (trừ Phi-3 trên CruxEval-O): chỉ làm prompt phụ thuộc mẫu thôi thì chưa chắc tốt hơn, thậm chí có thể tệ hơn, có lẽ do **quá nhiều tham số dẫn tới overfit**. LoPA ít tham số hơn và kết hợp rõ ràng thông tin task lẫn mẫu nên tốt hơn.

LoPA ngang LoRA, thường cách 1–4 điểm pass@1, với khoảng **2/3 số tham số**. LoPA **vượt LoRA trên CruxEval-I ở mọi model trừ Llama-3** (hơn khoảng 2 điểm với DeepSeek-Coder-7B tới 4 điểm với Phi-3). Tác giả giải thích: các FM này vốn đã học rộng và khái quát zero-shot tốt; cập nhật trực tiếp một phần trọng số vẫn có thể gây **quên kiến thức cũ (catastrophic forgetting)**. Soft prompt thì "gợi ra" kỹ năng cần thiết mà không làm hỏng kiến thức sẵn có.

**Hoàn thành code (MBPP).** IDPG và LoPA cải thiện gần như ngang nhau (trừ Llama-3), nhưng LoPA dùng ít tham số hơn nhiều: khoảng **1/2** so với IDPG ở CodeGen và chỉ **1/5** ở DeepSeek-Coder-7B. Vậy LoPA mở rộng tốt theo kích thước FM. Với Llama-3, mọi phương pháp tuning đều làm giảm hiệu năng, có thể do overfit; có lẽ Llama-3 đã được train trên MBPP.

**Tổng quan.** Tính trung bình trên mọi task và FM: LoPA hơn PT **28.52%** và hơn IDPG **20.16%** (tương đối), chỉ thua LoRA **0.54%**. LoPA **thắng LoRA 11/24 trường hợp**, nên trong các thử nghiệm này không có lợi thế hệ thống nào nghiêng về LoRA. Vì LoPA không cần xử lý riêng theo task ở server (prompt có thể tính ở bất kỳ đâu, kể cả client, rồi mới gửi lên server), tác giả cho rằng LoPA có thể là một lựa chọn thay thế LoRA hữu ích cho một số task.

### 4.3 Ablation (Phân tích thành phần)

**Theo độ dài soft prompt m (Hình 3).** Tăng m tức là thêm vector vào tập biểu diễn soft prompt, mở rộng offset subspace. Các vector thêm vào có độc lập và hữu ích hay không tuỳ cách học và tuỳ FM.
- PT và IDPG ban đầu tốt lên khi m tăng, nhưng sau đó **chững lại hoặc giảm** do overfit (PT trên SST-2, IDPG trên CruxEval-O).
- LoPA **không dao động đáng kể** theo m. Có lẽ do **Z_S dùng chung đóng vai trò regularizer**, ngăn prompt riêng cho mẫu bị overfit.
- LoPA với **m = 5 đã hơn PT và IDPG dùng m > 5**. Điều này gợi ý offset subspace thực ra có số chiều nhỏ, và LoPA biểu diễn nó chính xác hơn.

> **Hình 3.** Hiệu năng theo m trên RTE, MRPC, SST-2 và CruxEval-O (DeepSeek-Coder-1.3B). Trục hoành là số tham số train tương đối.

**Theo rank r (Hình 4, m = 10).**
- NLU (MRPC, RTE, SST-2 trên RoBERTa), r từ 1 lên 4: hiệu năng **tăng đều**, +1–2% ở SST-2 và tới **+8% ở MRPC**.
- CruxEval: tăng r **không** giúp tương ứng. Lý do là kích thước dữ liệu: NLU có hàng nghìn mẫu nên xấp xỉ được ma trận hạng cao hơn; CruxEval chỉ có vài trăm mẫu, tăng r thêm tham số nên dễ overfit.

**Theo encoder (Hình 5, DeepSeek-Coder-1.3B).**

| Encoder | CruxEval-I | CruxEval-O |
|---|---|---|
| CodeBERT (125M) | 41.2 | 32.8 |
| CodeSage (130M) | 43.0 | 34.5 |
| CodeSage (365M) | 42.2 | 34.5 |

CodeSage hơn CodeBERT khoảng 2 điểm, nhờ được pre-train bằng **contrastive learning** nên phân biệt biểu diễn code tốt hơn, và prompt sinh ra nắm được sắc thái của từng mẫu tốt hơn. Tác giả cũng thử **fine-tune encoder** cùng lúc nhưng không thấy cải thiện đáng kể. Encoder lớn hơn (365M) cũng không tốt hơn.

**Theo hàm ghép Z (Bảng 3, RoBERTa, GLUE).**

| Z = | SST-2 | MNLI | MRPC | QNLI | QQP | RTE | TB |
|---|---|---|---|---|---|---|---|
| concat(Z_S, Z_I) | 94.50 | 78.45 | 76.00 | 91.43 | 76.11 | 84.12 | 83.44 |
| max(Z_S, Z_I) | 95.99 | 89.37 | 88.62 | 93.74 | 78.44 | 85.92 | 88.68 |
| **Z_S ∘ g(Z_I)** | 95.99 | 89.22 | 91.09 | 93.74 | 89.72 | 83.39 | **90.53** |

Nối đơn giản tệ nhất; các hàm phi tuyến (max và gating đề xuất) tốt nhất. Các hàm phi tuyến khác để dành cho công trình sau.

---

## 5. Conclusion (Kết luận)

Tác giả giới thiệu LoPA, phương pháp soft prompt **phụ thuộc mẫu**. LoPA vượt các phương pháp cùng họ prompt tuning và ngang full fine-tuning và LoRA, trong khi dùng ít tham số train hơn. LoPA dùng xấp xỉ low-rank cho soft prompt riêng của mẫu, rồi kết hợp với soft prompt riêng của task qua một hàm cổng. Với cách thiết kế soft prompt có chủ đích hơn, công trình muốn đưa prompt tuning trở thành **lựa chọn mạnh thay thế các phương pháp adapter** khi tuỳ biến foundation model cho từng người dùng.

**Hạn chế và hướng phát triển.**
- Hiệu quả thay thế LoRA mới được chứng minh trên benchmark; với các task thực tế "lạ" thì có thể vẫn cần LoRA hoặc full fine-tuning. Đánh giá trên task thực tế là hướng tiếp theo.
- Mới giả định soft prompt đặt ở **đầu** input; có thể thử đặt ở cuối hoặc ngẫu nhiên.
- FM kết hợp với LoPA có thể xem như một **Conditional Auto-Encoder**, trong đó các vector soft prompt nằm trong một *không gian ẩn (latent)* thay vì offset subspace. Góc nhìn này đặt ra câu hỏi: phải chăng cải thiện đến từ việc *suy ra và nén* kiến thức của task rồi cung cấp như thông tin bổ sung? Hướng này có thể dẫn tới các hệ auto-encoding tinh vi hơn.

**Tác động rộng hơn.** Phương pháp giảm chi phí train và vận hành khi host và tuỳ biến FM. Nó cũng **tăng quyền riêng tư** vì việc tuỳ biến theo task diễn ra ở phía người dùng, không phải ở server.

---

## 7. Phụ lục

### 7.1 So sánh với lớp PHM (Parameterized Hypercomplex Multiplication)

Tác giả xét lớp PHM như một cách khác (thay cho phân rã low-rank) để giảm số tham số. Họ cài IDPG với `Z_I = PHM_W(X′)`, trong đó `W = Σ_{i=1..n} A_i ⊗ B_i` (⊗ là tích Kronecker, n cân bằng giữa số tham số và mức phân rã).

| Phương pháp | Tham số | SST-2 | MNLI | MRPC | QNLI | QQP | RTE | TB |
|---|---|---|---|---|---|---|---|---|
| **LoPA** | 1.60M | 95.99 | 89.22 | 91.09 | 93.74 | 89.72 | 83.39 | **90.53** |
| S-IDPG + FC | 2.89M | 95.30 | 84.50 | 78.60 | 90.48 | 84.88 | 77.26 | 85.17 |
| S-IDPG + PHM (n=8) | 0.37M | 95.07 | 83.46 | 76.12 | 85.45 | 77.35 | 67.14 | 80.77 |
| S-IDPG + PHM (n=16) | 0.20M | 94.61 | 83.66 | 76.68 | 81.16 | 80.39 | 65.34 | 80.31 |
| S-IDPG + PHM (n=32) | 0.17M | 94.72 | 81.45 | 74.99 | 84.59 | 81.90 | 68.23 | 80.98 |

PHM giảm tham số nhưng làm hiệu năng **giảm khoảng 10 điểm** so với LoPA. Có thể do phân rã Kronecker áp đặt cấu trúc lên W, hạn chế khả năng biểu diễn so với lớp fully-connected. Tác giả lưu ý LoPA cũng có thể kết hợp với PHM nếu muốn giảm tham số hơn nữa.

### 7.2 Phân tích hội tụ

Hình 6–8 so sánh training loss và hiệu năng validation của PT, IDPG, LoPA trên QQP, QNLI, MNLI. Các phương pháp phụ thuộc mẫu (IDPG, LoPA) **hội tụ nhanh hơn** prompt tuning truyền thống, và **LoPA hội tụ nhanh hơn và đạt accuracy/F1 cao hơn IDPG**. Đây là hình hữu ích để chiếu khi live train trên sân khấu (so với đường loss của nhóm).
