# Prompting + PEFT — Kiến thức cơ bản (dễ hiểu)

> Tài liệu nhập môn cho chủ đề **Prompting + PEFT nâng cao**.
> Nguồn chính: CS224N Lecture 9 "Efficient Adaptation" (`paper-basic/slide.pdf`, https://web.stanford.edu/class/cs224n/)
> và các paper trong `paper-basic/`.

---

## 0. Bài toán gốc: làm sao "dạy" một mô hình lớn làm việc của mình?

Ta có một mô hình ngôn ngữ đã được huấn luyện sẵn (pre-trained LLM: GPT, LLaMA, Qwen…).
Nó biết rất nhiều thứ chung chung, nhưng ta muốn nó làm **một việc cụ thể**:
phân loại cảm xúc review, trả lời câu hỏi về luật, sinh code theo style công ty…

Có 3 cách lớn, xếp từ "không đụng vào model" tới "sửa toàn bộ model":

| Cách | Sửa gì trong model? | Chi phí | Ví dụ |
|---|---|---|---|
| **Prompting** | Không sửa gì, chỉ viết câu lệnh đầu vào | Rẻ nhất | Zero-shot, Few-shot, Chain-of-Thought |
| **PEFT** (Parameter-Efficient Fine-Tuning) | Chỉ train **một phần rất nhỏ** tham số (thường < 1%) | Vừa phải | LoRA, Adapter, Prompt Tuning, Prefix Tuning |
| **Full Fine-Tuning** | Train **toàn bộ** tham số | Rất đắt | Fine-tune cả 7B tham số |

Hình dung đơn giản: model là một **đầu bếp giỏi**.
- *Prompting* = đưa cho đầu bếp một tờ giấy ghi yêu cầu thật rõ ràng.
- *PEFT* = cho đầu bếp học thêm một khoá ngắn hạn, chỉ ghi chép vào một cuốn sổ tay nhỏ (không thay đổi kiến thức cũ).
- *Full fine-tuning* = cho đầu bếp đi học lại từ đầu.

---

## 1. Prompting — điều khiển model bằng đầu vào

### 1.1 Zero-shot prompting
Chỉ mô tả nhiệm vụ, không cho ví dụ.
```
Phân loại cảm xúc của câu sau là Tích cực hay Tiêu cực:
"Quán này phục vụ chậm quá."
Đáp án:
```

### 1.2 Few-shot prompting / In-context learning (GPT-3, `paper-basic/few-shot.pdf`)
Cho vài ví dụ mẫu ngay trong prompt, model "bắt chước" theo — **không cập nhật trọng số**.
```
"Đồ ăn rất ngon"         -> Tích cực
"Nhân viên thô lỗ"       -> Tiêu cực
"Giá hợp lý, sẽ quay lại" -> Tích cực
"Quán này phục vụ chậm quá" ->
```
Ý chính của paper GPT-3: khi model đủ lớn (175B), khả năng học từ vài ví dụ trong prompt **tự xuất hiện** (emergent).

### 1.3 Chain-of-Thought (CoT) (`paper-basic/CoT.pdf`)
Cho ví dụ có **các bước suy luận trung gian**, model sẽ "nghĩ từng bước" và trả lời đúng hơn ở các bài toán logic/toán.
```
Q: Roger có 5 quả bóng. Mua thêm 2 hộp, mỗi hộp 3 quả. Có bao nhiêu quả?
A: Ban đầu có 5 quả. 2 hộp x 3 = 6 quả. 5 + 6 = 11. Đáp án là 11.
```
Biến thể zero-shot CoT: chỉ cần thêm câu *"Let's think step by step"*.

### 1.4 Hạn chế của prompting
- Kết quả **nhạy cảm** với cách viết prompt (đổi một chữ có thể đổi kết quả).
- Bị giới hạn bởi độ dài context, prompt dài thì tốn tiền/token.
- Model nhỏ thì prompting kém hiệu quả.
- Không "học" được từ hàng nghìn ví dụ của bạn.

→ Khi có dữ liệu riêng và muốn chất lượng cao hơn, ta chuyển sang **PEFT**.

---

## 2. PEFT — fine-tune nhưng tiết kiệm

### 2.1 Tại sao cần PEFT?
- Full fine-tuning GPT-3 175B: mỗi task phải lưu **một bản sao 175B tham số** → không khả thi.
- PEFT: **đóng băng (freeze)** model gốc, chỉ train một ít tham số mới. Mỗi task chỉ cần lưu vài MB.
- Một model gốc có thể phục vụ **nhiều task / nhiều người dùng** bằng cách gắn các "module nhỏ" khác nhau.

Slide CS224N chia PEFT thành các hướng chính:

### 2.2 Adapter (Houlsby et al., ICML 2019 — `paper-basic/PEFT.pdf`)
Chèn thêm một **mạng nhỏ** (down-projection → phi tuyến → up-projection) vào mỗi tầng Transformer.
Chỉ train adapter. Kết quả: đạt gần full fine-tuning trên GLUE mà chỉ thêm ~3.6% tham số mỗi task.
Nhược điểm: thêm tầng → **tăng độ trễ** khi chạy.

### 2.3 LoRA — Low-Rank Adaptation (Hu et al., ICLR 2022 — `paper-basic/loRa.pdf`)
Ý tưởng: phần thay đổi của trọng số khi fine-tune có **hạng thấp (low-rank)**. Thay vì học cả ma trận ΔW kích thước d×d, ta học hai ma trận nhỏ:

```
W_mới = W_gốc + ΔW,   ΔW = B · A
W_gốc: d × d (đóng băng)
B: d × r,  A: r × d,  với r rất nhỏ (vd. r = 8)
```
Ví dụ d = 4096, r = 8: ΔW đầy đủ có 16.7 triệu tham số, B·A chỉ có 65 nghìn (ít hơn **256 lần**).
- Sau khi train có thể **gộp** B·A vào W → không thêm độ trễ.
- Hiện là phương pháp PEFT phổ biến nhất (thư viện HuggingFace `peft`).

### 2.4 Prompt Tuning / Soft Prompt (Lester et al., 2021) — **cầu nối giữa Prompting và PEFT**
Thay vì tự viết prompt bằng chữ (*hard prompt*), ta để model **tự học** prompt dưới dạng các vector:

```
Đầu vào bình thường:  [emb("Quán"), emb("này"), emb("ngon")]
Prompt tuning:        [z1, z2, ..., zm,  emb("Quán"), emb("này"), emb("ngon")]
                       ^^^^^^^^^^^^^^^ m vector học được ("virtual tokens")
```
- Chỉ train các vector `z` (vd. 10 token × 1024 chiều = 10K tham số!), model đóng băng hoàn toàn.
- Ưu điểm lớn: soft prompt chỉ là **phần đầu của input** → server không cần sửa gì, một model phục vụ được nhiều task.
- Nhược điểm: thường **kém hơn LoRA**, nhất là với model nhỏ.

Biến thể: **Prefix Tuning** (Li & Liang, 2021) và **P-Tuning v2** chèn vector học được vào **mọi tầng** (vào key/value của attention) → mạnh hơn nhưng phải sửa ở server.

### 2.5 Pruning / Subnetwork (Lottery Ticket — `paper-basic/finding-sparse.pdf`, ICLR 2019)
Trong một mạng lớn tồn tại những **mạng con thưa** ("vé số trúng") mà nếu train riêng vẫn đạt độ chính xác tương đương.
Liên hệ với PEFT: không cần cập nhật mọi tham số — chỉ cần một phần nhỏ.

### 2.6 So sánh nhanh

| Phương pháp | Train cái gì | Số tham số train | Thêm độ trễ? | Server phải lưu module riêng mỗi task? |
|---|---|---|---|---|
| Full FT | Toàn bộ | 100% | Không | Lưu nguyên model mỗi task 😱 |
| Adapter | Mạng nhỏ trong mỗi tầng | ~1–4% | Có | Có |
| LoRA | Ma trận B, A low-rank | ~0.1–1% | Không (nếu merge) | Có (hoặc merge riêng mỗi task) |
| Prefix / P-Tuning v2 | Vector ở mọi tầng | ~0.1–1% | Ít | Có |
| **Prompt Tuning** | Vector ở tầng input | ~0.001–0.01% | Ít | **Không** — chỉ gửi kèm input |

---

## 3. "Nâng cao" là gì? — hướng nghiên cứu hiện nay

Các paper trong `paper-advance/` đều cố gắng giải bài toán:
> **Làm sao giữ ưu điểm của prompt tuning (nhẹ, không phải sửa server, nhiều task dùng chung một model) mà vẫn có chất lượng ngang LoRA / full fine-tuning?**

| Paper | Hội nghị | Ý chính một câu |
|---|---|---|
| **LoPA** – Prompt Tuning Strikes Back | NeurIPS 2024 | Soft prompt = phần chung cho task ⊙ phần riêng cho từng câu (low-rank) → ngang LoRA mà không cần adapter ở server |
| **PARA** – Prompt Aware Representation Adjustment | ACL 2024 (theo tên file; cần kiểm tra lại) | Mỗi tầng có một bộ sinh vector nhỏ, dựa vào prompt để điều chỉnh hidden state → hiệu quả cho multi-tenant |
| **POST** – Privacy-Preserving Soft Prompt Transfer | ICML 2025 | Train soft prompt trên model nhỏ (ở máy người dùng, có thể kèm differential privacy) rồi chuyển sang LLM lớn → không lộ dữ liệu riêng |

Chi tiết về paper được chọn và thiết kế app: xem `02-chon-paper-va-thiet-ke-app.md`.

---

## 4. Thuật ngữ (glossary)

- **Pre-trained model / Foundation model (FM)**: model đã học trên dữ liệu khổng lồ.
- **Freeze**: giữ nguyên trọng số, không cập nhật khi train.
- **Embedding**: vector biểu diễn một token.
- **Hard prompt**: prompt bằng chữ do người viết. **Soft prompt**: prompt là vector do máy học.
- **Low-rank**: ma trận có thể viết thành tích của hai ma trận "mỏng" → ít tham số.
- **Multi-tenant**: một server/model phục vụ nhiều khách hàng/task khác nhau cùng lúc.
- **GLUE**: bộ benchmark hiểu ngôn ngữ (SST-2 = cảm xúc, MNLI = suy luận, …).

## 5. Lộ trình tự học gợi ý (khoảng 1 tuần)
1. Đọc slide CS224N Lecture 9 (`paper-basic/slide.pdf`), phần Prompting → LoRA → Prompt-tuning.
2. Đọc abstract và hình 1 của các paper: GPT-3, CoT, LoRA.
3. Chạy thử notebook HuggingFace `peft`: LoRA và Prompt Tuning trên SST-2 với model nhỏ.
4. Đọc kỹ paper LoPA (mục 3 và hình 2) — đây là paper chính.
