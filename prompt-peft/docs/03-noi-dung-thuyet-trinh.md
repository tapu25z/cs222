# Nội dung thuyết trình — Một bài toán, hai lời giải: Text-to-LoRA và GEPA

> Bản thiết kế nội dung, theo từng slide. Mỗi slide có: nội dung chính, hình/bảng lấy từ paper, và ý người nói cần truyền đạt.
> Ý tưởng xuyên suốt: **cả hai paper giải cùng một bài toán**. Mỗi phương pháp được trình bày theo đúng 3 phần **Ý tưởng → Method → Evaluation**, sau đó so sánh và kết thúc bằng **câu hỏi mở: kết hợp hai cách như thế nào?**
> Số liệu lấy từ bản dịch đã đối chiếu: [`papers/01-text-to-lora/dich-chi-tiet.md`](../papers/01-text-to-lora/dich-chi-tiet.md), [`papers/02-gepa/dich-chi-tiet.md`](../papers/02-gepa/dich-chi-tiet.md).
> Chưa gồm phần demo app; các chỗ có thể chèn demo được đánh dấu **[demo tuỳ chọn]**.

---

## Bố cục tổng

| Phần | Nội dung | Người nói | Thời lượng gợi ý |
|---|---|---|---|
| 0 | Mở đầu | Người 3 | 5' |
| 1 | **Bài toán chung** | Người 3 | 15' |
| 2 | **Lời giải 1 — Text-to-LoRA**: Ý tưởng · Method · Evaluation | Người 1 | 45' |
| — | *Nghỉ* | | 10' |
| 3 | **Lời giải 2 — GEPA**: (DSPy nhanh) · Ý tưởng · Method · Evaluation | Người 2 | 50' |
| 4 | **Đặt cạnh nhau**: cùng bài toán, hai lời giải | Người 3 | 15' |
| 5 | **Câu hỏi mở: kết hợp hai cách?** | Người 3 dẫn, cả 3 cùng thảo luận | 25' |
| 6 | Kết luận + Q&A | Cả 3 | 15' |
| | **Tổng** | | **180'** |

Khoảng 64 slide (sau khi thêm các hộp kiến thức nền ở Phần 2). Phần 2 và 3 dùng **cùng khung slide** (cùng thứ tự, cùng kiểu hình) để khán giả dễ so sánh.

---

## Phần 0 — Mở đầu (Người 3, 5')

**Slide 0.1 — Tiêu đề.** "Một bài toán, hai lời giải: đưa kiến thức vào **trọng số** hay vào **prompt**?" · Text-to-LoRA (ICML 2025) · GEPA (ICLR 2026, Oral).

**Slide 0.2 — Một câu hỏi để giữ trong đầu.** Chiếu câu hỏi GSM8K dùng xuyên suốt buổi (lấy từ Hình 4 của paper T2L):
> *Toulouse có số cừu gấp đôi Charleston. Charleston có số cừu gấp 4 lần Seattle. Seattle có 20 con. Hỏi ba thành phố có tổng cộng bao nhiêu con cừu?* (Đáp án: 260)

Mistral-7B-Instruct trả lời **240** (sai). Câu hỏi cho cả buổi: *làm sao để model trả lời đúng mà không train lại model?*

---

## Phần 1 — Bài toán chung (Người 3, 15')

**Slide 1.1 — Bối cảnh.** Có một LLM tốt nhưng chung chung. Muốn nó làm tốt **một task mới**. Ba ràng buộc thực tế:
- **Không train lại cả model** (quá đắt, hoặc model đóng).
- **Ít dữ liệu** cho task mới (vài chục tới vài trăm ví dụ, hoặc không có).
- **Ít compute** cho mỗi task mới.

**Slide 1.2 — Hai cách quen thuộc và điểm yếu** (nhắc nhanh, team lý thuyết đã trình bày):
- **LoRA cho từng task**: tốt, nhưng mỗi task phải gom dữ liệu, chọn siêu tham số, train.
- **Prompt viết tay / few-shot**: không train, nhưng giòn, tốn token, phụ thuộc người viết.
- **RL (vd. GRPO + LoRA)**: mạnh nhưng cần **hàng chục nghìn lượt chạy**.

**Slide 1.3 — Phát biểu chung** (dùng chính công thức (1)–(2) của paper GEPA, vốn đã bao cả hai):
```
Hệ LLM:  y = Φ(x; ⟨Π, Θ⟩)        Π = prompt (câu lệnh),  Θ = trọng số
Mục tiêu:  ⟨Π*, Θ*⟩ = arg max  E_(x,m)~task [ μ(Φ(x; ⟨Π, Θ⟩), m) ]
Ràng buộc: rất ít dữ liệu và compute cho task mới
```
Ý người nói: *mọi cách thích nghi LLM đều là chọn **Π**, chọn **Θ**, hoặc cả hai.*

**Slide 1.4 — Hai lời giải của hôm nay.**

| | Text-to-LoRA | GEPA |
|---|---|---|
| Chỉnh gì | **Θ** (thêm LoRA ΔW), giữ Π | **Π** (viết lại câu lệnh), giữ Θ |
| Từ đâu | Một **câu mô tả task** | **Phản hồi bằng chữ** từ các lần chạy |
| Cách tìm lời giải | Một mạng đã học sẵn **sinh** ΔW trong 1 lần forward | **Tìm kiếm** tiến hoá, LLM tự suy ngẫm |

Điểm chung: **dùng ngôn ngữ tự nhiên để thích nghi LLM**. Điểm khác: kiến thức cuối cùng nằm **trong trọng số** hay **trong prompt**.

**Slide 1.5 — Khung so sánh dùng cho cả buổi.** Mỗi phương pháp sẽ được trả lời đúng 3 câu: **Ý tưởng** (vì sao nên làm vậy) · **Method** (làm thế nào) · **Evaluation** (đo ra sao, kết quả, hạn chế). Kết thúc bằng câu hỏi mở.

---

## Phần 2 — Lời giải 1: Text-to-LoRA (Người 1, 45')

> Bản cập nhật theo bản PPTX nhóm sửa: 26 slide (2.1–2.26). Thêm 5 slide giải thích (2.5, 2.9, 2.11, 2.13, 2.15) và **"hộp kiến thức nền"** (2.4, 2.7, 2.8, 2.12, 2.19–2.20), mỗi hộp gồm ba ý — paper gốc làm gì, vì sao T2L chọn nó, nó hơn phương án thay thế ở đâu. Slide **dự phòng** (đưa vào Q&A nếu thiếu giờ): 2.8, 2.9, 2.12, 2.13 (khoảng 5'). **Không cắt 2.4 và 2.19–2.20.** Hình ở 2.5, 2.6, 2.15 và lời giải thích ở 2.13 là của nhóm, chân slide đã ghi rõ.

### 2A. Ý tưởng và nền tảng (12')

**Slide 2.1 — Bài toán (2').** Có một task mới cần model làm tốt, ta làm gì? Cách thường gặp: gom dữ liệu, train một LoRA riêng — task nào cũng làm lại từ đầu. Hai câu hỏi của paper: (1) có tận dụng được kiến thức từ các task cũ (hàng trăm LoRA đã có) không; (2) chỉ từ một dòng mô tả, có sinh ra được hai ma trận LoRA (A, B) cho task mới không.

**Slide 2.2 — Các cách làm trước đó và chỗ hổng (3').**

| Hướng | Đại diện | Làm gì | Hổng ở đâu |
|---|---|---|---|
| Nén LoRA | Compress-then-Serve (Brüel-Gabrielsson 2024), VeRA (Kopiczko 2024) | Dùng SVD hoặc ma trận chung để giảm dung lượng | Chỉ nén LoRA đã có, không sinh cho task mới |
| Trộn hoặc chọn LoRA | Arrow Routing (Ostapenko 2024) | Chọn LoRA phù hợp trong thư viện | Bị giới hạn trong thư viện |
| Hypernetwork dùng ID task | Mahabadi 2021, He 2022 | Sinh adapter từ vector ID học được | Task mới chưa có ID → không zero-shot |
| Hypernetwork dùng input | Hyperdecoders (Ivison & Peters 2022) | Sinh adapter từ từng câu hỏi | Sinh lại mỗi câu nên tốn; người dùng không điều khiển được |

Câu chuyển ý: "T2L: sinh LoRA mới cho task mới, từ câu mô tả do người dùng viết."

**Slide 2.3 — Trực giác (2').** LoRA thủ công (mỗi task tìm dữ liệu, train LoRA; bộ sau không học được gì từ bộ trước) so với Text-to-LoRA tự động (đã học từ hàng trăm task, chỉ cần mô tả là ra ngay ma trận LoRA). Giả thuyết: các LoRA khác nhau nhưng có chung một cơ chế biểu diễn. Ví von máy may chỉ để trong ghi chú người nói.

**Slide 2.4 — Hộp kiến thức: Hypernetwork (5', quan trọng nhất).**
- *Paper gốc:* HyperNetworks (Ha, Dai, Le, 2016), `W_l = h_θ(z_l)`: mạng nhỏ sinh trọng số cho mạng lớn; train một lần, gặp đầu vào mới thì sinh trọng số ngay. Điều kiện: các trọng số cần sinh có cách biểu diễn giống nhau (cùng hình dạng, cùng vai trò). Ý cốt lõi: mã hoá gián tiếp.
- *Vì sao T2L chọn:* output là số thực (trọng số); nhỏ (5–55M so với 7B).
- *Hơn phương án khác:* fine-tune cần dữ liệu và thời gian; chọn từ thư viện bị giới hạn; hypernetwork sinh LoRA mới trong một forward.

### 2B. Method (16', thêm 5' slide dự phòng)

**Slide 2.5 — Công thức bản ngây thơ (2', mới).** `ΔW = h_θ(f(z))`: sinh cả LoRA một lần. Cả LoRA là 64 cặp A, B (2 module q, v × 32 lớp), 3,4M số. Vấn đề: lớp cuối (FC) nối từ 512 chiều ra 3,4M số → khoảng 1,7 tỷ tham số (512 × 3,4M), lớn hơn chính LoRA khoảng 500 lần. Hình minh hoạ của nhóm.

**Slide 2.6 — Công thức cốt lõi, cách T2L làm (2').** `ΔW_{m,l} = h_θ(concat[f(z), E[m], E[l]])`. Phụ lục F: f(z) 1024 chiều → linear còn 64 chiều; E[m], E[l] mỗi cái 32 chiều; gộp → các khối MLP có residual → linear head. Hình của nhóm: 64 hàng (m, l) trong một batch → thân MLP dùng chung → head nhỏ (bản L ≈ 34M, = 512 × 2 × 8 × 4096) → mỗi hàng một cặp A, B. Head nhỏ hơn bản ngây thơ khoảng 50 lần.

**Slide 2.7 — Hộp kiến thức: vì sao output là LoRA (1').** LoRA gốc (Hu et al., 2022): `h = W₀x + BᵀAx`, khởi tạo B = 0. W một lớp ~16,8M số, cặp LoRA rank 8 ~65k số. Không sinh prefix như Gisting: prefix chỉ tác động qua attention, LoRA sửa trọng số nhiều lớp và gộp được vào W₀. Có sẵn thư viện LoRA công khai.

**Slide 2.8 — Hộp kiến thức: text encoder và embedding vị trí (1', dự phòng).** gte-large-en-v1.5 (Alibaba, Li et al. 2023), CLS 1024 chiều. Vai trò: biến câu mô tả thành vector để hypernetwork đọc được; mô tả gần nghĩa → vector gần nhau → LoRA gần giống. E[m], E[l] giống positional embedding (cách hiểu của nhóm: để mạng tự học các lớp gần nhau cần LoRA gần giống).

**Slide 2.9 — Một câu mô tả đi qua encoder (2', mới, dự phòng).** Mô tả → gte-large (đóng băng, đọc hai chiều, lấy CLS) → 1024 số → linear (train cùng T2L) → 64 số = f(z). Vì sao encoder riêng: rẻ (~0,029 TFLOPs mỗi mô tả, chạy một lần cho cả task), tách khỏi base model (HyperTuning, HINT phải gắn với base model). Bảng 4: Mistral làm encoder 65,6 so với gte 65,9 → không kén encoder.

**Slide 2.10 — Ba kích thước L / M / S (2').** Chia nhỏ head: L sinh cả A và B (55M), M sinh A hoặc B + embedding A/B 128 chiều (34M), S sinh từng hàng + embedding rank (5M). Cột mới "số lần gọi cho cả LoRA": L 64, M 128, S 1024 (= 64 cặp × số mảnh mỗi cặp). Head nhỏ thì chia sẻ mạnh hơn nhưng dung lượng thấp hơn.

**Slide 2.11 — Chọn bản nào? Gọi nhiều lần có chậm không? (2', mới).** Bảng 2: L 67,7, M 67,5, S 65,9; Bảng 3: S bão hoà (65,9 ở 256 task, 65,2 ở 479). M là lựa chọn cân bằng: kém L 0,2 điểm, ít hơn ~38% tham số. **Lưu ý trung thực:** paper báo kết quả chính bằng L; phân tích FLOPs (Phụ lục I) dùng M. Gọi nhiều lần không chậm: các lần gọi là các hàng của cùng một batch, GPU chạy song song; hypernetwork M tốn 0,000005 TFLOPs so với 0,83 TFLOPs của LLM cho một câu hỏi. Kèm Hình 2 của paper.

**Slide 2.12 — Hộp kiến thức: Bias-HyperInit (1', dự phòng).** Beck et al. (2023), meta-RL: weight head = 0, bias khớp khởi tạo lớp đích. Trong T2L: bias head A ~ U(−1/d, 1/d), head B = 0 → ban đầu sinh LoRA chuẩn ΔW = 0. Không có thì train không ổn định, thường thất bại (Phụ lục G).

**Slide 2.13 — Vì sao phải khởi tạo để ΔW = 0? (1', mới, dự phòng).** Cách hiểu của nhóm (paper chỉ nói "thường thất bại"): khởi tạo ngẫu nhiên → ΔW ngẫu nhiên cộng vào mọi lớp → LLM hỏng ngay bước đầu, loss lớn, gradient lớn dội về hypernetwork. Bias-HyperInit → ΔW = 0 → bước đầu chính là LLM gốc, T2L chỉ học độ lệch nhỏ. Cùng ý với LoRA gốc khởi tạo B = 0.

**Slide 2.14 — Hai cách train (2').** Reconstruction: loss L1 (thực tế dự đoán z-score); 9 LoRA ~10 phút, 479 LoRA ~10 giờ; chạy được trên GPU phổ thông. SFT: gắn LoRA sinh ra vào LLM đóng băng, train trên 479 task SNI; tốn bộ nhớ hơn nhưng tổng quát tốt hơn. **Paper không ghi thời gian train SFT** (Phụ lục H chỉ nói mọi model vừa một H100 80GB). Hộp SNI: hơn 1600 task; T2L dùng 500 → bỏ 10 trùng benchmark + 11 validation → 479; mô tả GPT-4o-mini (128 mỗi task) 67,7 so với định nghĩa SNI 66,5 (Bảng 9).

**Slide 2.15 — Hai cách train, từng bước (2', mới).** Sơ đồ của nhóm. Reconstruction ("chép bài"): mô tả → T2L sinh Â, B̂ → so từng số với LoRA thật → loss L1; gradient chỉ qua T2L. SFT ("làm bài thật"): mô tả → T2L sinh ΔW → LLM đóng băng + ΔW trả lời câu hỏi → cross-entropy với đáp án mẫu; gradient đi ngược qua cả LLM (không cập nhật LLM) rồi vào T2L. SFT 66,3 so với reconstruction 61,8 (Bảng 6).

**Slide 2.16 — Vì sao reconstruction không tổng quát hoá (2').** Phụ lục D: trong không gian A, B tương quan ≈ 0. Phụ lục K: trong ΔW = BᵀA có tương quan dương (0,17–0,38) nhưng reconstruct trong ΔW vẫn không thắng. SFT 66,3 so với reconstruction 61,8 (Bảng 6).

**Slide 2.17 — Quay lại bài toán cừu (Hình 4) (1').** Gốc 240 sai; "solve this task please" 240 sai; mô tả toán 260 đúng; mô tả lập trình 260 đúng, cách giải khác. [demo tuỳ chọn]

### 2C. Evaluation (16')

**Slide 2.18 — Thiết lập (2', dạng bảng).** Model chính Mistral-7B-Instruct; thêm Llama-3.1-8B, Gemma-2-2B (cùng bộ hyperparameter); LoRA rank 8 trên q_proj, v_proj (3,4M tham số); dữ liệu train 479 task SNI, 128 mô tả mỗi task; 10 benchmark; T2L = TB 3 mô tả; 10 task SNI trùng benchmark đã bị loại. Lưu ý trung thực: benchmark chủ yếu là trắc nghiệm, khá giống task train; code không có trong dữ liệu train.

**Slide 2.19–2.20 — Hộp kiến thức: các baseline (4', không cắt; hai slide).**

| Baseline | Paper gốc làm gì | Vì sao được chọn | T2L hơn ở đâu |
|---|---|---|---|
| Prepend mô tả / 3-shot ICL (Brown 2020) | Đưa mô tả hoặc ví dụ vào prompt | Mốc "không thích nghi trọng số" | 67,7 so với 61,0; rẻ hơn ICL hơn 4 lần FLOPs |
| Multi-task LoRA | Một LoRA train trên cả 479 task | Cùng dữ liệu, không có mô tả — đối chứng sạch nhất | **67,7 so với 66,3**: giá trị của thích nghi theo mô tả |
| Arrow Routing (Ostapenko 2024) | Vector đại diện từ SVD của ΔW, mỗi token chọn/trộn LoRA | Mạnh nhất của hướng chọn/trộn, zero-shot | 73,9 so với 70,7 trên 8 task — chỉ so gián tiếp |
| Hyperdecoders (Ivison & Peters 2022) | Sinh adapter theo từng câu | Gần T2L nhất; tác giả cài lại bằng kiến trúc T2L | Điểm gần bằng (67,7 so với 67,3), nhưng sinh một lần cho cả task, gộp được, điều khiển bằng mô tả |
| Task-specific LoRA (oracle) | Train riêng trên dữ liệu benchmark | Trần trên | T2L chưa bằng: 73,9 so với 75,8 trên 8 task |

Related work khác: HyperTuning (Phang 2023), HINT (Ivison 2023) dùng T5/BART và gắn encoder với base model; HyperTuning và Lv et al. (2024) cần few-shot. T2L chỉ cần mô tả.

**Slide 2.21 — Câu hỏi 1: nén được không? (Bảng 1) (2').** T2L tái tạo 9 LoRA đạt **73,0–73,5** (tuỳ kiến trúc), ngang LoRA riêng 73,3. Có chỗ còn hơn (PIQA, Winogrande): nén có mất mát như regularization. Lưu ý: "LoRA gốc thấp hơn cả base model" chỉ đúng với PIQA (69,9 so với 72,8); với Winogrande LoRA gốc 51,1 vẫn cao hơn base 45,0.

**Slide 2.22 — Câu hỏi 2: zero-shot (Bảng 2) (3').** Nhấn **T2L 67,7 so với multi-task LoRA 66,3** (giá trị của mô tả) và so với Hyperdecoders 67,3 (ngang điểm, lợi thế cách dùng). **Không** dùng 67,7 so với 60,6 làm bằng chứng về mô tả, vì dòng "mô tả trong prompt" không được train gì.

**Slide 2.23 — Tổng quát sang model khác (Bảng 7, 8) (1').** Llama-3.1-8B: 73,2 → 76,9 (multi-task LoRA 76,5). Gemma-2-2B (kèm ICL): 60,7 → 66,4 (multi-task 65,2). Trên Gemma, GSM8K gần như không đổi (55,6 → 55,1). Cùng một bộ hyperparameter.

**Slide 2.24 — Ablation chọn lọc (2').** Mô tả khớp task 72–73; mô tả của task khác 51,4; chuỗi ngẫu nhiên 63,5 (Bảng 5). Thêm task train và compute thì tốt hơn, trừ S (Bảng 3). t-SNE: MBPP cạnh HumanEval (Hình 5).

**Slide 2.25 — Chi phí và hạn chế (2').** 0,86 TFLOPs mỗi câu so với 3-shot ICL 4,18 (ước tính ad-hoc, chỉ tính token đầu vào). Hạn chế: chưa bằng LoRA riêng; phụ thuộc mô tả; chỉ sinh LoRA (gợi ý điều chỉnh activation); chưa thử chuyển từ model nhỏ sang lớn.

**Slide 2.26 — Tóm tắt** (dùng lại ở Phần 4): *"Biên dịch một câu mô tả thành trọng số, trong một lần forward, không cần dữ liệu task."*

---

## Phần 3 — Lời giải 2: GEPA (Người 2, 50')

### 3A. Nền tảng nhanh: DSPy (7')

**Slide 3.1 — Chương trình LLM thay cho prompt viết tay.** Signature (`"question -> answer"`), Module (`ChainOfThought`), Program (ghép module), Metric, Optimizer. Bảng ví von PyTorch ↔ DSPy (file `docs/01`, mục 4).

**Slide 3.2 — Optimizer của DSPy làm gì.** Chạy chương trình, giữ các lần cho đáp án đúng, dùng làm ví dụ few-shot (BootstrapFewShot, MIPROv2). GEPA là một optimizer **khác** của DSPy: thay vì chọn ví dụ, nó **viết lại câu lệnh**.

### 3B. Ý tưởng (10')

**Slide 3.3 — Vấn đề cụ thể.** RL (GRPO) nén cả một lần chạy thành **một con số** (đúng/sai) → cần 24.000 rollout (Hình 1).

**Slide 3.4 — Trực giác.** Một lần chạy của LLM là **văn bản**: suy luận, lời gọi công cụ, thông báo lỗi. LLM **đọc hiểu** được văn bản. Vậy hãy cho LLM **đọc lại lỗi và tự sửa câu lệnh**. Ví von: RL giống chấm bài chỉ ghi điểm; GEPA giống giáo viên **ghi nhận xét** "sai ở bước cộng".

**Slide 3.5 — Quay lại câu hỏi cừu** (ví dụ **minh hoạ của nhóm**, không có trong paper):
- Câu lệnh gốc: "Given the fields `question`, produce the fields `answer`." → 240 ✗
- Phản hồi: "Đáp án đúng 260. Lời giải tính đúng Charleston = 80, Toulouse = 160 nhưng cộng sai tổng."
- Reflection LM viết lại câu lệnh, thêm: "Liệt kê số lượng từng đối tượng, rồi cộng lại và kiểm tra phép cộng trước khi trả lời." → 260 ✓

Ý người nói: *T2L cũng sửa được câu này, nhưng bằng trọng số. GEPA sửa bằng chữ, và ta đọc được nó đã sửa gì.*

### 3C. Method (15')

**Slide 3.6 — Vòng lặp tiến hoá** (Hình 3, Thuật toán 1): pool ứng viên → chọn một ứng viên → chạy trên minibatch 3 ví dụ → suy ngẫm → sinh prompt con → giữ nếu điểm minibatch tăng → chấm trên tập Pareto.

**Slide 3.7 — Hàm phản hồi μ_f.** Trả về **điểm + chữ**: ràng buộc nào chưa đạt (IFBench), tài liệu nào còn thiếu (HotpotQA, HoVer), lỗi biên dịch (code). Đây là chỗ GEPA lấy được nhiều thông tin hơn RL.

**Slide 3.8 — Meta-prompt suy ngẫm** (Phụ lục C, chiếu nguyên văn): "Tôi đưa trợ lý câu lệnh sau… đây là input, câu trả lời và phản hồi… hãy viết câu lệnh mới, đưa mọi kiến thức chuyên biệt của miền và chiến lược khái quát vào." Ý: *"bộ tối ưu" chỉ là một prompt bảo LLM viết lại prompt, đặt trong vòng lặp có chọn lọc.*

**Slide 3.9 — Chọn ứng viên theo Pareto** (Hình 4, Thuật toán 2): giữ mọi ứng viên **thắng ít nhất một ví dụ**, chọn ngẫu nhiên theo số ví dụ thắng. So với luôn chọn tốt nhất (cây thành "chổi", kẹt sau 1 bước).

**Slide 3.10 — Merge.** Ghép hai nhánh đã tối ưu các module khác nhau (Phụ lục D.1).

**Slide 3.11 — Prompt trước và sau** (Hình 2): một dòng → một bản hướng dẫn có "bài học rút ra".

**[demo tuỳ chọn]** Chạy GEPA ngân sách nhỏ, in prompt qua 1–2 vòng.

### 3D. Evaluation (18')

**Slide 3.12 — Thiết lập.** 6 benchmark (HotpotQA, IFBench, HoVer, PUPA, AIME-2025, LiveBench-Math); Qwen3-8B và GPT-4.1 mini; so với MIPROv2, TextGrad, Trace, **GRPO dùng LoRA** (24.000 rollout); ngân sách GEPA bằng MIPROv2.

**Slide 3.13 — Kết quả chính** (Bảng 1, Qwen3-8B, điểm tổng hợp):

| Phương pháp | Tổng hợp | Rollout |
|---|---|---|
| Baseline | 45.23 | — |
| MIPROv2 | 47.84 | |
| GRPO (LoRA) | 48.91 | 24.000 |
| **GEPA** | **54.85** | 1.839–7.051 |

Hơn GRPO tới 19 điểm, ít rollout hơn tới 35 lần; thắng 5/6 benchmark (thua AIME: 32.00 so với 38.00).

**Slide 3.14 — Hiệu quả mẫu** (Hình 1): GEPA bằng điểm validation tốt nhất của GRPO sau 243–1.179 rollout; nếu chỉ tính rollout train, cần 79–737.

**Slide 3.15 — Model đóng và chuyển model** (Bảng 2): GPT-4.1 mini 53.03 → **66.36** (GEPA+Merge). Prompt tối ưu trên Qwen3-8B dùng cho GPT-4.1 mini vẫn +9.

**Slide 3.16 — Ablation.** Pareto 61.28 so với luôn chọn tốt nhất 54.89 và beam search 53.95 (Bảng 3). Prompt của GEPA ngắn hơn tới 9.2 lần so với MIPROv2.

**Slide 3.17 — Chi phí.** Toàn bộ GEPA trên 6 benchmark với GPT-4.1 mini khoảng 86 USD.

**Slide 3.18 — Hạn chế.** Phần lớn rollout tốn cho validation; Merge có lúc làm giảm (Qwen3-8B); không thêm được kiến thức model chưa có; cần dữ liệu có nhãn và hàm phản hồi tốt.

**Slide 3.19 — Tóm tắt lời giải 2:** *"Tìm câu lệnh tốt hơn bằng cách cho LLM đọc lỗi của chính nó, vài trăm lượt chạy, không đụng trọng số."*

---

## Phần 4 — Đặt cạnh nhau (Người 3, 15')

**Slide 4.1 — Cùng bài toán, hai lời giải** (trở lại công thức ở Slide 1.3):
- T2L: `Θ = Θ₀ + h_θ(f(z))`, Π giữ nguyên.
- GEPA: `Π = tiến hoá(Π₀, phản hồi)`, Θ giữ nguyên.

**Slide 4.2 — Bảng so sánh.**

| Tiêu chí | Text-to-LoRA | GEPA |
|---|---|---|
| Thông tin dùng để thích nghi | Một câu mô tả + kiến thức đã nén từ 479 task | Dữ liệu task + phản hồi chữ |
| Cần dữ liệu task mới | **Không** | Có (khoảng 100–450 ví dụ trong paper) |
| Chi phí trước | **Lớn**, một lần: train hypernetwork bằng SFT trên 479 task (paper không ghi thời gian) | Không |
| Chi phí mỗi task mới | **Một lần forward** | Vài trăm tới vài nghìn rollout + reflection LM |
| Chi phí mỗi câu hỏi lúc dùng | Không thêm token | Prompt dài hơn (nhưng ngắn hơn few-shot) |
| Đọc hiểu được | Không (LoRA là ma trận số) | **Có** (prompt là chữ) |
| Model đóng (API) | Không dùng được | **Dùng được** |
| Điểm yếu | Mô tả kém → LoRA kém; chưa bằng LoRA riêng | Không thêm kiến thức mới; cần hàm phản hồi |

**Slide 4.3 — Khi nào dùng cách nào** (gợi ý của nhóm):
- Không có dữ liệu, cần nhanh, có trọng số model → **T2L**.
- Có ít dữ liệu, dùng model qua API, cần đọc được thay đổi, pipeline nhiều bước → **GEPA**.

**Slide 4.4 — Cầu nối sang câu hỏi mở.** Hai phương pháp chỉnh **hai biến khác nhau** của cùng công thức (Θ và Π), và cả hai đều nhận **chữ** làm đầu vào. Vậy chúng có thể **cộng dồn** không?

---

## Phần 5 — Câu hỏi mở: kết hợp hai cách như thế nào? (25')

> Phần này là **đề xuất và câu hỏi của nhóm**, không có trong hai paper. Nói rõ với khán giả.

**Slide 5.1 — Bằng chứng rằng kết hợp có thể có lợi.** BetterTogether (EMNLP 2024, cùng nhóm tác giả DSPy với GEPA): xen kẽ tối ưu prompt và fine-tune LoRA tốt nhất ở 7/9 cặp dataset × model, hơn chỉ tối ưu một trong hai. (Tài liệu: `papers/reference/previous/bettertogether/`.)

**Slide 5.2 — Hướng 1: Xếp chồng (dễ làm nhất).** T2L sinh LoRA từ mô tả → **GEPA tối ưu prompt cho model đã gắn LoRA**.
- Câu hỏi: hai mức cải thiện có cộng dồn không, hay chồng lấn nhau?
- Thử được ngay với checkpoint Gemma-2-2B có sẵn.

**Slide 5.3 — Hướng 2: GEPA tối ưu chính câu mô tả của T2L** (hướng nhóm thấy hứa hẹn nhất).
- Tham số được tiến hoá **không phải prompt của LLM** mà là **câu mô tả z** đưa vào T2L. Mỗi ứng viên z → T2L sinh LoRA → chạy trên minibatch → phản hồi → reflection LM viết lại z.
- Cơ sở: paper T2L cho thấy **mô tả quyết định chất lượng LoRA** (Bảng 5, Hình 4) và chính tác giả gợi ý dùng LLM để chỉnh mô tả (mục 5.3). GEPA chính là cách làm việc đó có hệ thống.
- Thực hiện được: thư viện `gepa` có API `optimize_anything` tối ưu **bất kỳ chuỗi văn bản nào** theo một hàm đánh giá.
- Ưu điểm: kết quả cuối **vẫn là LoRA** (không tốn token mỗi câu) nhưng được tìm bằng phản hồi thật từ dữ liệu.
- Rủi ro: mỗi lần chấm phải sinh LoRA và chạy model (rẻ vì sinh LoRA chỉ một forward); mô tả có thể "lệch" khỏi phân phối mô tả lúc train T2L.

**Slide 5.4 — Hướng 3: Biên dịch prompt của GEPA vào trọng số.** Câu lệnh GEPA tìm được thực chất là **một mô tả task rất chi tiết**. Đưa nó vào T2L làm z → được LoRA, bỏ được prompt dài lúc dùng.
- Câu hỏi: T2L train với mô tả ngắn (khoảng 48 token) có hiểu được câu lệnh dài và chi tiết của GEPA không?

**Slide 5.5 — Hướng 4 (nghiên cứu): Hypernetwork nhận phản hồi.** Thay vì chỉ nhận mô tả, hypernetwork nhận **phản hồi lỗi** và sinh **bản cập nhật trọng số** ("Feedback-to-LoRA"). Liên quan: Doc-to-LoRA (ICML 2026, sinh LoRA từ tài liệu), SEAL (NeurIPS 2025, model tự viết dữ liệu fine-tune).

**Slide 5.6 — Hướng 5: Pareto trên cặp (prompt, LoRA).** Mỗi ứng viên trong pool của GEPA là một cặp (câu lệnh, mô tả cho T2L). Merge của GEPA có thể mở rộng thành ghép hai LoRA.

**Slide 5.7 — Bảng tổng hợp các hướng.**

| Hướng | Tối ưu cái gì | Kết quả cuối | Độ khó | Câu hỏi chính |
|---|---|---|---|---|
| 1. Xếp chồng | Π trên model + LoRA | LoRA + prompt | Dễ | Có cộng dồn không? |
| 2. GEPA tối ưu mô tả z | z (đầu vào T2L) | LoRA | Vừa | Phản hồi có tìm được mô tả tốt hơn người viết không? |
| 3. Prompt GEPA → T2L | Không thêm | LoRA | Dễ | T2L có hiểu câu lệnh dài không? |
| 4. Feedback-to-LoRA | Train hypernetwork mới | LoRA | Khó (nghiên cứu) | Phản hồi chữ có đủ để sinh cập nhật trọng số? |
| 5. Pareto trên cặp | (Π, z) | LoRA + prompt | Vừa | Ghép LoRA có giữ được lợi ích không? |

**Slide 5.8 — Thí nghiệm nhỏ nhóm có thể làm** (nếu có thời gian chuẩn bị): Gemma-2-2B-it + checkpoint T2L Gemma, task PIQA hoặc HellaSwag (nơi T2L giúp rõ trên Gemma), so sánh: mô tả do người viết · mô tả GEPA tìm được (Hướng 2) · LoRA + prompt GEPA (Hướng 1). Nếu không kịp làm, trình bày như **đề xuất thí nghiệm**.

**Slide 5.9 — Câu hỏi thảo luận cho khán giả.**
1. Kiến thức nào nên nằm trong **trọng số**, kiến thức nào nên nằm trong **prompt**?
2. Khi đã có T2L, việc viết prompt có còn quan trọng? (Gợi ý: Hình 4 cho thấy có, chỉ là chuyển sang viết mô tả.)
3. Nếu GEPA tối ưu mô tả cho T2L, ta đang làm **prompting** hay **PEFT**?
4. Kết hợp có đáng với chi phí tăng thêm không?

---

## Phần 6 — Kết luận + Q&A (15')

**Slide 6.1 — Ba ý mang về.**
1. Hai paper giải **cùng một bài toán**: thích nghi LLM đã freeze, ít dữ liệu, ít compute, bằng **ngôn ngữ**.
2. **T2L**: biên dịch mô tả thành trọng số, không cần dữ liệu task. **GEPA**: tìm prompt tốt hơn từ phản hồi, thắng RL với ít lượt chạy hơn.
3. Chúng chỉnh **hai biến khác nhau** (Θ và Π) nên **có thể kết hợp**: đây là câu hỏi mở.

**Slide 6.2 — Q&A.** Câu hỏi phản biện đã chuẩn bị: xem `papers/01-text-to-lora/README.md` và `papers/02-gepa/README.md`, mục "Điểm cần nhấn".

---

## Phụ lục — hình và bảng cần chuẩn bị

| Slide | Nguồn | Ghi chú |
|---|---|---|
| 0.2, 2.17 | T2L Hình 4 | Vẽ lại dạng 4 ô, dịch sang tiếng Việt |
| 1.3, 4.1 | GEPA công thức (1)–(2) | Vẽ thành sơ đồ Φ⟨Π, Θ⟩ với hai mũi tên: T2L chỉnh Θ, GEPA chỉnh Π |
| 2.1 | T2L Hình 1 (trái) | |
| 2.11 | T2L Hình 2 | |
| 2.5, 2.6, 2.15 | Nhóm tự vẽ | Sơ đồ head ngây thơ / head nhỏ; hai cách train. Ghi rõ "minh hoạ của nhóm" |
| 2.16, 2.21–2.24 | T2L Bảng 1, 2, 5, 6, 7, 8; Phụ lục D, K; Hình 5 | Rút gọn còn dòng chính |
| 3.3, 3.14 | GEPA Hình 1 | |
| 3.6 | GEPA Hình 3 | |
| 3.8 | GEPA Phụ lục C | Nguyên văn tiếng Anh + dịch |
| 3.9 | GEPA Hình 4, Bảng 3 | |
| 3.11 | GEPA Hình 2 | |
| 3.13, 3.15 | GEPA Bảng 1, 2 | |
| 3.5 | Nhóm tự làm | Ghi rõ "minh hoạ của nhóm" |
| 5.1 | BetterTogether Bảng 1 | Chỉ một dòng kết luận |
| 5.8 | Kết quả thí nghiệm của nhóm | Lấy từ `04-thiet-ke-thi-nghiem-va-app.md` khi có |
