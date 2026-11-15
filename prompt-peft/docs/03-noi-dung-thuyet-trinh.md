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

Khoảng 45 slide. Phần 2 và 3 dùng **cùng khung slide** (cùng thứ tự, cùng kiểu hình) để khán giả dễ so sánh.

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

### 2A. Ý tưởng (10')

**Slide 2.1 — Vấn đề cụ thể.** Mỗi task một LoRA → hàng trăm adapter, không chia sẻ kiến thức. Paper hỏi hai câu:
1. Có thể nén hàng trăm LoRA vào **một mạng** không?
2. Có thể **sinh LoRA cho task chưa thấy chỉ từ mô tả bằng chữ** không?

**Slide 2.2 — Trực giác.** Các LoRA khác nhau "có chung một cơ chế thích nghi". Nếu học được cơ chế đó thì task mới chỉ cần **nói** cho mạng biết task là gì. Ví von: thay vì may đo từng bộ đồ (train từng LoRA), ta có **một máy may tự động** nhận mô tả "áo sơ mi công sở, cỡ M".

**Slide 2.3 — Hypernetwork là gì.** Mạng sinh **trọng số cho mạng khác**: `W_l = h_θ(φ_l)`. Ở đây: sinh A, B của LoRA cho từng module, từng layer.

### 2B. Method (17')

**Slide 2.4 — Công thức cốt lõi** (Hình 2 của paper):
```
ΔW_{m,l} = h_θ( concat[ f(z), E[m], E[l] ] )
z: mô tả task bằng chữ     f: gte-large-en-v1.5 (embedding câu)
E[m]: embedding loại module (q_proj, v_proj)    E[l]: embedding layer
```
Gộp mọi (m, l) vào một batch → **cả LoRA sinh trong một lần forward**.

**Slide 2.5 — Ba kích thước L / M / S.** Khác nhau ở head output: L sinh A và B cùng lúc (55M tham số), M dùng chung head cho A hoặc B (34M), S sinh từng rank một (5M). So sánh: một LoRA rank 8 (q, v) của Mistral-7B có 3.4M tham số.

**Slide 2.6 — Hai cách train.**
- **Reconstruction**: bắt chước thư viện LoRA có sẵn, loss L1. Rẻ (9 LoRA khoảng 10 phút).
- **SFT**: train thẳng trên dữ liệu của 479 task SNI, mỗi task 128 mô tả do GPT-4o-mini viết. Đắt (cần lan gradient qua LLM gốc), nhưng **khái quát tốt hơn**.

**Slide 2.7 — Vì sao reconstruction không khái quát** (Phụ lục D, slide "bất ngờ"): các LoRA **làm cùng việc lại không nằm gần nhau** trong không gian trọng số (tương quan ≈ 0). Bắt chước từng LoRA thì không học được "quy luật"; SFT học thẳng hành vi nên tránh được.

**Slide 2.8 — Quay lại câu hỏi cừu** (Hình 4). Cùng model Mistral-7B, chỉ đổi LoRA:
- Gốc → 240 ✗
- LoRA từ mô tả "solve this task please" → 240 ✗
- LoRA từ mô tả "This task challenges your problem-solving abilities through mathematical reasoning…" → **260 ✓**
- LoRA từ mô tả "Use your programming skill to think algorithmically…" → **260 ✓**, cách giải khác

Ý người nói: *chất lượng câu mô tả quyết định chất lượng LoRA. Prompting vẫn quan trọng, chỉ là nó được "biên dịch" thành trọng số.*

**[demo tuỳ chọn]** Gõ mô tả → sinh LoRA → hỏi lại câu cừu.

### 2C. Evaluation (18')

**Slide 2.9 — Thiết lập.** Model chính Mistral-7B-Instruct (thêm Llama-3.1-8B, Gemma-2-2B). 10 benchmark: ARC-C/E, BoolQ, GSM8K, HellaSwag, OpenBookQA, PIQA, Winogrande, HumanEval, MBPP. Đã loại 10 task SNI trùng với benchmark. Mỗi kết quả T2L là trung bình 3 mô tả.

**Slide 2.10 — Câu hỏi 1: nén được không?** (Bảng 1) T2L tái tạo 9 LoRA đạt 73.3–73.5 điểm TB, ngang LoRA riêng (73.3); có chỗ còn hơn nhờ nén có mất mát hoạt động như regularization.

**Slide 2.11 — Câu hỏi 2: zero-shot trên task chưa thấy** (Bảng 2, TB 10 task):

| Phương pháp | TB |
|---|---|
| Mistral-7B gốc | 55.8 |
| + mô tả task trong prompt | 60.6 |
| + 3-shot ICL | 61.0 |
| Multi-task LoRA | 66.3 |
| Hyperdecoders | 67.3 |
| **T2L (SFT) L** | **67.7** |

Nhấn: dòng "mô tả trong prompt" là đối chứng quan trọng. **Cùng một câu mô tả**, đưa vào trọng số (67.7) tốt hơn hẳn đưa vào prompt (60.6).

**Slide 2.12 — Tổng quát sang model khác** (Bảng 7, 8): Llama-3.1-8B 73.2 → 76.9; Gemma-2-2B (kèm ICL) 60.7 → 66.4. Lưu ý trung thực: trên Gemma, GSM8K gần như không đổi (55.6 → 55.1).

**Slide 2.13 — Ablation chọn lọc.**
- Mô tả khớp task: 72–73 điểm; mô tả của task khác: 51.4; chuỗi ngẫu nhiên: 63.5 (Bảng 5).
- SFT 66.3 so với reconstruction 61.8 (Bảng 6).
- Thêm task train + thêm compute → tốt hơn (Bảng 3).
- t-SNE: LoRA của task giống nhau nằm gần nhau, MBPP cạnh HumanEval (Hình 5).

**Slide 2.14 — Chi phí.** Sinh LoRA + trả lời: 0.86 TFLOPs mỗi câu; 3-shot ICL: 4.18 TFLOPs. Nhưng train hypernetwork SFT: khoảng 5 ngày trên H100 mỗi model (một lần).

**Slide 2.15 — Hạn chế.** Chưa bằng LoRA train riêng; phụ thuộc chất lượng mô tả (do GPT-4o-mini viết trong paper); chỉ có checkpoint cho vài model; chỉ sinh LoRA.

**Slide 2.16 — Tóm tắt lời giải 1** (một dòng, dùng lại ở Phần 4): *"Biên dịch một câu mô tả thành trọng số, trong một lần forward, không cần dữ liệu task."*

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
| Chi phí trước | **Rất lớn**, một lần (khoảng 5 ngày H100) | Không |
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
| 0.2, 2.8 | T2L Hình 4 | Vẽ lại dạng 4 ô, dịch sang tiếng Việt |
| 1.3, 4.1 | GEPA công thức (1)–(2) | Vẽ thành sơ đồ Φ⟨Π, Θ⟩ với hai mũi tên: T2L chỉnh Θ, GEPA chỉnh Π |
| 2.4, 2.5 | T2L Hình 2 | |
| 2.10–2.13 | T2L Bảng 1, 2, 5, 6, 7, 8; Hình 5 | Rút gọn còn dòng chính |
| 3.3, 3.14 | GEPA Hình 1 | |
| 3.6 | GEPA Hình 3 | |
| 3.8 | GEPA Phụ lục C | Nguyên văn tiếng Anh + dịch |
| 3.9 | GEPA Hình 4, Bảng 3 | |
| 3.11 | GEPA Hình 2 | |
| 3.13, 3.15 | GEPA Bảng 1, 2 | |
| 3.5 | Nhóm tự làm | Ghi rõ "minh hoạ của nhóm" |
| 5.1 | BetterTogether Bảng 1 | Chỉ một dòng kết luận |
