# Thiết kế thí nghiệm và app — Text-to-LoRA vs GEPA trên cùng model, cùng dataset

> Kế hoạch (chưa có code). Code sẽ viết ở máy khác theo tài liệu này.
> Thay cho phần thiết kế app cũ trong `01-chon-paper-va-thiet-ke-app.md` (mục 5–7).
> Số liệu paper lấy từ `papers/01-text-to-lora/dich-chi-tiet.md` và `papers/02-gepa/dich-chi-tiet.md`.

---

## 1. Mục tiêu

Chạy **cả hai pipeline trên cùng một model gốc và cùng một bộ dataset**, rồi so sánh công bằng.

| Câu hỏi | Trả lời bằng |
|---|---|
| **RQ1.** Cùng model, cùng dữ liệu: đưa kiến thức vào trọng số (T2L) hay vào prompt (GEPA) tốt hơn, ở task nào? | Bảng accuracy theo task × phương án |
| **RQ2.** Kết hợp hai cách có cộng dồn không? | Phương án T2L + GEPA (mục 5) |
| **RQ3.** Mỗi cách tốn bao nhiêu (dữ liệu, GPU, tiền API, token mỗi câu)? | Bảng chi phí |

---

## 2. Model gốc chung

Ràng buộc: T2L **chỉ có checkpoint** cho ba model (repo SakanaAI/text-to-lora). Tự train T2L cho model khác là không khả thi, nên model chung phải là một trong ba model này. GEPA thì chạy được trên model nào cũng được.

| Model | Checkpoint T2L | VRAM chạy (bf16, ước lượng) | Ghi chú |
|---|---|---|---|
| **Llama-3.1-8B-Instruct** | `trained_t2l/llama_8b_t2l` | ~16GB trọng số → cần GPU ~24GB | **Đề xuất.** Mạnh nhất trong ba; paper T2L có số trên các dataset này (Bảng 7) để đối chiếu |
| Gemma-2-2B-it | `trained_t2l/gemma_2b_t2l` | ~5GB | Dự phòng nếu GPU nhỏ. Paper chạy Gemma **kèm ICL** cho mọi phương án |
| Mistral-7B-Instruct-v0.2 | `trained_t2l/mistral_7b_t2l` | ~15GB | Model chính của paper T2L, nhưng yếu hơn Llama ở toán và code |

Không dùng Qwen3-8B (model của paper GEPA) vì T2L không có checkpoint cho nó.

**Số tham chiếu** (paper T2L, Bảng 7, Llama-3.1-8B-Instruct; prompt đánh giá của paper, không phải của nhóm):

| | ARC-C | BoolQ | HellaSwag | HumanEval |
|---|---|---|---|---|
| Model gốc | 73,3 | 80,4 | 66,6 | 66,5 |
| 3-shot ICL | 80,7 | 80,0 | 59,3 | 66,5 |
| Thêm mô tả task vào prompt | 80,2 | 79,9 | 69,8 | 68,3 |
| Multi-task LoRA | 82,0 | 83,3 | 70,8 | 63,4 |
| T2L (SFT) L | 82,4 | 84,4 | 72,8 | 64,6 |

Lưu ý: với Llama, T2L **thấp hơn** model gốc ở HumanEval (64,6 so với 66,5). Đây là điểm đáng xem lại trong thí nghiệm của nhóm.

---

## 3. Bộ dataset chung

**Lưu ý trung thực:** không có dataset nào nằm trong **cả hai** paper. ARC-Challenge, HellaSwag, BoolQ, HumanEval là benchmark của T2L; AIME 2025 là benchmark của GEPA. Bộ năm dataset này **phủ cả hai paper**, nên mỗi phương pháp sẽ được thử ở "sân nhà" và "sân khách".

| Dataset | Nguồn (Hugging Face) | Train + val cho GEPA | Test | Metric | Có trong paper |
|---|---|---|---|---|---|
| **ARC-Challenge** | `allenai/ai2_arc` (ARC-Challenge) | 150 từ train + 300 từ train/validation | test (1.172 câu) | Accuracy | T2L |
| **HellaSwag** | `Rowan/hellaswag` | 150 + 300 từ train | validation (test không có nhãn); lấy 1.000 câu cố định seed nếu thiếu thời gian | Accuracy | T2L |
| **BoolQ** | `google/boolq` | 150 + 300 từ train | validation (3.270 câu, hoặc 1.000 câu cố định seed) | Accuracy | T2L |
| **HumanEval** | `openai/openai_humaneval` | **Không có split train** → dùng **MBPP** (train/validation) để GEPA tối ưu | toàn bộ 164 bài | pass@1 (chạy unit test) | T2L |
| **AIME 2025** | `MathArena/aime_2025` | AIME 2022–2024 (90 bài, chia đôi train/val, như paper GEPA) | 30 bài × 5 lần sinh | Đúng đáp án số nguyên, TB 5 lần | GEPA |

Ghi chú:
- Kích thước train/val 150/300 theo paper GEPA. Ghi lại seed và danh sách id của từng split vào `artifacts/splits/`.
- **HumanEval**: tối ưu trên MBPP rồi test trên HumanEval là **chuyển dataset** — cần nói rõ khi báo cáo.
- **AIME 2025 có rủi ro "sàn"**: model 8B instruct không chuyên suy luận thường giải được rất ít bài AIME, nên mọi phương án có thể gần 0. Chỉ có 30 bài, nên 1 bài = 3,3 điểm. **Đề xuất:** giữ AIME làm bài "thử sức", và cân nhắc **thêm GSM8K** (có trong paper T2L, cùng loại toán với GEPA, có 1.319 câu test) để có tín hiệu toán rõ hơn. Cần nhóm quyết định (mục 13).

---

## 4. Một bộ đánh giá thống nhất (harness)

Mọi phương án chạy qua **cùng một harness**, chỉ khác LoRA và/hoặc prompt. Không lấy số từ script đánh giá riêng của T2L để so với số của GEPA.

| Mục | Quy định chung |
|---|---|
| Phục vụ model | Một server **vLLM** cho model gốc, bật LoRA (`--enable-lora --max-lora-rank 8`); LoRA của T2L đăng ký như một "model name" riêng → đổi LoRA không cần nạp lại model |
| Chat template | Template gốc của model |
| Giải mã | Greedy (temperature 0) cho ARC-C, HellaSwag, BoolQ, HumanEval. AIME: temperature 0.6, 5 lần, lấy trung bình (như paper GEPA) |
| Độ dài sinh tối đa | Trắc nghiệm và BoolQ ~512 token (cho phép suy luận ngắn); HumanEval ~1.024; AIME ~4.096 |
| Định dạng đề | Trắc nghiệm: câu hỏi + phương án A/B/C/D, yêu cầu kết thúc bằng `Answer: <chữ cái>`. BoolQ: đoạn văn + câu hỏi, trả lời yes/no. HumanEval: chữ ký hàm + docstring → hoàn thành code. AIME: đề → đáp án cuối dạng `### <số>` |
| Tách đáp án | Một hàm duy nhất cho mỗi dataset, dùng chung cho mọi phương án |
| Chạy code | Sandbox có timeout cho HumanEval/MBPP (không chạy code sinh ra trực tiếp trên máy chính) |
| Báo cáo | Accuracy + **khoảng tin cậy 95% bằng bootstrap**; AIME báo cả số bài đúng |

**Kiểm tra tái hiện (bước đầu):** chạy model gốc và T2L trên ARC-C, BoolQ bằng script `run_eval.py` của repo T2L, so với Bảng 7 ở trên. Nếu lệch nhiều thì sửa cài đặt trước khi chạy harness của nhóm.

---

## 5. Các phương án so sánh

| # | Phương án | Thay đổi | Dữ liệu có nhãn dùng | Bắt buộc? |
|---|---|---|---|---|
| A0 | Model gốc, zero-shot | — | 0 | Có |
| A1 | Gốc + **cùng câu mô tả** đưa vào prompt | Π | 0 | Có — đối chứng cho T2L |
| A2 | 3-shot ICL | Π | 3 | Có (HumanEval: bỏ, như paper T2L) |
| A3 | **T2L**: LoRA sinh từ mô tả (TB 3 mô tả) | Θ | 0 | Có |
| A4 | **GEPA**: prompt tối ưu trên model gốc | Π | ~450 | Có |
| A5 | **T2L + GEPA**: GEPA tối ưu prompt cho model đã gắn LoRA của A3 | Θ + Π | ~450 | Có — trả lời RQ2 |
| A6 | GEPA tối ưu **câu mô tả** đưa vào T2L (Hướng 2 trong slide) | z → Θ | ~450 | Mở rộng |
| A7 | LoRA train riêng bằng `peft` trên train split (oracle) | Θ | train split | Mở rộng; không làm cho AIME |

**Công bằng:**
- A0–A3 dùng 0–3 ví dụ có nhãn, còn A4–A5 dùng khoảng 450. Bảng kết quả phải có cột "dữ liệu dùng". Có thể chạy thêm A4 bản ít dữ liệu (vd. 30 train + 30 val) để so trong điều kiện gần nhau hơn.
- Cả hai phương pháp đều dùng một **LLM ngoài mạnh hơn**: paper T2L dùng GPT-4o-mini viết mô tả, GEPA dùng reflection LM. **Dùng cùng một LLM ngoài** (vd. GPT-4.1 mini) cho cả hai việc, và tính chi phí của nó vào RQ3.

---

## 6. Pipeline Text-to-LoRA (A1, A3)

1. **Mô tả task** (3 mô tả cho mỗi dataset):
   - Kiểm tra file mô tả đánh giá có sẵn trong repo T2L (README nhắc `configs/hyper_lora_decontam_lol_tasks.yaml` và thư mục `tasks/`). Dùng nếu có ARC-C, HellaSwag, BoolQ, HumanEval.
   - AIME 2025 (và dataset nào thiếu): sinh bằng LLM ngoài theo prompt ở Phụ lục L của paper (đưa 5 cặp hỏi–đáp, yêu cầu mô tả ngắn, không nêu định dạng đáp án). Dùng cặp hỏi–đáp từ **train** (AIME 2022–2024), không dùng test.
   - Lưu 3 mô tả vào `artifacts/descriptions/<dataset>.json`; A1 dùng đúng các mô tả này.
2. **Sinh LoRA**: chạy `scripts/generate_lora.py` của repo T2L với checkpoint của model đã chọn → mỗi mô tả cho một thư mục adapter PEFT (rank 8, q_proj + v_proj).
3. **Đăng ký vào vLLM** như LoRA module, rồi đánh giá qua harness.
4. Báo cáo trung bình 3 mô tả, và cả từng mô tả (để thấy độ nhạy với mô tả).

## 7. Pipeline GEPA (A4, A5)

1. **Chương trình DSPy** cho mỗi dataset, giữ đơn giản và giống nhau:
   - ARC-C, HellaSwag, BoolQ, AIME: một module `ChainOfThought` (như AIME trong paper GEPA).
   - HumanEval/MBPP: một module sinh code từ chữ ký hàm + docstring.
2. **Hàm phản hồi μ_f** (trả điểm + chữ):

| Dataset | Điểm | Phản hồi bằng chữ |
|---|---|---|
| ARC-C, HellaSwag | 1/0 | "Đáp án đúng là X; bạn chọn Y." |
| BoolQ | 1/0 | "Đáp án đúng là yes/no", kèm câu liên quan trong đoạn văn nếu tìm được |
| AIME | 1/0 | Đáp án đúng; kèm lời giải tham khảo nếu dataset có |
| MBPP → HumanEval | tỷ lệ test qua | Test nào sai, thông báo lỗi hoặc traceback (cắt ngắn) |

3. **Cấu hình**: `dspy.GEPA` với minibatch 3, reflection LM là LLM ngoài ở mục 5, ngân sách `max_metric_calls` cố định cho mỗi dataset (đề xuất ~1.000; paper dùng 1.839–7.051 rollout). Lưu log mọi ứng viên, cha mẹ, điểm và prompt.
4. **A5**: trỏ task LM của DSPy tới "model name" là LoRA của A3 trên cùng server vLLM; phần còn lại giữ nguyên.
5. Đánh giá prompt tốt nhất qua harness trên tập test.

---

## 8. Ước lượng khối lượng tính toán

Số lần sinh trên tập test cho mỗi phương án: ARC-C 1.172 + HellaSwag ~1.000 + BoolQ ~1.000 + HumanEval 164 + AIME 150 (30 × 5) ≈ **3.500**. Sáu phương án bắt buộc (A0–A5, với A3 và A5 nhân 3 mô tả) ≈ **25.000–30.000 lần sinh**, cộng **~5.000 rollout tối ưu GEPA** cho 5 dataset (×2 cho A4 và A5).

Với vLLM trên một GPU 24GB, khối lượng này cỡ **vài giờ đến khoảng một ngày GPU**, tuỳ độ dài sinh (AIME dài nhất). Đây là ước lượng; **đo throughput thật ở tuần 1** rồi chỉnh kích thước test hoặc ngân sách GEPA. Chi phí API cho reflection LM dự kiến nhỏ hơn mức 86 USD của paper vì ngân sách nhỏ hơn; đặt giới hạn chi tiêu.

---

## 9. Kết quả cần ra

1. **Bảng chính**: dataset × phương án (A0–A5), accuracy ± CI.
2. **Bảng chi phí**: mỗi phương án — dữ liệu có nhãn dùng, GPU-giờ, số rollout, tiền API, token mỗi câu khi dùng.
3. **Độ nhạy với mô tả** (T2L): kết quả từng mô tả.
4. **Đường cong GEPA**: điểm validation theo số rollout, cho A4 và A5.
5. **Ví dụ định tính**: một câu AIME hoặc HumanEval qua A0, A3, A4, A5; prompt trước và sau GEPA; cây ứng viên.

Các kết quả này thay cho các ô `[__]` ở slide 5.8 và làm nội dung cho phần Đấu trường nếu dùng.

---

## 10. App demo (dựng trên harness)

Một **FastAPI server** dùng chung server vLLM ở mục 4, và **web client 3 tab**, cùng bộ 5 dataset:

| Tab | Chức năng |
|---|---|
| **LoRA Studio** (T2L) | Chọn dataset hoặc tự gõ mô tả → sinh LoRA (vài giây) → chạy một câu của dataset với model gốc và model + LoRA; xem kết quả từng mô tả |
| **Prompt Evolution** (GEPA) | Chọn dataset → xem log GEPA đã chạy: prompt qua từng vòng, phản hồi, điểm theo rollout, cây ứng viên và biên Pareto |
| **Đấu trường** | Chọn dataset và một câu → chạy A0, A3, A4, A5 song song; hiện bảng kết quả và chi phí đã tính sẵn |

Endpoint gợi ý: `/t2l/generate`, `/t2l/run`, `/gepa/runs`, `/gepa/tree`, `/arena/run`, `/arena/table`. Mọi kết quả nặng đọc từ `artifacts/`, nên app vẫn hiện được bảng khi không có GPU.

---

## 11. Cấu trúc code (cho máy sẽ code)

Xem `app/README.md`. Tóm tắt:

```
app/
  config.yaml          # model, đường dẫn checkpoint T2L, endpoint vLLM, LLM ngoài, ngân sách GEPA, seed
  data/                # loader + split cố định cho 5 dataset (+ MBPP, AIME 2022–2024)
  harness/             # prompt template, tách đáp án, chạy code sandbox, bootstrap CI
  t2l/                 # mô tả, sinh LoRA, đăng ký vLLM
  gepa_lab/            # chương trình DSPy, hàm phản hồi, chạy GEPA, đọc log cây
  experiments/         # chạy A0–A7 → artifacts/results/*.csv
  server/, client/     # app demo
  notebooks/           # notebook cho từng phần khi thuyết trình
  artifacts/           # splits, descriptions, lora, gepa logs, results
```

---

## 12. Lộ trình

| Tuần | Việc | Team chính |
|---|---|---|
| 1 | Chốt model và danh sách dataset; dựng vLLM + LoRA; loader + split; **đo throughput**; tái hiện Bảng 7 trên ARC-C, BoolQ | So sánh + App, T2L |
| 2 | Harness hoàn chỉnh; A0, A1, A2 trên 5 dataset; mô tả cho 5 dataset | So sánh + App, T2L |
| 3 | A3 (T2L); chương trình DSPy + hàm phản hồi; chạy GEPA thử ngân sách nhỏ | T2L, GEPA |
| 4 | A4 đầy đủ; A5; (mở rộng) A6, A7 | GEPA, So sánh |
| 5 | Bảng kết quả, chi phí, hình; app 3 tab; cập nhật slide 5.8 | Cả nhóm |
| 6 | Tổng duyệt, video dự phòng | Cả nhóm |

---

## 13. Cần quyết định trước khi code

- [ ] **Model chung**: Llama-3.1-8B-Instruct (đề xuất) hay Gemma-2-2B-it (nếu GPU nhỏ)?
- [ ] **AIME 2025**: giữ làm bài thử sức, và có **thêm GSM8K** không?
- [ ] **Kích thước test** của HellaSwag và BoolQ: dùng đủ validation hay 1.000 câu cố định?
- [ ] **LLM ngoài** (viết mô tả + reflection): model nào, ngân sách API bao nhiêu?
- [ ] **Ngân sách GEPA** mỗi dataset (đề xuất ~1.000 lần chấm)?
- [ ] Có làm A6 (GEPA tối ưu mô tả cho T2L) và A7 (LoRA oracle) không?

---

## 14. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| AIME gần 0 cho mọi phương án | Báo số bài đúng; thêm GSM8K; nói rõ là giới hạn của model 8B |
| HumanEval: T2L thấp hơn model gốc (đã thấy trong paper với Llama) | Báo cáo trung thực; xem từng mô tả |
| Prompt đánh giá của nhóm khác paper → số khác paper | Bước tái hiện ở tuần 1; so sánh tương đối giữa các phương án |
| Code sinh ra chạy trên máy | Sandbox, timeout, không có mạng |
| vLLM và LoRA của T2L không tương thích định dạng | Thử ngay tuần 1; dự phòng chạy bằng `transformers` + `peft` (chậm hơn) |
| Phiên bản `dspy`, `gepa`, repo T2L thay đổi | Cố định phiên bản và commit trong `requirements.txt` và `config.yaml` |
