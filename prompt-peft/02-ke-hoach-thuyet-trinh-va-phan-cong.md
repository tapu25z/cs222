# Kế hoạch thuyết trình và phân công — LoPA + BetterTogether

> Buổi thuyết trình **khoảng 3 tiếng**, nhóm **15 người**. Lý thuyết cơ bản do team khác trình bày.
> Thiết kế paper và app: xem `01-chon-paper-va-thiet-ke-app.md`.

---

## 1. Kịch bản buổi thuyết trình (180 phút)

| Thời gian | Phần | Nội dung | Người trình bày |
|---|---|---|---|
| 0:00–0:08 | **Mở đầu** | Đề tài, vì sao chọn 2 paper (soft prompt vs. prompt chữ + LoRA), chạy thử app bản cuối | Trưởng nhóm |
| **0:08–1:28** | **Paper 1 — LoPA** | | **Team LoPA** |
| 0:08–0:35 | Paper | Vấn đề (Fig. 1) → công thức `Z_S ⊙ σ(u·vᵀ)` (Fig. 2) → so với PT, IDPG, LoRA → kết quả (Table 1, Fig. 4) | 2 người |
| 0:35–1:15 | Live build | 6 bước, mục 2.1 | 4 người |
| 1:15–1:28 | Kết quả | Bảng 5 phương pháp × 3 skill, ablation rank r, chi phí | 1 người |
| 1:28–1:38 | *Nghỉ* | | |
| **1:38–2:50** | **Paper 2 — BetterTogether** | | **Team BetterTogether** |
| 1:38–2:03 | Paper | LM program và DSPy, vì sao không có nhãn trung gian, bootstrap trace, xen kẽ prompt ↔ LoRA, kết quả | 2 người |
| 2:03–2:38 | Live build | 6 bước, mục 2.2 | 4 người |
| 2:38–2:50 | Kết quả | Bảng 4 cấu hình × 2 skill, xem trace trước/sau tối ưu | 1 người |
| 2:50–3:00 | **Kết luận + Q&A** | So sánh 2 paper (mục 3), hạn chế, hướng mở rộng | Trưởng nhóm + cả nhóm |

---

## 2. Các bước live build

### 2.1 Phần LoPA (40 phút)
| Bước | Nội dung | Khán giả thấy |
|---|---|---|
| 1 | Dữ liệu SST-2, UIT-VSFC, CruxEval-O; baseline zero/few-shot trên backbone freeze | Accuracy baseline |
| 2 | Prompt Tuning (m = 10) | Loss giảm, khoảng 15K tham số |
| 3 | LoRA bằng `peft` | Accuracy cao, nhưng adapter phải gắn vào model |
| 4 | **LoPA**: viết `lopa.py` (khoảng 50 dòng), train ngắn, load checkpoint | Accuracy ngang LoRA, file skill vài MB |
| 5 | Endpoint `/lopa/predict`: đổi skill không reload model | Đo độ trễ |
| 6 | Web client: chọn skill + bảng so sánh | App Phần 1 |

### 2.2 Phần BetterTogether (35 phút)
| Bước | Nội dung | Khán giả thấy |
|---|---|---|
| 1 | Viết DSPy program: CoT cho GSM8K, 2-hop cho HotPotQA; chạy zero-shot | Trace thô, accuracy baseline |
| 2 | Tối ưu prompt (`BootstrapFewShotWithRandomSearch`) — chạy thu nhỏ, load kết quả đầy đủ | Prompt sau tối ưu (các ví dụ được chọn) |
| 3 | Thu trace đúng → dữ liệu fine-tune | Vài mẫu dữ liệu tự sinh |
| 4 | Fine-tune LoRA (`BootstrapFinetune`, QLoRA) — load adapter đã train | Accuracy "chỉ LoRA" |
| 5 | Tối ưu prompt lần nữa trên model đã fine-tune → **BetterTogether** | Accuracy cao nhất |
| 6 | Endpoint `/bt/run` + tab mới trên client, xem trace | App Phần 2 |

Nguyên tắc demo: mọi thứ train sẵn; trên sân khấu chỉ chạy bản thu nhỏ; có video quay sẵn. Hạ tầng: xem mục 5 của file 01.

---

## 3. Slide so sánh cuối buổi

| | LoPA | BetterTogether |
|---|---|---|
| Prompt | Soft (vector), sinh theo từng input | Hard (chữ + few-shot), tối ưu tự động |
| PEFT | Không sửa model, chỉ thêm prefix | LoRA trên trọng số |
| Cần nhãn | Nhãn cuối, train có gradient | Chỉ nhãn cuối, không cần gradient qua pipeline |
| Hợp với | Một bước, nhiều task trên một server | Pipeline nhiều bước, suy luận |
| Chi phí server | Không có adapter | Có adapter LoRA (hoặc merge) |

---

## 4. Phân công 15 người

| Nhóm | Số người | Trách nhiệm |
|---|---|---|
| **Trưởng nhóm / MC** | 1 | Điều phối, kịch bản, canh giờ, mở đầu và kết luận, ghép slide |
| **Team LoPA** | 7 | |
| ↳ Paper | 2 | Đọc kỹ LoPA (và lướt PT, IDPG, DePT), slide phần paper, chuẩn bị câu hỏi phản biện |
| ↳ Data + baseline | 1 | 3 dataset, template prompt, `evaluate.py` |
| ↳ Train | 3 | Mỗi người một skill: PT, LoRA, LoPA; ablation rank |
| ↳ Kết quả | 1 | Bảng, biểu đồ; trình bày phần kết quả |
| **Team BetterTogether** | 6 | |
| ↳ Paper | 2 | Đọc kỹ BetterTogether + DSPy (ICLR 2024), slide phần paper |
| ↳ DSPy program + tối ưu prompt | 2 | `programs.py`, retriever BM25, `optimize.py` (phần prompt) |
| ↳ Fine-tune + kết quả | 2 | QLoRA, phục vụ adapter bằng vLLM, bảng kết quả, trace |
| **App / hạ tầng** | 1 | FastAPI server, web client, máy GPU demo, bản dự phòng CPU, video |

Tổng: 1 + 7 + 6 + 1 = **15 người**. Người phụ trách app nên được 1 người ở mỗi team hỗ trợ ghép endpoint.

---

## 5. Lộ trình chuẩn bị (6 tuần, điều chỉnh theo ngày thuyết trình)

| Tuần | Team LoPA | Team BetterTogether | App |
|---|---|---|---|
| 1 | Đọc paper, chốt backbone, dữ liệu | Đọc paper + tutorial DSPy, tải PDF vào repo | Khung `app/`, `config.yaml` |
| 2 | Baseline + PT | DSPy program chạy được với model nhỏ, baseline | Server khung |
| 3 | LoRA + `lopa.py` trên SST-2 | Tối ưu prompt đầy đủ trên GPU riêng | `/lopa/predict` |
| 4 | LoPA cho 3 skill, ablation | Fine-tune LoRA + vòng prompt thứ 2 | `/bt/run`, vLLM + LoRA |
| 5 | Slide + kết quả | Slide + kết quả | Client hoàn chỉnh, **tổng duyệt 1** |
| 6 | Sửa theo góp ý | Sửa theo góp ý | Dự phòng CPU/GGUF, video, **tổng duyệt 2** trên đúng máy demo |

Mốc go/no-go:
- **Cuối tuần 3**: LoPA chưa chạy được → giảm xuống 0.5B hoặc bỏ CruxEval-O. Tối ưu prompt quá chậm → giảm tập train/dev, chỉ giữ GSM8K.
- **Cuối tuần 4**: LoRA 7–8B không vừa GPU → chuyển BetterTogether sang Qwen2.5-3B.
- **Cuối tuần 5**: app chạy trọn từ đầu đến cuối; không thêm tính năng mới.

---

## 6. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Kết quả không giống paper | Báo cáo trung thực; so sánh tương đối giữa các phương pháp trên cùng setup; ghi rõ chỗ khác paper (model, retriever) |
| DSPy tối ưu tốn nhiều lần gọi LLM | Chạy trên vLLM local (không tốn tiền API); giảm số candidate; cache kết quả |
| Phiên bản DSPy thay đổi API | Cố định phiên bản trong `requirements.txt` |
| GPU, mạng hỏng khi thuyết trình | Dự phòng CPU + GGUF, kết quả tính sẵn, video |
| Hai team trình bày trùng nội dung | Trưởng nhóm giữ kịch bản chung; mở đầu nói luôn khung so sánh để hai phần nối nhau |
