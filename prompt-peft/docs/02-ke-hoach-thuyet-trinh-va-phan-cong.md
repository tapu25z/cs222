# Kế hoạch thuyết trình và phân công — Text-to-LoRA + GEPA

> Buổi thuyết trình **khoảng 3 tiếng**, nhóm **15 người**. Lý thuyết cơ bản (prompting, LoRA) do team khác trình bày.
> Thiết kế paper và app: xem `01-chon-paper-va-thiet-ke-app.md`.

---

## 1. Kịch bản buổi thuyết trình (180 phút)

| Thời gian | Phần | Nội dung | Người trình bày |
|---|---|---|---|
| 0:00–0:08 | **Mở đầu** | Câu hỏi xuyên suốt: *đưa kiến thức vào trọng số hay vào prompt?* Hai hướng: T2L (chữ → LoRA), GEPA (thay LoRA bằng prompt). Chạy thử app bản cuối | Trưởng nhóm |
| **0:08–1:25** | **Paper 1 — Text-to-LoRA** | | **Team T2L** |
| 0:08–0:35 | Paper | Vấn đề mỗi task một LoRA → hypernetwork là gì → kiến trúc L/M/S (Hình 2) → reconstruction vs SFT → kết quả (Bảng 1, 2) → mô tả tốt và kém (Hình 4, Bảng 5) → t-SNE (Hình 5) | 2 người |
| 0:35–1:12 | Live build | 6 bước, mục 2.1 | 4 người |
| 1:12–1:25 | Kết quả & phân tích | Bảng so sánh của nhóm trên Gemma-2-2B, thí nghiệm mô tả tiếng Việt, hạn chế | 1 người |
| 1:25–1:35 | *Nghỉ* | (đổi model trên máy demo nếu cần) | |
| **1:35–2:52** | **Paper 2 — GEPA** | | **Team GEPA** |
| 1:35–1:45 | DSPy là gì | Signature, Module, Program, Metric, Optimizer; `inspect_history()` (file 01, mục 4) | 1 người |
| 1:45–2:07 | Paper | RL cần hàng chục nghìn rollout → ý tưởng suy ngẫm bằng ngôn ngữ → vòng tiến hoá (Hình 3, Thuật toán 1) → meta-prompt (Phụ lục C) → Pareto (Hình 4, Bảng 3) → kết quả (Hình 1, Bảng 1, 2) | 2 người |
| 2:07–2:40 | Live build | 6 bước, mục 2.2 | 3 người |
| 2:40–2:52 | Kết quả & phân tích | Bảng Baseline / MIPROv2 / GEPA của nhóm, đường cong theo rollout, cây tiến hoá, prompt trước và sau | 1 người |
| 2:52–3:00 | **Kết luận + Q&A** | Slide so sánh 2 paper (mục 3), hạn chế, hướng mở rộng (Doc-to-LoRA, ACE…) | Trưởng nhóm + cả nhóm |

---

## 2. Các bước live build

### 2.1 Phần Text-to-LoRA (37 phút)

| Bước | Nội dung | Khán giả thấy |
|---|---|---|
| 1 | Load Gemma-2-2B-it; chạy vài câu GSM8K và ARC: model gốc, thêm mô tả vào prompt, 3-shot ICL | Đáp án và accuracy baseline |
| 2 | Load checkpoint T2L; in kiến trúc và số tham số; embed một mô tả bằng gte-large | Hypernetwork chỉ vài chục triệu tham số |
| 3 | **Sinh LoRA từ mô tả** → in shape A, B từng layer → lưu adapter | Mất vài giây, file vài MB |
| 4 | Gắn LoRA bằng `peft`, chạy lại câu hỏi bước 1; chạy trên tập con benchmark (load kết quả tính sẵn) | Accuracy tăng mà không train |
| 5 | Thí nghiệm mô tả: chi tiết, "solve this task please", chuỗi ngẫu nhiên, tiếng Việt | Bảng giống Bảng 5 và ví dụ giống Hình 4 |
| 6 | Tab "LoRA Studio" trên app: khán giả đề xuất một mô tả, nhóm sinh LoRA ngay | App Phần 1 |

### 2.2 Phần GEPA (33 phút)

| Bước | Nội dung | Khán giả thấy |
|---|---|---|
| 1 | Viết chương trình DSPy cho task ràng buộc (IFBench) + metric + **hàm phản hồi** trả chữ | Hàm phản hồi chỉ vài dòng |
| 2 | Chạy baseline; xem một trace thất bại và phản hồi của nó | "Ràng buộc X chưa đạt" |
| 3 | **Chạy GEPA live** với ngân sách nhỏ; in prompt trước và sau mỗi vòng suy ngẫm | Prompt tự viết lại, điểm minibatch tăng |
| 4 | Load kết quả GEPA và MIPROv2 đầy đủ (đã chạy trước, cùng ngân sách) | Bảng điểm và đường cong theo rollout |
| 5 | Vẽ cây tiến hoá và biên Pareto | Hình giống Hình 4 của paper |
| 6 | Tab "Prompt Evolution" trên app: chạy cùng câu hỏi với prompt gốc và prompt GEPA | App Phần 2 |

Nguyên tắc demo:
- Mọi kết quả nặng (bảng benchmark, lần chạy GEPA và MIPROv2 đầy đủ) **tính trước**; trên sân khấu chỉ chạy phần nhẹ.
- Có **video quay sẵn** từng bước.
- Hạ tầng: xem mục 6 của file 01.

---

## 3. Slide so sánh cuối buổi

| | Text-to-LoRA | GEPA |
|---|---|---|
| Đầu vào bằng chữ | Mô tả task (người viết) | Trace + phản hồi lỗi (môi trường sinh) |
| Thứ được thay đổi | **Trọng số** (LoRA, q_proj và v_proj) | **Prompt** (câu lệnh từng module) |
| Cần train trước | Có, một lần (hypernetwork, rất đắt) | Không |
| Chi phí cho task mới | Một lần forward (vài giây) | Vài trăm tới vài nghìn rollout |
| Cần dữ liệu task mới | Không (zero-shot) | Có, ít (vd. 150 train + 300 val) |
| Dùng được với model đóng (API) | Không (cần trọng số) | Có |
| Hạn chế chính | Chưa bằng LoRA train riêng; phụ thuộc chất lượng mô tả | Không thêm được kiến thức model chưa có; tốn rollout cho validation |

Thông điệp chốt: **"Ngôn ngữ là giao diện chung để thích nghi LLM"**, kể cả khi kết quả cuối là trọng số (T2L) hay là prompt (GEPA).

---

## 4. Phân công 15 người

| Nhóm | Số người | Trách nhiệm |
|---|---|---|
| **Trưởng nhóm / MC** | 1 | Điều phối, kịch bản, canh giờ, mở đầu và kết luận, ghép slide |
| **Team Text-to-LoRA** | 7 | |
| ↳ Paper | 2 | Đọc kỹ T2L (+ lướt Hyperdecoders, Gisting, Doc-to-LoRA), slide phần paper, câu hỏi phản biện |
| ↳ Môi trường + checkpoint | 1 | Cài code SakanaAI, tải checkpoint Gemma (và Mistral nếu đủ GPU), script sinh LoRA |
| ↳ Đánh giá | 2 | Tập con benchmark, so sánh 4 cách, thí nghiệm mô tả (gồm tiếng Việt) |
| ↳ Phân tích + mở rộng | 1 | t-SNE LoRA; (mở rộng) train vài LoRA oracle và T2L reconstruction nhỏ |
| ↳ Kết quả | 1 | Bảng, biểu đồ; trình bày phần kết quả |
| **Team GEPA** | 6 | |
| ↳ Paper + DSPy | 2 | Đọc kỹ GEPA và DSPy; 1 người trình bày "DSPy là gì", cả hai trình bày phần paper |
| ↳ Chương trình + hàm phản hồi | 2 | Chương trình DSPy cho IFBench và HotpotQA, metric, μ_f, retriever BM25 |
| ↳ Chạy optimizer + kết quả | 2 | Chạy GEPA và MIPROv2 cùng ngân sách, log, cây tiến hoá, đường cong; trình bày kết quả |
| **App / hạ tầng** | 1 | FastAPI, web client, máy GPU demo, vLLM, API key, video dự phòng |

Tổng: 1 + 7 + 6 + 1 = **15 người**. Người phụ trách app nên được 1 người ở mỗi team hỗ trợ ghép endpoint.

---

## 5. Lộ trình chuẩn bị (6 tuần, điều chỉnh theo ngày thuyết trình)

| Tuần | Team T2L | Team GEPA | App |
|---|---|---|---|
| 1 | Đọc paper; cài code SakanaAI; sinh được LoRA đầu tiên với Gemma-2-2B | Đọc paper + DSPy; `pip install gepa dspy`; chạy ví dụ AIME trong README của GEPA | Khung `app/`, `config.yaml`; dựng vLLM |
| 2 | Baseline (gốc, mô tả, ICL) trên tập con benchmark; **đo VRAM** | Chương trình IFBench + hàm phản hồi; baseline | Endpoint `/t2l/generate` |
| 3 | T2L trên mọi benchmark con; thí nghiệm mô tả | GEPA trên IFBench (ngân sách nhỏ → vừa) | `/t2l/chat` (đổi adapter nóng) |
| 4 | t-SNE; (mở rộng) LoRA oracle, T2L reconstruction | HotpotQA; MIPROv2 cùng ngân sách; log cây | `/gepa/*`, vẽ cây |
| 5 | Slide + kết quả | Slide + kết quả | Client hoàn chỉnh, **tổng duyệt 1** |
| 6 | Sửa theo góp ý | Sửa theo góp ý | Video dự phòng, **tổng duyệt 2** trên đúng máy demo |

Mốc go/no-go:
- **Cuối tuần 1**: chưa chạy được code T2L → dùng đúng môi trường `uv` của repo gốc, hoặc chạy trên Colab/Kaggle.
- **Cuối tuần 3**: GEPA quá chậm hoặc tốn → giảm ngân sách, dùng Qwen3-4B, chỉ giữ IFBench.
- **Cuối tuần 5**: app chạy trọn từ đầu đến cuối; không thêm tính năng mới.

---

## 6. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Kết quả T2L trên Gemma thấp hơn paper | Báo cáo trung thực; dùng đúng prompt đánh giá của repo (`run_eval.py`); so sánh tương đối |
| Mô tả tiếng Việt cho LoRA kém | Đây là kết quả thí nghiệm hợp lệ (T2L train bằng tiếng Anh); dùng làm điểm thảo luận |
| GEPA tốn tiền API | Đặt giới hạn chi tiêu; ngân sách nhỏ; dùng task LM local, chỉ reflection qua API |
| Không tái hiện được GRPO | Nói rõ là trích số của paper; GRPO cần 24.000 rollout và GPU 80GB |
| Phiên bản `gepa`, `dspy`, code T2L thay đổi | Cố định phiên bản trong `requirements.txt`; ghi commit của repo SakanaAI đã dùng |
| GPU, mạng hỏng khi thuyết trình | Client hiển thị kết quả tính sẵn không cần GPU; video dự phòng |
