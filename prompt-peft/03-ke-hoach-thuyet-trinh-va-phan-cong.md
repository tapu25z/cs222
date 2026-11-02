# Kế hoạch thuyết trình và phân công — Prompting + PEFT nâng cao (LoPA)

> Bối cảnh: buổi thuyết trình **khoảng 3 tiếng**, nhóm **15 người**. Paper chính: **LoPA (NeurIPS 2024)**.
> App: **"One Model, Many Skills"** — FastAPI + 3 skill (SST-2 EN, UIT-VSFC VI, CruxEval-O code). Xem `02-chon-paper-va-thiet-ke-app.md`.

---

## 1. Kịch bản buổi thuyết trình (180 phút)

| Thời gian | Phần | Nội dung | Người trình bày | Demo |
|---|---|---|---|---|
| 0:00–0:10 | **Mở đầu** | Bài toán: một LLM, nhiều người dùng, nhiều task. Giới thiệu app hoàn chỉnh (chạy thử 30 giây) | MC / Trưởng nhóm | App bản cuối |
| 0:10–0:40 | **A. Prompting** | Zero-shot → Few-shot (GPT-3) → Chain-of-Thought. Hạn chế của prompting | Nhóm Lý thuyết (2 người) | Notebook: cả 3 kiểu prompt trên cả 3 skill |
| 0:40–1:10 | **B. PEFT** | Vì sao cần PEFT, Adapter, LoRA (tính số tham số), Prompt Tuning, bảng so sánh | Nhóm Lý thuyết (2 người) | Đếm số tham số train thực tế bằng code |
| 1:10–1:45 | **C. Paper LoPA** | Động lực (Fig. 1) → công thức `Z = Z_S ⊙ σ(u·vᵀ)` (Fig. 2) → vì sao tốt hơn PT/IDPG → kết quả chính (Table 1, Fig. 4) | Nhóm Paper (3 người) | — |
| 1:45–1:55 | *Nghỉ* | | | |
| 1:55–2:40 | **D. Live build** | Build app từng bước (mục 2). Mỗi người trình bày một bước khoảng 6–7 phút | Nhóm Data + Model + App (khoảng 7 người) | Notebook + FastAPI |
| 2:40–2:52 | **E. Kết quả & phân tích** | Bảng 5 phương pháp × 3 skill, ablation rank r, chi phí (tham số, dung lượng, VRAM), lỗi thường gặp | Nhóm Model (1–2 người) | Biểu đồ |
| 2:52–3:00 | **F. Kết luận + Q&A** | Hạn chế của LoPA, hướng mở rộng (POST: privacy, PARA), Q&A | Trưởng nhóm + cả nhóm | — |

Thời gian dự phòng: phần D đã tính dư khoảng 5 phút cho sự cố. Nếu bị trễ thì cắt phần ablation ở E.

---

## 2. Các bước live build (phần D)

| Bước | Nội dung | Kết quả cho khán giả thấy | Phụ trách |
|---|---|---|---|
| 1 | Dữ liệu: load SST-2, UIT-VSFC, CruxEval-O; template prompt chung | Vài mẫu dữ liệu, cách chia train/test | Data |
| 2 | Baseline prompting (zero/few-shot, CoT cho code) trên backbone freeze | Accuracy / exact match baseline | Data |
| 3 | Prompt Tuning (m = 10) | Loss giảm; số tham số khoảng 15K | Model |
| 4 | LoRA với thư viện `peft` | Accuracy; adapter phải gắn vào model | Model |
| 5 | **LoPA**: viết `lopa.py` (khoảng 50 dòng), train ngắn rồi load checkpoint | Accuracy ngang LoRA; file skill vài MB | Model |
| 6 | FastAPI server: một LLM, endpoint `/generate` nhận soft prompt | Đổi skill **không reload model**; đo độ trễ | App |
| 7 | Web client: chọn skill, nhập câu, bảng so sánh 5 phương pháp | App hoàn chỉnh | App |

Nguyên tắc demo an toàn:
- Mọi checkpoint đều **train sẵn** trước buổi thuyết trình; trên sân khấu chỉ train khoảng 50–100 bước để thấy loss giảm.
- Có **video quay sẵn** từng bước, phòng khi mất mạng hoặc GPU.
- Chạy chính trên GPU riêng; Colab/Kaggle làm máy dự phòng (backbone 0.5B).

---

## 3. Phân công 15 người

| Nhóm | Số người | Trách nhiệm | Sản phẩm |
|---|---|---|---|
| **Trưởng nhóm / MC** | 1 | Điều phối, ghép slide, tổng duyệt, canh giờ, phần mở đầu và kết luận | Kịch bản, slide tổng |
| **Lý thuyết** | 4 | 2 người phần Prompting, 2 người phần PEFT; notebook minh hoạ nhỏ | Slide phần A, B, notebook minh hoạ |
| **Paper** | 3 | Đọc kỹ LoPA (và lướt PT, IDPG, LoRA), giải thích công thức, vẽ lại hình, chuẩn bị câu hỏi phản biện | Slide phần C, file tóm tắt paper |
| **Data** | 2 | Tải, tiền xử lý, chia 3 dataset; template prompt; script đánh giá chung | `data/`, `evaluate.py` |
| **Model** | 3 | Mỗi người phụ trách một skill: train PT, LoRA, LoPA; ablation rank; bảng kết quả | `lopa.py`, `train.py`, checkpoint, bảng/biểu đồ |
| **App** | 2 | Một người làm FastAPI server, một người làm web client; đóng gói skill; notebook demo | `server/`, `client/`, `notebooks/demo.ipynb` |

Tổng cộng: 1 + 4 + 3 + 2 + 3 + 2 = **15 người**.
Mỗi nhóm cử một người trình bày phần của mình. Các nhóm Data, Model, App cùng trình bày phần D.

---

## 4. Lộ trình chuẩn bị (6 tuần, điều chỉnh theo ngày thuyết trình)

| Tuần | Mốc | Nhóm chính |
|---|---|---|
| 1 | Cả nhóm đọc `01-...md` và slide CS224N Lecture 9. Nhóm Paper đọc LoPA. Chốt backbone, tạo repo `app/` khung | Tất cả |
| 2 | Dữ liệu sẵn sàng, chạy được baseline prompting trên 3 skill. Có bản nháp slide A, B | Data, Lý thuyết |
| 3 | Train được PT và LoRA cho cả 3 skill. Viết xong `lopa.py`, chạy được trên SST-2 | Model |
| 4 | LoPA cho cả 3 skill, ablation rank. FastAPI server chạy được với soft prompt | Model, App |
| 5 | Client, bảng so sánh, notebook demo. Slide hoàn chỉnh. **Tổng duyệt lần 1** (bấm giờ) | App, tất cả |
| 6 | Sửa theo góp ý, quay video dự phòng. **Tổng duyệt lần 2** trên đúng máy sẽ dùng | Tất cả |

Mốc kiểm tra (go/no-go):
- **Cuối tuần 3**: nếu LoPA chưa train được thì giảm backbone xuống 0.5B, hoặc bỏ CruxEval-O và chỉ giữ 2 skill phân loại.
- **Cuối tuần 5**: app phải chạy trọn từ đầu đến cuối; từ đây không thêm tính năng mới.

---

## 5. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| LoPA không đạt được kết quả như paper với Qwen | Báo cáo trung thực; so sánh tương đối với PT/LoRA trên cùng setup; đối chiếu với code chính thức của paper |
| CruxEval-O quá khó với model 1.5B (điểm rất thấp) | Dùng làm ví dụ minh hoạ hạn chế; vẫn so sánh tương đối giữa các phương pháp |
| Truyền soft prompt qua API (tensor m × d) | Gửi dạng float16 base64; hoặc gửi id skill và để service client sinh Z (ghi rõ lựa chọn này trong slide) |
| GPU, mạng hỏng khi thuyết trình | Video quay sẵn, máy dự phòng Colab, checkpoint đã tải về máy |
| 15 người, nội dung bị trùng lặp | Trưởng nhóm giữ một file kịch bản chung; mỗi phần có danh sách "đã nói ở phần trước" |
