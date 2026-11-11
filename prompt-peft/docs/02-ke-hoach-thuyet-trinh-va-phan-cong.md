# Kế hoạch thuyết trình và phân công — Text-to-LoRA + GEPA

> Buổi thuyết trình **khoảng 3 tiếng**. Nhóm **15 người**, nhưng **chỉ 3 người thuyết trình**; 12 người còn lại chuẩn bị nội dung, code, kết quả và hỗ trợ kỹ thuật phía sau.
> Lý thuyết cơ bản (prompting, LoRA) do team khác trình bày. Thiết kế paper và app: xem `01-chon-paper-va-thiet-ke-app.md`.

---

## 1. Ba người thuyết trình

| Người | Vai | Phụ trách nói | Tổng thời gian nói |
|---|---|---|---|
| **Người 1** | Chuyên gia Text-to-LoRA | Paper T2L + live build T2L | khoảng 55 phút |
| **Người 2** | Chuyên gia GEPA | DSPy + paper GEPA + live build GEPA | khoảng 60 phút |
| **Người 3** | MC, kết quả, so sánh | Mở đầu, kết quả của cả hai phần, Đấu trường, kết luận, điều phối Q&A | khoảng 55 phút |

Người 3 nói xen giữa các phần của Người 1 và Người 2, nên không ai phải nói liền hơn khoảng 30 phút. Người 1 và Người 2 phải hiểu phần của nhau đủ để trả lời Q&A.

---

## 2. Kịch bản 180 phút

| Thời gian | Phút | Phần | Nội dung | Người nói |
|---|---|---|---|---|
| 0:00–0:08 | 8 | **Mở đầu** | Câu hỏi xuyên suốt: *đưa kiến thức vào trọng số hay vào prompt?* Hai hướng: T2L (chữ → LoRA), GEPA (thay LoRA bằng prompt). Chạy thử app bản cuối | Người 3 |
| 0:08–0:33 | 25 | **Paper T2L** | Vấn đề mỗi task một LoRA → hypernetwork → kiến trúc L/M/S (Hình 2) → reconstruction vs SFT → kết quả (Bảng 1, 2) → mô tả tốt và kém (Hình 4, Bảng 5) → t-SNE (Hình 5) | Người 1 |
| 0:33–1:03 | 30 | **Live build T2L** | 6 bước, mục 3.1 | Người 1 |
| 1:03–1:13 | 10 | **Kết quả T2L** | Bảng so sánh của nhóm trên model gốc, thí nghiệm mô tả (gồm tiếng Việt), hạn chế | Người 3 |
| 1:13–1:23 | 10 | *Nghỉ* | Đổi model trên máy demo nếu cần | |
| 1:23–1:33 | 10 | **DSPy là gì** | Signature, Module, Program, Metric, Optimizer; `inspect_history()` (file 01, mục 4) | Người 2 |
| 1:33–1:55 | 22 | **Paper GEPA** | RL cần hàng chục nghìn rollout → suy ngẫm bằng ngôn ngữ → vòng tiến hoá (Hình 3, Thuật toán 1) → meta-prompt (Phụ lục C) → Pareto (Hình 4, Bảng 3) → kết quả (Hình 1, Bảng 1, 2) | Người 2 |
| 1:55–2:23 | 28 | **Live build GEPA** | 6 bước, mục 3.2 | Người 2 |
| 2:23–2:33 | 10 | **Kết quả GEPA** | Bảng Baseline / MIPROv2 / GEPA của nhóm, đường cong theo rollout, cây tiến hoá, prompt trước và sau | Người 3 |
| 2:33–2:47 | 14 | **Đấu trường** | So sánh T2L và GEPA trên cùng bài toán (file 01, mục 7): bảng 7 phương án, chạy live một câu GSM8K, bảng chi phí | Người 3 (Người 1, 2 bấm demo phần của mình) |
| 2:47–3:00 | 13 | **Kết luận + Q&A** | Slide "khi nào dùng cách nào" (mục 4), hạn chế, hướng mở rộng (Doc-to-LoRA, ACE…); Q&A | Người 3 điều phối, cả 3 trả lời |

Tổng: 8 + 25 + 30 + 10 + 10 + 10 + 22 + 28 + 10 + 14 + 13 = **180 phút**.

**Dự phòng thời gian**: nếu trễ, rút phần live build (dùng kết quả tính sẵn thay cho chạy thật) trước, giữ nguyên Đấu trường và Kết luận.

---

## 3. Các bước live build

### 3.1 Live build Text-to-LoRA (30 phút, Người 1)

| Bước | Nội dung | Khán giả thấy |
|---|---|---|
| 1 | Load model gốc; chạy vài câu: gốc, thêm mô tả vào prompt, 3-shot ICL | Đáp án baseline |
| 2 | Load checkpoint T2L; in kiến trúc và số tham số; embed một mô tả bằng gte-large | Hypernetwork chỉ vài chục triệu tham số |
| 3 | **Sinh LoRA từ mô tả** → in shape A, B từng layer → lưu adapter | Mất vài giây, file vài MB |
| 4 | Gắn LoRA bằng `peft`, chạy lại câu hỏi bước 1; chiếu kết quả trên tập con benchmark (tính sẵn) | Kết quả tăng mà không train |
| 5 | Thí nghiệm mô tả: chi tiết, "solve this task please", chuỗi ngẫu nhiên, tiếng Việt | Bảng giống Bảng 5, ví dụ giống Hình 4 |
| 6 | Tab "LoRA Studio": khán giả đề xuất một mô tả, sinh LoRA ngay | App tab 1 |

### 3.2 Live build GEPA (28 phút, Người 2)

| Bước | Nội dung | Khán giả thấy |
|---|---|---|
| 1 | Chương trình DSPy cho task ràng buộc (IFBench) + metric + **hàm phản hồi** trả chữ | Hàm phản hồi chỉ vài dòng |
| 2 | Chạy baseline; xem một trace thất bại và phản hồi của nó | "Ràng buộc X chưa đạt" |
| 3 | **Chạy GEPA live** với ngân sách nhỏ; in prompt trước và sau mỗi vòng suy ngẫm | Prompt tự viết lại, điểm minibatch tăng |
| 4 | Load kết quả GEPA và MIPROv2 đầy đủ (chạy trước, cùng ngân sách) | Bảng điểm, đường cong theo rollout |
| 5 | Vẽ cây tiến hoá và biên Pareto | Hình giống Hình 4 của paper |
| 6 | Tab "Prompt Evolution": cùng câu hỏi với prompt gốc và prompt GEPA | App tab 2 |

Nguyên tắc demo:
- Mọi kết quả nặng (bảng benchmark, lần chạy GEPA/MIPROv2 đầy đủ, bảng Đấu trường) **tính trước**; trên sân khấu chỉ chạy phần nhẹ.
- Mỗi người thuyết trình có **một người hỗ trợ kỹ thuật ngồi cạnh máy** (không nói), sẵn sàng chuyển sang kết quả tính sẵn hoặc video nếu demo lỗi.
- Có **video quay sẵn** từng bước. Hạ tầng: xem file 01, mục 6.

---

## 4. Slide so sánh cuối buổi

| | Text-to-LoRA | GEPA |
|---|---|---|
| Đầu vào bằng chữ | Mô tả task (người viết) | Trace + phản hồi lỗi (môi trường sinh) |
| Thứ được thay đổi | **Trọng số** (LoRA, q_proj và v_proj) | **Prompt** (câu lệnh từng module) |
| Cần train trước | Có, một lần (hypernetwork, rất đắt) | Không |
| Chi phí cho task mới | Một lần forward (vài giây) | Vài trăm tới vài nghìn rollout |
| Cần dữ liệu task mới | Không (zero-shot) | Có, ít (vd. 150 train + 300 val) |
| Dùng được với model đóng (API) | Không (cần trọng số) | Có |
| Hạn chế chính | Chưa bằng LoRA train riêng; phụ thuộc chất lượng mô tả | Không thêm được kiến thức model chưa có; tốn rollout cho validation |

Thông điệp chốt: **"Ngôn ngữ là giao diện chung để thích nghi LLM"**, dù kết quả cuối là trọng số (T2L) hay prompt (GEPA), và hai cách có thể **kết hợp** (phương án 6 trong Đấu trường).

---

## 5. Phân công 15 người

Ba team, mỗi team 5 người, mỗi team có đúng **1 người thuyết trình**.

| Team | Người | Trách nhiệm |
|---|---|---|
| **Team T2L** (5) | **Người 1 — thuyết trình** | Đọc kỹ T2L; slide phần paper; nói paper và live build T2L |
| | 1 người | Môi trường: cài code SakanaAI, tải checkpoint, script sinh LoRA; **hỗ trợ kỹ thuật cạnh máy** khi Người 1 demo |
| | 2 người | Đánh giá: tập con benchmark, so sánh gốc / mô tả / ICL / T2L, thí nghiệm mô tả (gồm tiếng Việt) |
| | 1 người | t-SNE LoRA; (mở rộng) LoRA oracle và T2L reconstruction nhỏ; soạn câu hỏi phản biện cho Người 1 tập trả lời |
| **Team GEPA** (5) | **Người 2 — thuyết trình** | Đọc kỹ GEPA và DSPy; slide phần DSPy và paper; nói DSPy, paper và live build GEPA |
| | 2 người | Chương trình DSPy (IFBench, HotpotQA), metric, hàm phản hồi, retriever BM25 |
| | 1 người | Chạy GEPA và MIPROv2 cùng ngân sách, log, cây tiến hoá, đường cong; **hỗ trợ kỹ thuật cạnh máy** khi Người 2 demo |
| | 1 người | Quản lý API key và chi phí; soạn câu hỏi phản biện cho Người 2 |
| **Team So sánh + App** (5) | **Người 3 — thuyết trình** | MC; slide mở đầu, kết quả, Đấu trường, kết luận; canh giờ; điều phối Q&A |
| | 2 người | Thí nghiệm Đấu trường: chạy 7 phương án trên cùng model và tập test, bảng chi phí |
| | 1 người | App: FastAPI, web client 3 tab, máy GPU demo, vLLM |
| | 1 người | Ghép slide tổng, thống nhất hình vẽ, quay video dự phòng, kịch bản chuyển phần |

Tổng: 5 + 5 + 5 = **15 người**.

---

## 6. Lộ trình chuẩn bị (6 tuần, điều chỉnh theo ngày thuyết trình)

| Tuần | Team T2L | Team GEPA | Team So sánh + App | 3 người thuyết trình |
|---|---|---|---|---|
| 1 | Cài code SakanaAI; sinh LoRA đầu tiên; **đo VRAM** | `pip install gepa dspy`; chạy ví dụ AIME trong README | Chốt model gốc chung (Mistral-7B hay Gemma-2-2B) theo GPU; khung `app/`; dựng vLLM | Đọc paper của mình |
| 2 | Baseline (gốc, mô tả, ICL) trên tập con benchmark | Chương trình IFBench + hàm phản hồi; baseline | Chương trình GSM8K chung cho Đấu trường; `/t2l/generate` | Bản nháp slide |
| 3 | T2L trên mọi benchmark con; thí nghiệm mô tả | GEPA trên IFBench | Đấu trường: phương án 1–4 | Tập nói phần paper (bấm giờ) |
| 4 | t-SNE; (mở rộng) LoRA oracle | HotpotQA; MIPROv2 cùng ngân sách; log cây | Đấu trường: phương án 5–6 (và 7 nếu kịp); `/gepa/*`, `/arena/*` | Tập live build với người hỗ trợ kỹ thuật |
| 5 | Chốt kết quả | Chốt kết quả | Client 3 tab hoàn chỉnh; **tổng duyệt 1** (đủ 180 phút) | Sửa slide theo góp ý |
| 6 | Hỗ trợ Q&A | Hỗ trợ Q&A | Video dự phòng; **tổng duyệt 2** trên đúng máy demo | Tập Q&A với câu hỏi phản biện |

Mốc go/no-go:
- **Cuối tuần 1**: chọn model gốc chung. Có GPU khoảng 24GB thì dùng Mistral-7B; không thì Gemma-2-2B và thêm PIQA/HellaSwag vào Đấu trường.
- **Cuối tuần 3**: GEPA quá chậm hoặc tốn → giảm ngân sách, chỉ giữ IFBench và GSM8K.
- **Cuối tuần 5**: app chạy trọn từ đầu đến cuối; không thêm tính năng mới.

---

## 7. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Một người thuyết trình vắng | Mỗi người thuyết trình có một người cùng team đã tập phần đó làm **dự bị** (người soạn câu hỏi phản biện) |
| Người thuyết trình nói quá giờ | Người 3 canh giờ; mỗi live build có điểm cắt sẵn chuyển sang kết quả tính sẵn |
| Kết quả T2L trên model gốc thấp hơn paper | Báo cáo trung thực; dùng đúng script đánh giá của repo; so sánh tương đối |
| Mô tả tiếng Việt cho LoRA kém | Là kết quả thí nghiệm hợp lệ (T2L train bằng tiếng Anh); dùng làm điểm thảo luận |
| GEPA tốn tiền API | Đặt giới hạn chi tiêu; ngân sách nhỏ; task LM chạy local |
| Không tái hiện được GRPO | Nói rõ là trích số của paper (24.000 rollout, GPU 80GB) |
| Phiên bản `gepa`, `dspy`, code T2L thay đổi | Cố định phiên bản; ghi commit của repo SakanaAI đã dùng |
| GPU, mạng hỏng khi thuyết trình | Client hiển thị kết quả tính sẵn không cần GPU; video dự phòng |
