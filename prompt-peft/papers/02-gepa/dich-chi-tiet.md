# GEPA — Dịch và diễn giải chi tiết

> Bản dịch có diễn giải của *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* (Agrawal et al., ICLR 2026 Oral). PDF (bản ICLR): [`gepa-iclr2026.pdf`](gepa-iclr2026.pdf).
> Ô **💡 Giải thích** là phần nhóm thêm vào. Số liệu lấy từ **bản ICLR** (khác bản arXiv v2 một chút, vd. "up to 19pp" thay vì "up to 20%"). PDF dài 87 trang; phần phụ lục chỉ dịch các mục quan trọng.

---

## Abstract

LLM ngày càng được thích nghi cho task cụ thể bằng **học tăng cường (RL)** như **GRPO (Group Relative Policy Optimization)**, thường cần **hàng nghìn rollout** để học task mới. Tác giả lập luận rằng bản chất **dễ diễn giải của ngôn ngữ** thường là **môi trường học giàu thông tin hơn nhiều** cho LLM, so với policy gradient lấy từ phần thưởng vô hướng, thưa.

Để kiểm chứng, tác giả giới thiệu **GEPA (Genetic-Pareto)**, một **bộ tối ưu prompt** khai thác triệt để **suy ngẫm bằng ngôn ngữ tự nhiên** để học các quy tắc cấp cao từ thử–sai. Với bất kỳ hệ AI nào chứa một hoặc nhiều prompt LLM, GEPA:
- lấy mẫu các **quỹ đạo** (suy luận, lời gọi công cụ, output của công cụ);
- **suy ngẫm** về chúng bằng ngôn ngữ để **chẩn đoán vấn đề**, **đề xuất và thử** cập nhật prompt;
- **kết hợp các bài học bổ trợ nhau** từ **biên Pareto** của chính các lần thử.

Nhờ thiết kế này, GEPA thường biến **chỉ vài rollout** thành **tăng chất lượng lớn**.
- Trên 6 task, GEPA **hơn GRPO trung bình 6 điểm phần trăm, tối đa 19 điểm**, trong khi dùng **ít rollout hơn tới 35 lần**.
- GEPA cũng **hơn MIPROv2** (bộ tối ưu prompt hàng đầu) **hơn 10 điểm** (vd. +12 trên AIME-2025).
- GEPA cho kết quả hứa hẹn khi dùng làm **chiến lược tìm kiếm lúc suy luận** để tối ưu code.

Code: https://github.com/gepa-ai/gepa.

> **Hình 1.** So sánh quá trình học của GEPA với MIPROv2 và GRPO (24.000 rollout) trên (a) HotpotQA và (b) IFBench với Qwen3-8B. Khi số rollout tăng, các bộ tối ưu prompt học **nhanh hơn nhiều** so với GRPO. GEPA vượt hẳn cả GRPO lẫn MIPROv2 ở điểm cuối. Dấu sao là điểm trên tập test giữ riêng.

---

## 1. Introduction (Giới thiệu)

LLM cho phép xây agent và hệ thống kết hợp đặc tả ngôn ngữ "mờ" với công cụ như truy hồi và chạy code. Câu hỏi đặt ra: nên **tối ưu** LLM cho hiệu năng trên task như thế nào?

Một cách phổ biến là **RL với phần thưởng kiểm chứng được (RLVR)**, ví dụ **GRPO**, coi thước đo thành công là **phần thưởng vô hướng ở cuối rollout** để ước lượng policy gradient. Hiệu quả nhưng trong thực tế thường cần **hàng chục nghìn rollout** để khớp task mới; các công trình gần đây dùng GRPO thường cần tới hàng trăm nghìn rollout để train. Sự kém hiệu quả về mẫu này nhanh chóng thành **nút thắt**: nhiều ứng dụng gọi công cụ đắt tiền, có ngân sách suy luận hạn chế, hoặc **không thể fine-tune trọng số** của LLM lớn nhất hay tốt nhất.

Tác giả nhận thấy: rollout từ cả những hệ LLM phức tạp nhất cũng có thể **viết ra thành văn bản**. Chúng chỉ gồm câu lệnh của từng module LLM, chuỗi suy luận, lời gọi công cụ, và có thể cả hoạt động bên trong hàm thưởng (vd. thông báo lỗi biên dịch trước khi bị nén thành điểm). Vì LLM hiện đại đọc hiểu các quỹ đạo này dễ dàng, tác giả lập luận rằng **thuật toán học có chủ đích bằng ngôn ngữ, bằng cách suy ngẫm về quỹ đạo**, tận dụng được **tiên nghiệm ngôn ngữ mạnh** của LLM tốt hơn nhiều so với RL thông thường.

**GEPA** là một bộ tối ưu prompt **dựa trên suy ngẫm** cho **hệ AI ghép (compound AI system)**, kết hợp suy ngẫm bằng chữ với **tìm kiếm tiến hoá đa mục tiêu**.
- GEPA **đột biến prompt lặp đi lặp lại** bằng phản hồi ngôn ngữ từ các rollout mới.
- Mỗi lần đột biến, prompt ứng viên được tạo từ một **tổ tiên**, tích luỹ **bài học cấp cao** từ quan sát và phản hồi của LLM.
- Để tránh **cực trị địa phương** (vấn đề của tối ưu prompt tham lam), GEPA giữ một **biên Pareto**: thay vì chỉ tiến hoá prompt tốt nhất toàn cục, nó khám phá ngẫu nhiên **các prompt tốt nhất cho từng ví dụ**. Sự đa dạng này giúp khái quát tốt và tránh kẹt.

Đánh giá trên suy luận nhiều bước (HotpotQA), toán (AIME, LiveBench-Math), làm theo chỉ dẫn (IFBench), uỷ quyền bảo vệ quyền riêng tư (PUPA), và kiểm chứng có truy hồi (HoVer), với model mở (**Qwen3-8B**) và đóng (**GPT-4.1 mini**).
- Trên Qwen3-8B, GEPA **hơn GRPO (24.000 rollout) tới 19 điểm** mà dùng **ít rollout hơn tới 35 lần**, trung bình **+6 điểm** trên 6 task.
- GEPA vượt **MIPROv2** (state-of-the-art trước đó) trên **mọi benchmark và model**, tổng cộng **+13 điểm**, hơn gấp đôi mức +5.6 của MIPROv2.

Về định tính, prompt do GEPA học **rất phong phú**. Hình 2 là trích đoạn prompt cho module tạo truy vấn của hệ hỏi đáp nhiều bước (HotpotQA); Hình 26 cho thấy **chỉ một lần cập nhật suy ngẫm** thường đã tăng mạnh điểm. GEPA cũng được dùng làm chiến lược tìm kiếm lúc suy luận cho **tối ưu code** trên NPUEval và KernelBench (Phụ lục E), và cho **tìm prompt đối kháng** (Phụ lục F).

---

## 2. Problem Statement (Phát biểu bài toán)

**Hệ AI ghép (compound AI system)**: hệ thống module gồm một hoặc nhiều lời gọi LLM, có thể xen kẽ với lời gọi công cụ ngoài, điều phối bằng luồng điều khiển bất kỳ. Định nghĩa này bao gồm **agent**, **hệ đa agent**, và các khung như ReAct, Archon.

Hình thức hoá: `Φ = (M, C, X, Y)`.
- `M = ⟨M_1, …, M_|M|⟩` là các **module ngôn ngữ**; C là **logic luồng điều khiển**; X, Y là schema input/output toàn cục.
- Mỗi module `M_i = (π_i, θ_i, X_i, Y_i)`: **π_i** là prompt (hệ thống) gồm câu lệnh và ví dụ few-shot; **θ_i** là trọng số model; X_i, Y_i là schema input/output.
- Khi chạy, C điều phối thứ tự và cách gọi các module (vd. chuyển output module này sang module khác, gọi có điều kiện, dùng API công cụ), có thể gọi một module nhiều lần theo thứ tự bất kỳ.

Với Φ, gọi `Π_Φ = ⟨π_1, …, π_|M|⟩` là tập prompt và `Θ_Φ = ⟨θ_1, …, θ_|M|⟩` là tập trọng số. Tham số học được là `⟨Π, Θ⟩_Φ`. Với một mẫu task `(x, m)` (x là input; m là metadata để chấm như đáp án đúng, rubric, unit test), hệ sinh ra `y = Φ(x; ⟨Π, Θ⟩_Φ)`. Metric `μ: Y × M → [0, 1]` đo chất lượng của y (exact match, F1, tỷ lệ pass…). Bài toán tối ưu, với T là phân phối task:
```
⟨Π*, Θ*⟩_Φ = arg max_{⟨Π,Θ⟩_Φ}  E_{(x,m)~T} [ μ(Φ(x; ⟨Π,Θ⟩_Φ), m) ]                (1)
```
Cách phát biểu này cho phép cập nhật **cả prompt lẫn trọng số**, nên so sánh được các thuật toán tối ưu ở **không gian tham số khác nhau** (vd. GEPA tối ưu prompt, GRPO tối ưu trọng số).

**Tối ưu hiệu quả mẫu.** Thực tế, **rollout** (một lần gọi Φ cộng một lần chấm μ) thường **đắt** về tính toán, tiền, hoặc thời gian. Bộ tối ưu bị giới hạn **tối đa B rollout** trên tập `D_train = {(x, m)_i}_{i=1..N}`, có toàn quyền truy cập μ. Mục tiêu:
```
⟨Π*, Θ*⟩_Φ = arg max E_{(x,m)~T} [ μ(Φ(x; ⟨Π,Θ⟩_Φ), m) ]   s.t.  #rollouts ≤ B      (2)
```
Thách thức cốt lõi: **làm sao rút được tối đa tín hiệu học từ mỗi rollout đắt đỏ**, để thích nghi hiệu quả hệ AI module phức tạp trong điều kiện ít dữ liệu hoặc ngân sách hạn chế?

> **Hình 2.** Prompt GEPA sinh cho bước truy hồi thứ hai của hệ hỏi đáp nhiều bước (GPT-4.1 mini).
> - **Prompt gốc**: "Given the fields question, summary_1, produce the fields query."
> - **Prompt sau GEPA** (rút gọn): giải thích nhiệm vụ (sinh truy vấn cho *hop thứ hai*, tìm tài liệu **chưa** có ở hop đầu); "Bài học và quan sát chính" (tài liệu hop đầu thường chỉ phủ một thực thể; tài liệu còn thiếu thường là khái niệm liên quan được nhắc trong summary_1 nhưng không hỏi trực tiếp; ví dụ: nếu summary_1 nói về một giáo xứ nhỏ mà câu hỏi hỏi dân số vùng rộng hơn, hãy truy vấn vùng đó); "Cách xây truy vấn"; "Chiến lược thực tế"; "Output". Phụ lục N so sánh prompt GEPA với prompt MIPROv2 cho mọi task.

---

## 3. GEPA: Reflective Prompt Evolution (Tiến hoá prompt bằng suy ngẫm)

GEPA là bộ tối ưu **hiệu quả mẫu** cho hệ AI ghép, dựa trên ba nguyên tắc: **tiến hoá prompt kiểu di truyền**, **suy ngẫm bằng phản hồi ngôn ngữ**, và **chọn ứng viên theo Pareto** (3.1).

Input của GEPA: hệ Φ khởi tạo với prompt đơn giản, tập `D_train`, metric μ, **hàm phản hồi μ_f**, và ngân sách B. Lưu ý GEPA **chỉ tiến hoá prompt Π_Φ**; trọng số Θ_Φ **giữ nguyên**.

> **Hình 3.** Mỗi vòng, GEPA đề xuất ứng viên mới bằng cách cải tiến ứng viên có sẵn theo một trong hai chiến lược: **Reflective Prompt Mutation** hoặc **System Aware Merge**. Ứng viên được thử trên minibatch trước, nếu cải thiện thì thử trên tập lớn hơn. Thay vì luôn đột biến ứng viên tốt nhất (dễ kẹt cực trị địa phương), GEPA dùng **chọn mẫu theo Pareto**: lọc và lấy mẫu từ danh sách ứng viên tốt nhất **cho từng ví dụ**, đảm bảo đủ đa dạng.

**Vòng lặp tối ưu di truyền.** GEPA bắt đầu với một **pool ứng viên** P chỉ gồm hệ gốc; mỗi ứng viên là một bản cụ thể `⟨Π, Θ_frozen⟩_Φ`. Sau đó lặp, đề xuất ứng viên mới cho tới khi hết ngân sách. Ứng viên mới được sinh từ ứng viên cũ bằng **đột biến suy ngẫm** hoặc **lai ghép**, kế thừa tín hiệu học từ cha mẹ và từ rollout của chính nó, nên GEPA **tích luỹ kiến thức dọc cây di truyền**. Mỗi vòng:
1. chọn ứng viên triển vọng;
2. đề xuất và thử một biến thể trên minibatch;
3. nếu tốt hơn cha mẹ thì thêm vào P (ghi lại tổ tiên) và chấm trên **D_pareto** (tập validation dùng để chọn).

Hết ngân sách, GEPA trả về ứng viên có **điểm tổng tốt nhất trên D_pareto**.

**Đột biến prompt bằng suy ngẫm.** Trace ngôn ngữ sinh ra khi chạy hệ AI ghép cho thấy rõ **hành vi và trách nhiệm** của từng module, vì chúng ghi lại suy luận trung gian. Ghép trace với **kết quả cuối** (thành công hay thất bại), ta có giá trị **chẩn đoán** lớn: lần ngược lỗi hoặc thành công về **quyết định cụ thể ở cấp module**. LLM có thể dùng trace này để **suy ngẫm**, thực hiện **gán trách nhiệm ngầm (credit assignment)** cho module liên quan, rồi cập nhật có mục tiêu từng module. Cụ thể:
- Chạy ứng viên được chọn trên một **minibatch** câu hỏi lấy ngẫu nhiên từ tập train, ghi trace.
- Từ trace, lấy input, output và suy luận của module; gọi **μ_f**, trả về **điểm số và phản hồi dạng chữ** (vd. lỗi biên dịch, rubric không đạt).
- Chọn **một module** cần cập nhật theo chính sách **round-robin**.
- Đưa cho **reflection LM**: prompt hiện tại, quỹ đạo, điểm, phản hồi. Nhiệm vụ là **quy thành công/thất bại về các thành phần của prompt** và đề xuất câu lệnh mới.
- Chạy lại chương trình với module mới trên minibatch; nếu điểm tăng thì thêm vào pool.

Meta-prompt dùng cho suy ngẫm ở Phụ lục C; thuật toán đầy đủ ở Thuật toán 1.

**Trace đánh giá là tín hiệu chẩn đoán.** Văn bản LLM sinh ra là **trace thực thi**. Văn bản môi trường sinh ra khi tính thưởng (vd. lỗi biên dịch trước khi cho điểm 0) là **trace đánh giá**. Nhiều metric dùng chiến lược phong phú (biên dịch, chạy, đo hiệu năng code) và sinh văn bản trước khi ra một con số. GEPA mở rộng μ thành **hàm phản hồi μ_f**, trả về điểm kèm **`feedback_text`**. Phản hồi có thể **riêng cho từng module** (vd. trong hệ nhiều bước, sau mỗi hop). Nếu có người chấm viết giải thích cho điểm, có thể thêm vào D_train làm `feedback_text` phụ.

> 💡 **Giải thích.** Đây là khác biệt cốt lõi với RL. GRPO chỉ biết "lần này được 0 điểm". GEPA biết "được 0 điểm **vì** câu trả lời thiếu từ khoá X / truy vấn hop 2 lặp lại hop 1 / code lỗi ở dòng 12". Phản hồi dạng chữ cho nhiều thông tin hơn, nên cần ít lần thử hơn.

### 3.1 Chọn ứng viên theo Pareto

Cách ngây thơ là luôn chọn ứng viên tốt nhất, nhưng cách này thường **kẹt cực trị địa phương**: khi đã có chiến lược trội, khó vượt qua, và bộ tối ưu tiêu hết ngân sách mà không học được chiến lược mới. Hình 4a minh hoạ: sau khi tìm được một chiến lược mới (nút con đầu tiên), tìm kiếm cứ cố tinh chỉnh nó, thất bại, rồi cạn ngân sách.

GEPA dùng chiến lược **"illumination" theo Pareto** (Thuật toán 2):
- Với **mỗi ví dụ train**, ghi lại điểm cao nhất đạt được bởi bất kỳ ứng viên nào, tạo thành **biên Pareto**.
- Giữ các ứng viên **đạt điểm cao nhất ở ít nhất một ví dụ**; loại các ứng viên bị **trội hoàn toàn**.
- Từ tập đã lọc, **chọn ngẫu nhiên**, xác suất tỷ lệ với **số ví dụ ứng viên đó dẫn đầu**.

Nhờ vậy GEPA thoát cực trị địa phương mà không làm phình tìm kiếm, cân bằng khám phá và khai thác bằng cách dồn tài nguyên cho các ứng viên mang **chiến lược "chiến thắng"**.

> 💡 **Ví dụ.** Prompt A giỏi câu hỏi về người, prompt B giỏi câu hỏi về địa danh. Điểm trung bình của B hơi thấp hơn A. Chọn "tốt nhất" thì B bị bỏ mãi. Chọn theo Pareto thì B vẫn có cơ hội được phát triển, và sau này có thể ghép với A.

---

## 4. Evaluation (Đánh giá)

Chia train/validation/test chuẩn. Bộ tối ưu được truy cập đầy đủ tập train (cả nhãn). Có thể theo dõi điểm validation (vd. để dừng sớm), nhưng **không được xem trực tiếp nội dung** các mẫu validation.

- **6 benchmark**: AIME-2025, LiveBench-Math, HotpotQA, IFBench, HoVer, PUPA, mỗi benchmark đi kèm một hệ AI ghép và hàm phản hồi có sẵn.
- **Model**: Qwen3-8B và GPT-4.1 mini, cấu hình suy luận chuẩn.
- **So sánh với**: MIPROv2, Trace (OptoPrime), TextGrad, và **GRPO**. GRPO dùng **LoRA** vì chi phí thấp và đã được dùng thành công với GRPO; Hình 10 có thêm so sánh với GRPO **fine-tune toàn bộ**.
- Chi tiết benchmark, hệ thống, hàm phản hồi (G.1), cấu hình suy luận (G.2), chi phí (G.3), cấu hình bộ tối ưu (G.4) ở Phụ lục G.

### Bảng 1 — Qwen3-8B

GEPA và GEPA+Merge tốt hơn GRPO với ít rollout hơn nhiều ở mọi benchmark trừ AIME. Ví dụ IFBench: GEPA tìm được prompt tối ưu chỉ sau **678 rollout**, đạt 38.61%, hơn GRPO (35.88%) dùng 24.000 rollout.

| Qwen3-8B | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math | Tổng hợp | Cải thiện |
|---|---|---|---|---|---|---|---|---|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 27.33 | 48.70 | 45.23 | — |
| GRPO | 43.33 | 35.88 | 38.67 | 86.66 | **38.00** | 51.26 | 48.91 | +3.68 |
| MIPROv2 | 55.33 | 36.22 | 47.33 | 81.55 | 20.00 | 46.60 | 47.84 | +2.61 |
| **GEPA** | 62.33 | **38.61** | **52.33** | **91.85** | 32.00 | **51.95** | **54.85** | **+9.62** |
| GEPA+Merge | **64.33** | 28.23 | 51.67 | 86.26 | 32.00 | **51.95** | 52.40 | +7.17 |
| *Tổng ngân sách (số rollout)* | | | | | | | | |
| GEPA (+Merge) | 6.871 | 3.593 | 7.051 | 2.426 | 1.839 | 1.839 | 3.936 | — |
| GRPO | 24.000 | 24.000 | 24.000 | 24.000 | 24.000 | 24.000 | 24.000 | — |

**Quan sát 1: Tiến hoá prompt bằng suy ngẫm rất hiệu quả về mẫu và có thể vượt RL trong không gian trọng số.**
- Trên 6 benchmark, GEPA thích nghi nhanh và khái quát tốt: **hơn GRPO (24.000 rollout) tới 19 điểm**, dùng **ít rollout hơn tới 35 lần**.
- Đạt điểm test tối ưu với ít rollout hơn 4–35 lần, và **vượt GRPO ở 5/6 task** với mức 19.0, 2.73, 13.66, 5.19 và 0.7 điểm.
- GEPA **bằng điểm validation tốt nhất của GRPO** chỉ sau **243, 402, 330, 1.143, 1.179 và 306 rollout**, tức hiệu quả mẫu cao hơn **tới 78 lần**.
- GEPA+Merge nới rộng khoảng cách, **hơn GRPO 21 điểm** với ngân sách tương đương GEPA.

Phần lớn ngân sách rollout của GEPA dùng cho **validation**, chỉ để chọn ứng viên chứ không tạo tín hiệu học. Nếu chỉ tính rollout trên tập train, GEPA cần **79–737 rollout** để đạt hiệu năng tối ưu. Để bằng điểm validation tốt nhất của GRPO chỉ cần **102, 32, 6 và 179** rollout train cho 4 task. Đây là minh chứng cho hiệu quả mẫu cao của học bằng suy ngẫm.

Vì theo dõi điểm validation chiếm phần lớn ngân sách, có thể tăng hiệu quả hơn nữa bằng tập validation nhỏ hơn hoặc theo dõi trên tập con chọn động. Tác giả để ngỏ cho công trình sau. Hình 1a, 1b, 13c, 14c là đường hiệu năng theo số rollout của mọi bộ tối ưu trên HotpotQA, IFBench, HoVer, PUPA.

### Bảng 2 — GPT-4.1 mini

Là hệ tối ưu prompt, GEPA dùng được ngay với **model đóng**, vượt các bộ tối ưu prompt tốt nhất: MIPROv2 (2 cấu hình: chỉ tối ưu câu lệnh "MIPROv2-No-Demos", và tối ưu cả câu lệnh lẫn few-shot "MIPROv2"), Trace (OptoPrime), TextGrad. Prompt tối ưu bởi GEPA còn **khái quát chéo model**: "GEPA-Qwen-Opt" (tối ưu hoàn toàn cho và bằng Qwen3-8B yếu hơn) đạt **+9%** khi đem dùng cho GPT-4.1 mini không sửa gì, **vượt mọi baseline** đã tối ưu trực tiếp trên GPT-4.1 mini.

| GPT-4.1 mini | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math | Tổng hợp | Cải thiện |
|---|---|---|---|---|---|---|---|---|
| Baseline | 38.00 | 47.79 | 46.33 | 78.57 | 49.33 | 58.20 | 53.03 | — |
| Trace (OptoPrime) | 60.33 | 51.19 | 46.00 | 74.18 | 45.33 | 60.74 | 56.30 | +3.27 |
| MIPROv2-No-Demos | 38.00 | 52.04 | 51.33 | 91.85 | 48.67 | 60.97 | 57.14 | +4.11 |
| MIPROv2 | 58.00 | 49.15 | 48.33 | 83.37 | 51.33 | 61.84 | 58.67 | +5.64 |
| TextGrad | 62.33 | 48.64 | 47.67 | 85.68 | 46.67 | 63.84 | 59.14 | +6.11 |
| GEPA | **69.00** | 52.72 | 51.67 | 94.47 | **59.33** | **64.13** | 65.22 | +12.19 |
| **GEPA+Merge** | 65.67 | **55.95** | **56.67** | **96.46** | **59.33** | **64.13** | **66.36** | **+13.33** |
| *Tối ưu bằng Qwen3-8B, chạy trên GPT-4.1 mini* | | | | | | | | |
| GEPA-Qwen-Opt | 65.67 | 49.83 | 54.67 | 90.05 | 52.67 | 59.31 | 62.03 | +9.00 |

**Quan sát 2: Tiến hoá prompt bằng suy ngẫm giúp tối ưu chỉ câu lệnh mà vẫn vượt tối ưu đồng thời câu lệnh và few-shot.**
- GEPA luôn hơn MIPROv2 ở mọi cấu hình, cách biệt tới **11.1 điểm** (GPT-4.1 mini) và **12 điểm** (Qwen3-8B).
- GEPA và GEPA+Merge **hơn gấp đôi** mức tăng tổng của MIPROv2 (+13.33 và +12.19 so với +5.64).
- Các công trình trước (Opsahl-Ong et al., Wan et al., 2024) cho thấy **tối ưu ví dụ few-shot** thường hơn tối ưu câu lệnh. Kết quả này gợi ý **xu hướng đang đổi chiều**, nhờ khả năng làm theo chỉ dẫn và tự suy ngẫm của LLM tiến bộ, cùng thiết kế của GEPA.
- **Khoảng cách khái quát** (chênh lệch validation và test) của câu lệnh tiến hoá bằng suy ngẫm **thấp hơn** (Hình 15).
- Khác các công trình trước (cải thiện chủ yếu nhờ "giả ví dụ"), prompt của GEPA thường chứa **hướng dẫn khai báo chi tiết** để hoàn thành task (Hình 2). Prompt đầy đủ ở Phụ lục N.

**Quan sát 3: Chiến lược chọn ứng viên tiếp theo ảnh hưởng mạnh tới quá trình và kết quả; chọn theo Pareto có lợi thế rõ.** So sánh với **SelectBestCandidate** (luôn chọn ứng viên tốt nhất, giống TextGrad) và **BeamSearch(N=4)** (giữ top-N, giống APO). GEPA theo Pareto hơn BeamSearch tới **11.33 điểm** và hơn SelectBestCandidate tới **8.17 điểm**; tổng hợp hơn **7.33** và **6.4 điểm**.

### Bảng 3 — Chiến lược chọn ứng viên (Qwen3-8B, cùng khung tiến hoá)

| Qwen3-8B | HotpotQA | IFBench | HoVer | PUPA | Tổng hợp | Cải thiện |
|---|---|---|---|---|---|---|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 48.84 | — |
| SelectBestCandidate | 58.33 | 30.44 | 45.33 | 85.45 | 54.89 | +6.05 |
| BeamSearch | 57.33 | 36.39 | 41.00 | 81.08 | 53.95 | +5.11 |
| **GEPA (Pareto)** | **62.33** | **38.61** | **52.33** | **91.85** | **61.28** | **+12.44** |

> **Hình 4.** (Trái) Luôn chọn ứng viên tốt nhất → kẹt cực trị địa phương sau một vòng: cây tìm kiếm là một "chổi" toả ra từ một nút. (Phải) Chọn theo Pareto → cây tìm kiếm cân bằng, nhiều nhánh, tìm được chương trình tốt hơn với cùng ngân sách.

Các quan sát thêm (Phụ lục H):
- **Quan sát 4**: prompt tối ưu câu lệnh **rẻ hơn và khái quát tốt hơn** prompt few-shot.
- **Quan sát 5**: chiến lược lai ghép có nhận biết hệ thống như **Merge** giúp được, nhưng hiệu quả phụ thuộc cách chia ngân sách giữa đột biến và lai ghép, và thời điểm gọi.
- **Quan sát 6**: prompt tối ưu bởi GEPA **khái quát chéo model**.

---

## 5. Extended Applications (Ứng dụng mở rộng)

**GEPA cho tìm kiếm lúc suy luận.** Đặt tập validation bằng tập train (`D_val = D_train`); khi đó GEPA lặp lại việc đề xuất lời giải tốt hơn cho từng task đích. Thử nghiệm sinh code cho phần cứng:
- viết **kernel cho NPU AMD** (NPUEval): tăng mức tận dụng vector trung bình từ **4.25%** (tinh chỉnh tuần tự) lên **30.52%**;
- sinh **kernel CUDA cho GPU NVIDIA** (KernelBench) với GPT-4o: tỷ lệ task có kernel CUDA **nhanh hơn PyTorch** gốc tăng từ gần **0%** lên **hơn 20%**.

**GEPA cho tìm prompt đối kháng.** Đảo ngược phần thưởng, tức tìm câu lệnh **làm giảm** hiệu năng. Trên AIME-2025 với GPT-5 Mini, từ một câu lệnh đơn giản, GEPA tiến hoá được một **câu gây nhiễu kiểu "kiến thức vui"** làm pass@1 giảm **từ 76% xuống 10%**.
- Tìm kiếm dùng các bài AIME 2022–2024; đánh giá trên AIME-2025, 5 lần mỗi bài (30 bài, 150 lần sinh).
- Cách này lộ ra các tương tác làm theo chỉ dẫn **giòn** (vd. thông tin thừa kèm yêu cầu định dạng nghiêm ngặt), tạo ra **bộ kiểm thử độ bền** tái sử dụng được khi triển khai.
- Kết quả tương tự các nghiên cứu về trigger đối kháng không phụ thuộc câu hỏi (tăng lỗi 2–7 lần). Ở đây câu lệnh của GEPA tăng lỗi **3.8 lần** (từ 24% lên 90%).

---

## 6. Related Work (Công trình liên quan)

- **Tối ưu prompt**: cải thiện LLM nhưng thường cần chuyên môn thủ công (vd. CoT). Để mở rộng, các phương pháp gần đây dùng LLM tự tối ưu prompt. GEPA cũng dùng LLM nhưng khác ở chỗ dùng **phản hồi dạng chữ từ môi trường**, **tìm kiếm có nhận biết Pareto**, và **chiến lược tiến hoá cho từng module con**.
- **Thuật toán tiến hoá**: EvoPrompt tiến hoá quần thể prompt; Rainbow Teaming dùng tiến hoá theo chất lượng–đa dạng cho prompt đối kháng. GEPA thêm phản hồi riêng của miền để đột biến có mục tiêu, nên hiệu quả mẫu cao hơn. AlphaEvolve và OpenEvolve áp dụng tìm kiếm tiến hoá trực tiếp để viết lại code; AlphaEvolve nhắm một bài toán khó, còn GEPA đưa tiến hoá vào **prompt xuyên nhiều miền**, kết hợp tối ưu biên Pareto để chuyển chiến thuật giữa các bài toán liên quan.
- **Cải thiện dựa trên phản hồi**: thường dùng RL (vd. tín hiệu bỏ phiếu đa số), nhưng kém hiệu quả mẫu khi phần thưởng tính chậm. Cách khác là **học trong không gian ngôn ngữ**: tự bootstrap trong context, bộ nhớ và kỹ năng cho quy trình, tổng hợp chiến lược lúc test (Dynamic Cheatsheet), bộ đệm suy luận. GEPA dùng ví dụ để đề xuất **câu lệnh mới**, tạo ra quy tắc riêng cho task.
- **Tối ưu hệ AI ghép và agent**: DSPy tìm hoặc bootstrap ví dụ few-shot; TextGrad lan truyền ngược phản hồi chữ; MIPROv2 tối ưu đồng thời câu lệnh và ví dụ bằng tối ưu Bayes. Các cách này chủ yếu dựa vào **phần thưởng toàn cục**. Agent-Pro, Optimas có phần thưởng cục bộ cho từng module. GEPA kết hợp phần thưởng toàn cục với phản hồi chữ từ môi trường cho từng module, và giữ biên Pareto trên từng mẫu dữ liệu.

---

## 7. Conclusion (Kết luận)

GEPA là bộ tối ưu prompt cho agent và quy trình LLM bất kỳ, dùng **suy ngẫm tường minh** và **chọn theo Pareto**. Nó cho thấy hiệu quả mẫu vượt trội so với RL (GRPO), đồng thời vượt bộ tối ưu prompt hàng đầu (MIPROv2). Nhờ đưa phản hồi ngôn ngữ vào tường minh và giữ một pool đa dạng ứng viên tối ưu Pareto, GEPA **thích nghi nhanh** hệ AI cho task mới. Kết quả gợi ý **suy ngẫm bằng ngôn ngữ** là chiến lược mở rộng được để tối ưu quy trình AI phức tạp trong thực tế, nhất là khi **tài nguyên hạn chế**. GEPA cũng hứa hẹn như một chiến lược tìm kiếm lúc suy luận, viết được code trong các miền khó.

---

## Phụ lục (chọn lọc)

### C. Meta-prompt suy ngẫm và cập nhật prompt của GEPA

Nguyên văn (giữ tiếng Anh vì đây là prompt thật GEPA dùng):
```
I provided an assistant with the following instructions to perform a task for me:
```
<current instruction>
```
The following are examples of different task inputs provided to the assistant
along with the assistant's response for each of them, and some feedback on how the
assistant's response could be better:
```
<Inputs, Outputs and Feedback for minibatch of examples>
```
Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task
description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all
niche and domain specific factual information about the task and include it in the
instruction, as a lot of it may not be available to the assistant in the future.
The assistant may have utilized a generalizable strategy to solve the task, if so,
include that in the instruction as well.

Provide the new instructions within ``` blocks.
```

Dịch ý: *"Tôi đã đưa cho trợ lý câu lệnh sau… Dưới đây là các input, câu trả lời của trợ lý và phản hồi về cách làm tốt hơn. Hãy viết câu lệnh mới: đọc kỹ input để suy ra định dạng và mô tả task chi tiết; đọc mọi câu trả lời và phản hồi; đưa mọi thông tin chuyên biệt của miền vào câu lệnh (vì sau này trợ lý có thể không có); nếu trợ lý đã dùng một chiến lược khái quát được thì cũng đưa vào."*

> 💡 Đây là phần dễ chiếu nhất khi thuyết trình: "bộ tối ưu" thực chất chỉ là **một prompt bảo LLM viết lại prompt**, nhưng được đặt trong một vòng lặp tiến hoá có chọn lọc.

### D. Thuật toán

**Thuật toán 1 — GEPA**
```
Input: hệ Φ, tập D_train, metric μ, hàm phản hồi μ_f, ngân sách B
Siêu tham số: kích thước minibatch b, kích thước tập Pareto n_pareto

1:  Chia D_train thành D_feedback và D_pareto, sao cho |D_pareto| = n_pareto
2:  Khởi tạo pool P ← [Φ], cha mẹ A ← [None]
3:  for mỗi (x_i, m_i) trong D_pareto:  S_Φ[i] ← μ(Φ(x_i), m_i)
6:  while chưa hết ngân sách B:
7:      k ← SELECTCANDIDATE(P, S)                 # chọn theo Pareto (Thuật toán 2)
8:      j ← SELECTMODULE(Φ_k)                     # round-robin
9:      M ← minibatch kích thước b từ D_feedback
10:     Thu phản hồi, điểm, trace cho Φ_k[j] trên M bằng μ_f
11:     π′_j ← UPDATEPROMPT(π_j, phản hồi, trace[j])   # gọi reflection LM với meta-prompt
12:     Φ′ ← bản sao Φ_k với module j cập nhật bằng π′_j
13:     σ, σ′ ← điểm trung bình trên M (trước, sau)
14:     if σ′ tốt hơn:
15:         thêm Φ′ vào P; thêm k vào A
16:         for mỗi (x_i, m_i) trong D_pareto:  S_Φ′[i] ← μ(Φ′(x_i), m_i)
20: return Φ* có điểm trung bình cao nhất trên D_pareto
```

**Thuật toán 2 — Chọn ứng viên theo Pareto**
```
function SELECTCANDIDATE(P, S):
    for mỗi ví dụ i:
        s*[i] ← max_k S_P[k][i]                       # điểm cao nhất trên ví dụ i
        P*[i] ← {P[k] : S_P[k][i] = s*[i]}            # các ứng viên đạt điểm đó
    C ← các ứng viên duy nhất trong ∪_i P*[i]
    D ← ∅
    while tồn tại Φ ∈ C \ D bị một ứng viên khác trong C \ D trội hơn:
        D ← D ∪ {Φ}                                   # loại ứng viên bị trội
    Bỏ D khỏi mỗi P*[i] để có P̂*[i]
    f[Φ] ← số ví dụ i mà Φ ∈ P̂*[i]
    Lấy mẫu Φ_k từ Ĉ với xác suất ∝ f[Φ_k]
    return k
```

**D.1 Merge — lai ghép có nhận biết hệ thống (Thuật toán 3, 4).** Chọn hai ứng viên i, j có **chung tổ tiên** a nhưng đã tối ưu **các module khác nhau** (chiến lược bổ trợ), cả hai đều tối ưu Pareto và đều tốt hơn tổ tiên. Với mỗi module: nếu chỉ i thay đổi so với tổ tiên thì lấy của i, nếu chỉ j thay đổi thì lấy của j, nếu cả hai đều đổi thì lấy của ứng viên điểm cao hơn. Điều kiện chặt nên merge xảy ra **khá hiếm**.

### G. Thiết lập đánh giá (chi tiết)

**G.1 Benchmark, hệ AI tham chiếu, hàm phản hồi**
- **HotpotQA**: 113K câu hỏi dựa trên Wikipedia, cần suy luận qua nhiều tài liệu. Dùng chương trình HoVerMultiHop, sửa hop cuối thành **trả lời câu hỏi**. Phản hồi chữ: **các tài liệu liên quan còn cần truy hồi** ở mỗi bước. 150 train / 300 val / 300 test.
- **IFBench**: kiểm tra làm theo chỉ dẫn chính xác, đặc biệt ràng buộc output ("chỉ trả lời yes/no", "nhắc một từ ít nhất 3 lần"). Test gồm 58 ràng buộc mới, ngoài phân phối. Train/val lấy từ IF-RLVR Train. Hệ 2 bước: trả lời, rồi viết lại câu trả lời theo ràng buộc. Phản hồi: **ràng buộc nào đã đạt và chưa đạt**. 150 / 300 / 294.
- **AIME-2025**: 2 bộ × 15 câu = 30 câu. Train và val từ AIME 2022–2024 (90 câu, chia đều); test là AIME-2025, mỗi câu lặp 5 lần. Hệ: một bước ChainOfThought.
- **LiveBench-Math**: 368 câu (lấy ngày 30/7/2025), xáo trộn (seed 0) và chia đều train/val/test. Hệ: một bước ChainOfThought.
- **HoVer**: kiểm chứng tuyên bố nhiều bước trên Wikipedia; đánh giá khả năng viết truy vấn tìm **mọi tài liệu cần thiết**. Chương trình HoverMultiHop: tối đa 3 hop truy hồi, 2 module viết truy vấn, 2 module tóm tắt tài liệu. Phản hồi: **tài liệu đúng đã tìm được và tài liệu còn thiếu**. 150 / 300 / 300.
- **PUPA**: uỷ quyền có bảo vệ quyền riêng tư, tức trả lời bằng tổ hợp model tin cậy và không tin cậy, giữ chất lượng mà **giảm rò rỉ thông tin cá nhân (PII)**. Hệ PAPILLON: viết lại câu hỏi và viết lại câu trả lời. Phản hồi: điểm chất lượng và điểm rò rỉ PII. 111 / 111 / 221.

**G.2 Model và tham số suy luận**: mỗi hệ AI ghép khởi tạo một model cho mọi module. Context tối đa 16.384 token.
- **Qwen3-8B** (cho mọi thí nghiệm mở, kể cả GRPO): temperature 0.6, top-p 0.95, top-k 20 cho cả train lẫn suy luận.
- **GPT-4.1 mini** (`gpt-4.1-mini-2025-04-14`, qua API OpenAI): temperature 1.0.

**G.3 Chi phí**: toàn bộ thí nghiệm Bảng 2 với GPT-4.1 mini **dưới 500 USD**: GEPA 86 USD, GEPA-Merge 67 USD, MIPROv2 76 USD, Trace và TextGrad cộng lại 172 USD.

**G.4 Bộ tối ưu**
- **MIPROv2**: tối ưu đồng thời câu lệnh và ví dụ bằng tối ưu Bayes (TPE). Chạy `auto=heavy`: 18 câu lệnh ứng viên, 18 bộ few-shot. Số rollout 2.270 (PUPA) tới 6.926 (HoVer).
- **Trace, TextGrad**: cài trong framework Trace, cùng kiến trúc, cùng prompt đầu, cùng dữ liệu, cùng ngân sách và hàm phản hồi như GEPA.
- **GRPO**:
  - Hệ nhiều module: mỗi bước có nhóm 12, 4 ví dụ train (batch 48), **LoRA rank 16, α 64, dropout 0.05**, bf16, nhắm `q, k, v, o, up, down, gate`. lr 1e-5, β 0.01, chuẩn hoá thang thưởng, clip gradient 0.1, cộng dồn 20 bước. Chạy **500 bước = 24.000 rollout**, validation mỗi 20 bước. Phần cứng: 1× H100/A100 80GB cộng GPU riêng để sinh rollout.
  - Hệ một module (Hình 10): **fine-tune toàn bộ**, nhóm 16, batch 32 trên 8 GPU, FSDP2, lr 1e-6.
- **GEPA**: hai biến thể GEPA và GEPA+Merge, cùng hai ablation dùng SelectBestCandidate. **Minibatch 3**; merge tối đa **5** lần. Ngân sách của GEPA **giới hạn bằng số rollout MIPROv2 đã dùng** cho từng benchmark (chênh lệch luôn dưới 10.15%). Tập train dùng làm D_feedback, tập val dùng làm D_pareto.

### H. Kết quả và phân tích (tiếp)

**Quan sát 4: Prompt tối ưu câu lệnh rẻ hơn và khái quát tốt hơn prompt few-shot.** Câu lệnh tiến hoá bằng suy ngẫm thường **ngắn hơn nhiều** so với prompt nhiều ví dụ (một ví dụ few-shot cho task phức tạp đã có thể rất dài; MIPROv2 còn tối ưu nhiều ví dụ dùng cùng lúc). Prompt của GEPA và GEPA+Merge **ngắn hơn tới 9.2 lần** so với MIPROv2. Nhìn chung, bộ tối ưu đạt điểm cao hơn có xu hướng tạo prompt ngắn hơn (Hình 16), giúp giảm chi phí API (tính theo token input), giảm độ trễ và tăng hiệu quả phục vụ.

**Quan sát 5: Lai ghép có nhận biết hệ thống có thể tăng mạnh, nhưng cần nghiên cứu thêm cách chia ngân sách và thời điểm gọi.** GEPA+Merge hơn GEPA tới **5%** nhờ tìm các nhánh tối ưu khác nhau đã học chiến lược bổ trợ (tiến hoá các module khác nhau) và ghép phiên bản tốt nhất của từng module. Merge rất tốt với GPT-4.1 mini nhưng làm **giảm** hiệu năng với Qwen3-8B (chỉ có lợi ở 1/4 task). Lý do có thể là cách chia ngân sách giữa đột biến và lai ghép, và thời điểm gọi lai ghép (dùng chung siêu tham số cho hai model). Lai ghép có lợi nhất khi đã có **các nhánh độc lập** đều tốt.

**Quan sát 6: Prompt tối ưu bởi GEPA khái quát chéo model.** "GEPA-Qwen-Opt" (tối ưu bằng Qwen3-8B, chạy trên GPT-4.1 mini) tăng tổng **+9.00%** trên 6 benchmark (tới +27.67% trên HotpotQA). Đáng chú ý, prompt chuyển sang này **vượt** MIPROv2 (+5.64%), TextGrad (+6.11%), Trace (+3.27%), dù các phương pháp đó tối ưu **trực tiếp** trên GPT-4.1 mini.

**Hình 10 (so với GRPO fine-tune toàn bộ, HoVer 2-hop, Qwen3-8B).** Khoảng cách giữa GEPA và GRPO **full fine-tune** tương tự như khi so với GRPO **LoRA**. Vậy GEPA giữ được lợi thế so với cả hai dạng GRPO, tham số đầy đủ lẫn PEFT.

> 💡 **Liên hệ đề tài:** Bảng 1 thực chất là cuộc đối đầu **Prompting (GEPA) vs PEFT (GRPO + LoRA rank 16)** trên cùng model Qwen3-8B. Đây là slide "chốt" để nối GEPA với chủ đề Prompting + PEFT.
