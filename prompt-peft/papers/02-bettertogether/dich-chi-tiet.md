# BetterTogether — Dịch và diễn giải chi tiết

> Bản dịch có diễn giải của *Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together* (Soylu, Potts, Khattab — EMNLP 2024, trang 10696–10710). PDF: [`bettertogether-emnlp2024.pdf`](bettertogether-emnlp2024.pdf).
> Ô **💡 Giải thích** là phần nhóm thêm vào, không có trong paper. Nên đọc [`../03-dspy/`](../03-dspy/) trước.

---

## Abstract

Các hệ thống NLP ngày càng có dạng **pipeline nhiều module** phức tạp, ví dụ Retrieval Augmented Generation (RAG). Mỗi module có thể dùng một LM riêng và một prompt template riêng. Các hệ thống ghép này thường **không có nhãn trung gian** và **không có dòng gradient** để tối ưu từng module, nên tối ưu toàn bộ hệ thống từ đầu tới cuối là việc khó.

Paper tìm cách tối ưu **cả trọng số LM** của từng module **lẫn prompt template** đi kèm, để tối đa một metric của task cuối. Lần đầu tiên, tác giả đề xuất **kết hợp** tối ưu trọng số và tối ưu prompt cho một pipeline LM, bằng cách **xen kẽ** hai bước để **chính LM đó tự dạy mình**.

Thí nghiệm gồm hỏi đáp nhiều bước, suy luận toán, và phân loại dựa trên đặc trưng, với `mistral-7b`, `llama-2-7b`, `llama-3-8b`. Các chiến lược **BetterTogether** (tối ưu trọng số và prompt cùng nhau) hơn "chỉ tối ưu trọng số" tới **60%** và hơn "chỉ tối ưu prompt" tới **6%**, tính trung bình theo LM và task. Optimizer BetterTogether được phát hành trong DSPy (http://dspy.ai).

---

## 1. Introduction (Giới thiệu)

Khả năng của LM ngày càng tăng, và các công trình gần đây cho thấy có thể xây hệ thống NLP mạnh hơn bằng cách **ghép nhiều kỹ năng của LM thành pipeline**: RAG, suy luận nhiều bước, trích xuất thông tin, v.v.

Những **LM Program** như vậy giúp kiểm soát thiết kế tốt hơn, vì chúng chia bài toán thành các bài toán con nhỏ, dễ quản lý, giao cho LM. Nếu dạy được LM làm chính xác các bài toán con dễ hơn này và **giao tiếp tốt với nhau** trong pipeline nhiều bước, ta có thể mở rộng đáng kể phạm vi các hệ thống NLP đáng tin cậy.

Với mục tiêu đó, Khattab et al. (2024) giới thiệu **DSPy**, framework để định nghĩa và tối ưu LM Program. Trong DSPy:
- Một chương trình là một hàm **Φ** ghép một tập các bước, gọi là **language module** `M = ⟨M_1, …, M_|M|⟩`, thành pipeline.
- Mỗi module `M_i` mô tả một **phép biến đổi ngôn ngữ "mờ"** cần học (vd. tóm tắt một tài liệu).
- Để làm vậy, mỗi module học một **prompt (template) π** để gọi một **LM có trọng số θ**.
- Bài toán tối ưu: tối đa hiệu năng kỳ vọng (theo metric **µ**) của Φ trên một tập input, bằng cách cập nhật **π và θ** của từng module.

Các công trình trước (DSPy, MIPRO) chủ yếu tối ưu **prompt dạng chuỗi rời rạc** của từng module, và chỉ thử những cách đơn giản để fine-tune trọng số. Paper này nghiên cứu thực nghiệm việc cập nhật **cả trọng số và prompt** của từng module cùng lúc. Việc này khó vì:
- Φ thường **không khả vi**;
- các module **thiếu nhãn output**, và lại **phụ thuộc lẫn nhau** phức tạp;
- trong thực tế, tập train thường **rất nhỏ** và chỉ được gọi LM **một số ít lần** khi train và suy luận.

Để giải quyết, tác giả đề xuất **xen kẽ** fine-tune trọng số và tối ưu prompt, và đánh giá các chiến lược tối ưu xấp xỉ trong đó **nhãn train cho mọi module được bootstrap** (tự sinh). Trên HotPotQA, GSM8K và Iris, các chiến lược kết hợp này rất hiệu quả với ba LM: tăng **5–78%** trên HotPotQA, **2.5–10%** trên GSM8K, **3.5–88%** trên Iris so với chiến lược chỉ prompt hoặc chỉ trọng số, trung bình trên `mistral-7b-instruct-v0.2`, `llama-2-7b-chat`, `llama-3-8b-instruct`.

---

## 2. Problem Statement (Phát biểu bài toán)

Cho một LM program **Φ**, hoạt động như hàm hộp đen `Φ: X → Y`. X và Y thường là ngôn ngữ tự nhiên (vd. câu hỏi và câu trả lời do chương trình sinh). Khi chạy, Φ gọi một hoặc nhiều lần mỗi language module trong `M = ⟨M_1, …, M_|M|⟩` (|M| ≥ 1).

Ví dụ: chương trình hỏi đáp **nhiều bước có truy hồi (multi-hop RAG)**. Pipeline phổ biến này chia input thành các câu hỏi con, dùng chúng để lần lượt tìm đoạn văn liên quan (vd. từ Wikipedia) cho tới khi trả lời được câu hỏi một cách trung thực. Tổng quát, mỗi module `M_i: X_i → Y_i` là một **lời gọi LM khai báo (declarative)**, định nghĩa bằng ngôn ngữ tự nhiên một miền input X_i (vd. câu hỏi và các đoạn văn đã tìm) và miền output Y_i (vd. một truy vấn để tìm thêm đoạn văn).

Mỗi module được hiện thực hoá đầy đủ bằng hai thứ:
1. **chuỗi prompt π_i**, trong đó input X_i được điền vào để sinh output Y_i;
2. **trọng số θ_i** gán cho tham số của LM khi chạy module này.

Ký hiệu **Φ⟨Θ,Π⟩** là phiên bản của Φ với prompt Π và trọng số Θ được gán cụ thể.

Chỉ cho một **tập train nhỏ** `X = {(x_1, m_1), …, (x_|X|, m_|X|)}`, gồm input `x_i` và metadata tuỳ chọn `m_i` (như nhãn output hoặc gợi ý khác để kiểm tra một lần chạy đúng hay sai), cùng một **metric** `µ: Y × M → R`. Mục tiêu là cấu hình prompt và trọng số của các module để tối đa:

```
arg max_{Θ,Π}  (1/|X|) · Σ_{(x,m)∈X}  µ( Φ⟨Θ,Π⟩(x), m )
```

Người tinh chỉnh pipeline LM thực chất đang cố đạt mục tiêu này, và đây cũng là một phần rất lớn trong bài toán tối ưu của DSPy. Tiếc là bài toán này **không giải chính xác được (intractable)**: không gian tìm kiếm rất lớn, và **không có gradient hay nhãn trung gian** cho từng module. Vì vậy tác giả tìm các chiến lược **xấp xỉ**.

> 💡 **Giải thích.** Φ⟨Θ,Π⟩ giống như một "model" mà có hai loại "tham số": Π là **chữ** (prompt, ví dụ few-shot), Θ là **số** (trọng số LM). Không thể đạo hàm theo Π (chữ rời rạc), và không thể lan truyền gradient qua bước tìm kiếm. Nên không dùng được gradient descent cho cả pipeline.

---

## 3. BetterTogether: xen kẽ bước tối ưu trọng số và tối ưu prompt

Giả thuyết của tác giả: khi một LM lớn được dùng để **tự dạy mình** làm task mà LM program định nghĩa, thì **cả fine-tune trọng số lẫn tối ưu prompt đều cần thiết** để đạt chất lượng cao nhất. Cụ thể:
1. **Tối ưu prompt trước khi fine-tune** giúp có **nhiều điểm dữ liệu thành công hơn** để fine-tune.
2. **Tối ưu prompt sau khi fine-tune** giúp **điều chỉnh hành vi** của LM program, cho output chất lượng cao hơn.

Điều này có thể gây bất ngờ, vì fine-tune thường được coi là công cụ mạnh hơn, nhất là khi cả hai cách cuối cùng đều dùng **cùng một tập input train X**.

### Thuật toán 1 — BetterTogether

```
Input: Program Φ⟨Θ,Π⟩ = Φ_Θ ⊙ Φ_Π,
       trọng số module Θ = [θ_1, …, θ_|Φ|], prompt module Π = [π_1, …, π_|Φ|]
       Tập train X, metric µ

function BETTERTOGETHER(Φ⟨Θ,Π⟩, X, µ)
    Π′  ← OPTIMIZEPROMPTS(Φ⟨Θ,Π⟩,  X, µ)     # 1. tối ưu prompt
    Θ′  ← FINETUNEWEIGHTS(Φ⟨Θ,Π′⟩, X, µ)     # 2. fine-tune với prompt đã tối ưu
    Π″  ← OPTIMIZEPROMPTS(Φ⟨Θ′,Π⟩, X, µ)     # 3. tối ưu prompt lại với trọng số mới
    return Φ⟨Θ′,Π″⟩
```

Với một chương trình Φ, thuật toán:
1. tối ưu prompt của Φ;
2. fine-tune trọng số LM bằng dữ liệu **bootstrap từ prompt đã tối ưu**;
3. tối ưu prompt **lần nữa**, dùng trọng số đã fine-tune.

Về nguyên tắc, mỗi bước đều có thể bỏ qua. Điều này tạo ra các tổ hợp sẽ được đánh giá ở mục 4:

| # | Chiến lược | Ký hiệu |
|---|---|---|
| 1 | Chương trình gốc, prompt đơn giản do người dùng viết, không fine-tune | Vanilla zero-shot |
| 2 | Chỉ tối ưu prompt | Π |
| 3 | Chỉ tối ưu trọng số | Θ |
| 4 | Tối ưu prompt hai lần (lần 2 bắt đầu từ chương trình đã tối ưu) | Π→Π |
| 5 | Tối ưu trọng số hai lần | Θ→Θ |
| 6 | Prompt rồi trọng số | Π→Θ |
| 7 | Trọng số rồi prompt | Θ→Π |
| 8 | Prompt, trọng số, rồi prompt | Π→Θ→Π |

Tác giả kỳ vọng **ba chiến lược cuối** luôn hơn năm chiến lược đầu.

> 💡 **Giải thích.** Chiến lược 4 và 5 là **đối chứng công bằng**: chúng tốn hai vòng tối ưu giống BetterTogether, nhưng chỉ dùng một loại. Nếu Π→Θ hơn cả Π→Π lẫn Θ→Θ thì lợi ích đến từ việc **kết hợp hai loại**, không phải chỉ vì tối ưu nhiều vòng hơn.

### Hiện thực hoá bằng họ thuật toán Bootstrap-*

Để Thuật toán 1 hoàn chỉnh, cần chọn cách cụ thể cho tối ưu prompt và fine-tune. Tác giả dùng họ **Bootstrap-\*** của DSPy. Các thuật toán này **chạy phiên bản ban đầu của chương trình** trên các ví dụ `(x_i, m_i) ∈ X` và **ghi lại input/output ở từng module** khi output cuối là "đúng", tức `µ(Φ(x_i), m_i) ≥ λ` với ngưỡng λ (vd. 1.0 với accuracy nhị phân).

Điểm quan trọng: đúng với cách phát biểu bài toán, cả tối ưu prompt lẫn trọng số **không train trên dữ liệu gán nhãn tay**, mà train trên **trace do chính chương trình tự sinh**.

### Thuật toán 2 — Hiện thực hai bước tối ưu

```
function BOOTSTRAPFEWSHOTRS(Φ⟨Θ,Π⟩, X, µ)            # tối ưu prompt
    T, V ← SPLITINTOTRAINANDVALIDATION(X)
    τ ← BOOTSTRAPTRACES(Φ⟨Θ,Π⟩, T)
    τ ← FILTERTRACES(τ, µ)                            # chỉ giữ trace đúng
    A ← {}
    for τ′ in SAMPLEFEWSHOTSUBSETS(τ):
        Π′ ← CONSTRUCTFEWSHOTPROMPTS(τ′)
        σ ← (1/|V|) Σ_{(x_i,m_i)∈V} µ(Φ⟨Θ,Π′⟩(x_i), m_i)   # điểm trên tập validation
        A ← A ∪ {(σ, Π′)}
    return Π_max                                      # bộ prompt có điểm cao nhất

function BOOTSTRAPFINETUNE(Φ⟨Θ,Π⟩, X, µ)             # fine-tune trọng số
    τ ← BOOTSTRAPTRACES(Φ⟨Θ,Π⟩, X)
    τ ← FILTERTRACES(τ, µ)
    Θ′ ← TRAINLM(τ)
    return Θ′

OPTIMIZEPROMPTS := BOOTSTRAPFEWSHOTRS
FINETUNEWEIGHTS := BOOTSTRAPFINETUNE
```

**Tối ưu prompt: BootstrapFewshotRS (BFRS).** BFRS tự sinh các ví dụ few-shot tiềm năng cho **mọi module**, rồi dùng **tìm kiếm ngẫu nhiên (random search)** để chọn ví dụ nào được đưa vào prompt.
1. Chia X thành tập train T và tập validation V.
2. Chạy Φ⟨Θ,Π⟩ trên các input của T, thu cặp input–output của mọi module cho mỗi `x_i`. Đây là một **trace τ**. Chỉ giữ các trace được µ chấm điểm cao.
3. Lấy mẫu nhiều **tập con nhỏ τ′** của các trace, mỗi tập con chứa một ví dụ few-shot tiềm năng cho từng module.
4. Chọn tập con mà khi dùng làm prompt few-shot cho **điểm cao nhất trên V**.

Chiến lược tìm kiếm đơn giản này được biết là luôn cải thiện mạnh chất lượng khi prompt LM program, thường hơn cả tối ưu câu lệnh (thủ công hay tự động) hoặc viết ví dụ bằng tay.

**Fine-tune: BootstrapFinetune (BFT).** Tác giả mở rộng BFT của DSPy. Với chương trình Φ, BFT tự sinh **rất nhiều ví dụ cho mọi module** và gộp thành **một bộ dữ liệu** để fine-tune trọng số LM, với một **mục tiêu đa nhiệm ngầm** (mỗi "nhiệm vụ con" là vai trò của một module).

Công trình trước chỉ dùng BFT trong bối cảnh rất hẹp: trên HotPotQA, Khattab et al. (2024) train một model **T5-Large** bằng trace từ chương trình few-shot dùng **Llama2-13b**. Họ không xét việc để một LM **tự dạy chính nó** qua BFT, cũng không xét vai trò của BFRS trong chương trình đã fine-tune. Paper này tập trung vào việc để model **tự dạy và tự cải thiện**, và lần đầu đề xuất kết hợp BFRS và BFT bằng **xen kẽ** để cùng một LM tự dạy mình tốt hơn so với chỉ tối ưu prompt hoặc chỉ tối ưu trọng số. Ý tưởng tương tự có thể thử khi một **model lớn hơn bootstrap cho model nhỏ hơn**; cách đó có thể còn tốt hơn nhưng nằm ngoài phạm vi paper.

> 💡 **Giải thích.** Cùng một nguồn dữ liệu (trace đúng) được dùng theo hai cách:
> - BFRS: chọn **vài** trace làm **ví dụ trong prompt** (thay đổi Π);
> - BFT: dùng **tất cả** trace đúng làm **dữ liệu train LoRA** (thay đổi Θ).
>
> Π tốt hơn → nhiều trace đúng hơn → dữ liệu fine-tune tốt hơn. Đó là lý do Π đi trước Θ.

---

## 4. Experimental Evaluation (Thí nghiệm)

**Thiết lập chung.**
- Ba dataset, mỗi dataset một task và một LM program: **HotPotQA** (suy luận nhiều bước), **GSM8K** (suy luận số học), **Iris** (phân loại).
- Ba model: `mistral-7b-instruct-v0.2`, `llama-2-7b-chat`, `llama-3-8b-instruct`.
- Trong một thí nghiệm, **cùng một model** được dùng cho tối ưu prompt, bootstrap trace và fine-tune.
- Mọi module khởi tạo với **cùng trọng số LM** nhưng **prompt template khác nhau**, chuyên cho nhiệm vụ của module (sinh truy vấn, trả lời…).
- Mọi chương trình và optimizer được cài như phần mở rộng của DSPy.

**Chia dữ liệu và quy trình.**
- Mỗi dataset chia train, development, test. Tập train được xáo trộn trước mỗi lần tối ưu (prompt hoặc trọng số), theo random seed.
- **Tối ưu prompt**: lấy mẫu hai tập train và validation **không chồng nhau** từ tập train ban đầu, dùng BFRS để tối ưu prompt của từng module, **giữ nguyên trọng số**.
- **Tối ưu trọng số**: dùng **toàn bộ** ví dụ train làm ứng viên sinh dữ liệu, đưa vào BFT. BFT:
  1. chạy Φ trên mọi ví dụ train;
  2. giữ trace có output cuối đúng, bỏ phần còn lại;
  3. lấy các cặp **prompt–completion** ở mức module cho mỗi trace;
  4. tạo cặp mới bằng cách **thay prompt do Φ sinh ra bằng một prompt vanilla**;
  5. gộp mọi cặp prompt–completion ở mức module thành một bộ dữ liệu;
  6. fine-tune trọng số LM của Φ trên bộ dữ liệu này;
  7. trả về Φ mới với trọng số của mọi module là LM đã fine-tune.
- Fine-tune bằng **LoRA**.

> 💡 **Giải thích bước 4.** Khi fine-tune, input được đổi sang **prompt vanilla** (không có ví dụ few-shot), còn output vẫn là completion tốt sinh từ prompt đã tối ưu. Model học cách cho ra output tốt **ngay cả khi không có ví dụ**, tức là "nội hoá" kiến thức từ few-shot vào trọng số.

**Suy luận nhiều bước — HotPotQA** (thiết lập "fullwiki").
- Hệ thống phải tìm **hai đoạn tóm tắt Wikipedia** trong kho **5 triệu** đoạn, rồi trả lời một câu hỏi dạng sự kiện.
- Chương trình có **3 module LM**: hai module đầu sinh truy vấn tìm kiếm (mỗi module một "hop"), module cuối sinh câu trả lời. Mỗi module dùng **Chain-of-Thought** (sinh chuỗi suy luận trước truy vấn hoặc câu trả lời).
- Truy vấn được gửi tới retriever **ColBERTv2** đã freeze.
- Đo bằng **exact match** sau khi chuẩn hoá (chữ hoa/thường, khoảng trắng hai đầu, dấu câu).
- Chia dữ liệu: **1000 train**, **500 dev** (từ tập train gốc), **1500 test** (từ tập validation gốc, vì test gốc không công khai). Tối ưu prompt dùng **100 train** và **250 validation** không chồng nhau.

**Suy luận số học — GSM8K.**
- Chương trình **1 module** dùng CoT: LM sinh chuỗi suy luận rồi tới câu trả lời.
- Đo accuracy bằng cách **lấy số cuối cùng trong dòng đầu tiên** của câu trả lời, so với đáp án.
- **1000 train**, **500 dev** (từ train gốc), **1319 test** (toàn bộ test). Tối ưu prompt dùng **100 train** và **250 validation**.

**Phân loại — Iris.**
- Bài toán kinh điển phân loại loài hoa iris. Dùng chương trình DSPy CoT **1 module**.
- Mục đích: kiểm tra xem một task **phân loại dựa trên đặc trưng số** có làm các phương pháp hoàn toàn dựa trên gradient (fine-tune) chiếm ưu thế lớn không, tức là mở rộng giả thuyết sang bối cảnh khác.
- Đo bằng exact match sau chuẩn hoá. Iris có **150 mẫu**, chia đều thành train, dev, test. Tối ưu prompt dùng **15 train** và **35 validation**.

### Bảng 1 — Kết quả chính (accuracy %, trung bình 3 lần chạy với seed khác nhau, trên tập test)

| Chiến lược | Mistral HotPotQA | Mistral GSM8K | Mistral Iris | Llama-2 HotPotQA | Llama-2 GSM8K | Llama-2 Iris | Llama-3 HotPotQA | Llama-3 GSM8K | Llama-3 Iris |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla zero-shot | 17.2 | 40.3 | 26.0 | 13.2 | 24.0 | 0.0 | 31.6 | 72.7 | 48.0 |
| Π (chỉ prompt) | 33.8 | 46.4 | 57.3 | 33.3 | 26.0 | 56.7 | **46.9** | 77.9 | 79.3 |
| Θ (chỉ trọng số) | 22.9 | 40.7 | 29.3 | 12.2 | 24.0 | – | 34.8 | 75.1 | 37.3 |
| Π→Π | 33.8 | 47.7 | 59.3 | 32.6 | 24.7 | 64.0 | 46.5 | 77.6 | **82.0** |
| Θ→Θ | 24.0 | 42.8 | 38.0 | 13.0 | 24.1 | – | 34.4 | 44.1 | 39.3 |
| *— BetterTogether —* | | | | | | | | | |
| Π→Θ | 36.3 | 47.3 | 30.7 | 32.7 | **27.3** | 26.7 | 42.8 | 77.6 | 44.0 |
| Θ→Π | 33.0 | **48.3** | **66.7** | 34.2 | 26.6 | – | 43.6 | **78.9** | 78.7 |
| Π→Θ→Π | **37.6** | 46.8 | 52.7 | **34.8** | 26.3 | **65.3** | 46.7 | 77.0 | 79.3 |

Chú thích của paper:
- Các chiến lược bắt đầu bằng tối ưu trọng số dùng chiến lược vanilla (zero-shot) để sinh bộ dữ liệu fine-tune ban đầu.
- Nếu model sinh quá ít hoặc không có output đúng nào khi chạy vanilla trên tập train, sẽ không đủ dữ liệu để fine-tune. Các ô đó ghi "–" (vd. Llama-2 trên Iris: vanilla được 0.0%).

> 💡 **Đọc bảng.** Số in đậm là cao nhất trong cột (nhóm tự đánh dấu theo số liệu).
> - 7/9 cột có điểm cao nhất thuộc nhóm BetterTogether. Hai ngoại lệ đều là Llama-3: HotPotQA (Π: 46.9) và Iris (Π→Π: 82.0).
> - **Θ đơn lẻ gần như vô dụng** khi bắt đầu từ zero-shot: dữ liệu fine-tune sinh từ prompt yếu thì cũng yếu. Đây là bằng chứng cho giả thuyết "prompt trước giúp có dữ liệu fine-tune tốt hơn".
> - Llama-3 GSM8K với Θ→Θ chỉ còn 44.1 (từ 72.7): fine-tune lặp lại có thể làm **hỏng** model. Bảng phụ lục D.2 cho thấy kết quả này dao động mạnh giữa 3 lần chạy (52.2 / 36.6 / 43.4).

---

## 5. Results & Discussion (Kết quả và thảo luận)

Kết quả trên tập test ở Bảng 1 là trung bình ba lần chạy với random seed khác nhau; kết quả từng lần ở Phụ lục D.

Ở **7/9 cặp** dataset × LM, chiến lược tốt nhất là chiến lược **dùng cả bước prompt (Π) và bước trọng số (Θ)**, dù **không có người thắng rõ ràng** giữa các chiến lược BetterTogether. Nhìn chung, **tối ưu prompt là thiết yếu với mọi task**, còn tối ưu prompt và trọng số cùng nhau cho cải thiện mạnh so với cấu hình tốt nhất chỉ tối ưu một trong hai.

Tóm lại, tác giả đề xuất xen kẽ tối ưu prompt và fine-tune trọng số, và thử một số chiến lược làm vậy. Trên HotPotQA, GSM8K và Iris, các chiến lược này rất hiệu quả để một LM **tự dạy mình** thực thi một LM program qua bootstrap, cho mức tăng **5–78%** (HotPotQA), **2.5–10%** (GSM8K), **3.5–88%** (Iris).

---

## 6. Limitations (Hạn chế)

- Bằng chứng mạnh nhưng chỉ từ **9 trường hợp** (3 task × 3 LM); task, chương trình hoặc LM khác có thể thay đổi kết quả theo cách khó lường.
- Chỉ thử tối ưu trọng số dưới dạng **LoRA fine-tune** model pre-trained. Về nguyên tắc, có thể có một chiến lược fine-tune khác **đủ mạnh và rẻ** để **không cần** tối ưu prompt nữa.
- Tác giả **chưa hiểu vì sao** cả hai đều quan trọng. Vai trò của tối ưu prompt và fine-tune trong LM program nhiều bước đều còn mới. Việc thiếu hiểu biết có thể dẫn tới **tương tác không lường trước** giữa các thành phần, khác với gradient descent cho mạng neural vốn đã được nghiên cứu hàng chục năm.

---

## Phụ lục

### A. Các chương trình

**HotPotQA** (3 module CoT + retriever freeze; mỗi module có prompt và trọng số LM riêng):
```python
class HotPotQAProgram(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(2)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        for hop in range(2):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = dsp.utils.deduplicate(context + passages)
        return self.generate_answer(context=context, question=question).copy(context=context)
```

**GSM8K** (1 module):
```python
class CoTProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    def forward(self, question):
        return self.generate_answer(question=question)
```

**Iris** (1 module, signature dạng class, do nhóm DSPy cung cấp):
```python
class IrisSignature(dspy.Signature):
    "Given the petal and sepal dimensions in cm, predict the iris species."
    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc='setosa, versicolor, or virginica')

class IrisProgram(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought(IrisSignature)
    def forward(self, petal_length, petal_width, sepal_length, sepal_width):
        return self.generate_answer(petal_length=petal_length, petal_width=petal_width,
                                    sepal_length=sepal_length, sepal_width=sepal_width)
```

Code này dùng API DSPy năm 2024 (`dspy.Retrieve`, `dsp.utils`). Phiên bản DSPy mới có thể khác; xem lưu ý trong `docs/01`, mục 3.

**Prompt vanilla mà DSPy sinh ra** (Bảng A.1–A.3). Ví dụ với module `generate_query[0]`:
```
Given the fields `context`, `question`, produce the fields `search_query`.
---
Follow the following format.
Context: ${context}
Question: ${question}
Reasoning: Let's think step by step in order to ${produce the search_query}. We ...
Search Query: ${search_query}
---
Context: N/A
Question: What animal subject to the Foster's rule is the smallest North American deer?
Reasoning: Let's think step by step in order to ...
Search Query: North American deer ear surface area Foster's rule smallest species
```
Ở hop thứ hai, context chứa các đoạn văn tìm được (Key deer, North American least shrew, Wynnea americana, Pudú). Module trả lời đưa ra đáp án đúng "Key deer".

> 💡 **Dùng khi thuyết trình.** Ví dụ này rất hợp để chiếu: **chuỗi suy luận ở hop 1 sai** (model bịa rằng "Foster's rule" là công thức tính cân nặng theo diện tích tai), nhưng chương trình vẫn ra **đáp án đúng**. Nghĩa là trace này vẫn được giữ làm ví dụ hoặc dữ liệu train, vì metric chỉ nhìn đáp án cuối. Đây là một hạn chế đáng thảo luận của bootstrap.

### B. Thông tin tài nguyên
- Model (HuggingFace id): `mistralai/Mistral-7b-Instruct-v0.2` (Apache 2.0), `meta-llama/Llama-2-7b-chat-hf` (Llama 2 Community License), `meta-llama/Meta-Llama-3-8B-Instruct` (Llama 3 Community License).
- Dataset: HotPotQA (Apache 2.0), GSM8K (MIT), Iris (CC BY 4.0).
- Mọi LM dùng trong paper đều dành cho **tiếng Anh**.

### C. Chi tiết cài đặt (quan trọng để tái hiện)

| Mục | Giá trị |
|---|---|
| Suy luận | Host model trong Docker bằng HuggingFace **text-generation-inference** |
| Sampling | temperature **0.1**, paper ghi "top_k 0.97" (giá trị dạng xác suất, nhiều khả năng là top_p); dừng khi gặp chuỗi dừng hoặc đủ **1024 token** (tính cả prompt) |
| Tối ưu prompt | **BootstrapFewShotRS**: tìm ngẫu nhiên **6 candidate program**, tối đa **3 ví dụ few-shot** mỗi module |
| Fine-tune | **LoRA** trên layer **query và key** của self-attention; **rank 32, alpha 64, dropout 0** |
| Train | **5 epoch**, **bfloat16**, learning rate **1e-5**, batch size hiệu dụng **8** |
| Chi phí | Khoảng **75 giờ GPU A100** để ra Bảng 1 |

BetterTogether đương nhiên tốn hơn so với chỉ tối ưu prompt hoặc chỉ fine-tune. So sánh chi phí giữa hai bước tuỳ thuộc cấu hình và kích thước dữ liệu.

### D. Kết quả mở rộng
Bảng D.1–D.3 liệt kê từng lần chạy (Run 1–3) cho HotPotQA, GSM8K, Iris. Xem PDF khi cần phân tích độ dao động.
