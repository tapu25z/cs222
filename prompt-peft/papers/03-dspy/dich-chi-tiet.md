# DSPy — Dịch và diễn giải chi tiết

> Bản dịch có diễn giải của *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines* (Khattab et al.).
> PDF trong repo: [`dspy-arxiv-2310.03714v1.pdf`](dspy-arxiv-2310.03714v1.pdf). Đây là **bản arXiv v1 (10/2023, preprint)**. Bản chính thức ở **ICLR 2024** đổi tên thành *…into State-of-the-Art Pipelines*; số liệu có thể khác chút. Nếu trích dẫn số liệu lên slide, nên đối chiếu với bản ICLR.
> Ô **💡 Giải thích** là phần nhóm thêm vào. Code trong paper dùng API DSPy năm 2023; thư viện hiện tại đã đổi nhiều tên (xem `docs/01`, mục 3).

---

## Abstract

Cộng đồng ML đang khám phá nhanh các kỹ thuật prompt LM và **xếp chồng chúng thành pipeline** để giải các task phức tạp. Đáng tiếc, các pipeline LM hiện có thường được cài bằng **"prompt template" viết cứng**, tức các chuỗi dài tìm ra bằng **thử–sai**.

Để phát triển và tối ưu pipeline LM có hệ thống hơn, tác giả giới thiệu **DSPy**, một **mô hình lập trình** trừu tượng hoá pipeline LM thành **đồ thị biến đổi văn bản (text transformation graph)**: các đồ thị tính toán dạng mệnh lệnh (imperative), trong đó LM được gọi qua các **module khai báo (declarative)**.
- Module DSPy **được tham số hoá**: chúng có thể **học** (bằng cách tạo và thu thập ví dụ minh hoạ) cách áp dụng tổ hợp các kỹ thuật prompting, fine-tuning, augmentation và suy luận.
- Tác giả thiết kế một **compiler** tối ưu bất kỳ pipeline DSPy nào để tối đa một metric cho trước.

Hai case study cho thấy chương trình DSPy ngắn gọn có thể biểu diễn và tối ưu các pipeline LM phức tạp: suy luận toán có lời văn, truy hồi nhiều bước, trả lời câu hỏi phức tạp, điều khiển vòng lặp agent. **Chỉ vài phút compile**, vài dòng DSPy cho phép GPT-3.5 và llama2-13b-chat **tự bootstrap** pipeline vượt few-shot prompting thông thường (thường hơn **25%** và **65%**) và vượt pipeline dùng ví dụ do chuyên gia viết (hơn **5–46%** và **16–40%**). Hơn nữa, chương trình DSPy compile cho LM mở, tương đối nhỏ như **T5 770M** và llama2-13b-chat **cạnh tranh được** với cách dùng chuỗi prompt chuyên gia viết cho GPT-3.5 (độc quyền).

Code: https://github.com/stanfordnlp/dspy.

---

## 1. Introduction (Giới thiệu)

LM cho phép xây hệ thống NLP ở **mức trừu tượng cao hơn** và **cần ít dữ liệu hơn** bao giờ hết. Điều này thúc đẩy bùng nổ các kỹ thuật "prompting" và fine-tune nhẹ để thích nghi LM với task mới, gợi ra suy luận có hệ thống, và bổ sung nguồn truy hồi hoặc công cụ. Hầu hết các kỹ thuật này được nghiên cứu **riêng lẻ**, nhưng ngày càng nhiều người quan tâm tới việc xây **pipeline nhiều bước và agent**, chia task phức tạp thành các lời gọi LM dễ quản lý hơn.

Đáng tiếc, LM **nhạy cảm với cách prompt** cho từng task, và vấn đề càng nặng trong pipeline, nơi nhiều lời gọi LM phải phối hợp với nhau. Vì vậy lời gọi LM trong các pipeline và framework phổ biến thường được cài bằng **prompt template viết cứng**: các chuỗi dài gồm hướng dẫn và ví dụ, **viết tay qua thử–sai**. Tác giả cho rằng cách này, dù phổ biến, **giòn và khó mở rộng**, về bản chất giống như **chỉnh tay trọng số của một bộ phân loại**. Một chuỗi prompt có thể không khái quát sang pipeline khác, LM khác, miền dữ liệu khác, hay thậm chí input khác.

Để thiết kế pipeline AI có hệ thống hơn, tác giả giới thiệu mô hình lập trình **DSPy** (đọc là "đi-ét-pai"; là thế hệ thứ hai của framework Demonstrate–Search–Predict, DSP). DSPy đẩy việc xây pipeline LM **ra khỏi việc thao tác chuỗi tự do** và **tới gần lập trình** hơn: ghép các toán tử module để tạo đồ thị biến đổi văn bản, trong đó **compiler tự sinh** chiến lược gọi LM và prompt tối ưu từ chương trình.

Cảm hứng từ sự đồng thuận đã hình thành quanh các trừu tượng mạng neural:
1. nhiều **layer đa dụng** có thể ghép module vào bất kỳ kiến trúc phức tạp nào;
2. trọng số model được **train bằng optimizer** thay vì chỉnh tay.

Cụ thể:
- **Mô hình lập trình (mục 3).** Chuyển các kỹ thuật prompting dạng chuỗi, kể cả những kỹ thuật phức tạp như Chain of Thought và ReAct, thành **module khai báo** có **signature** kiểu ngôn ngữ tự nhiên. Module DSPy là thành phần thích nghi theo task, **giống layer của mạng neural**, trừu tượng hoá một phép biến đổi văn bản (trả lời câu hỏi, tóm tắt paper…). Mỗi module được **tham số hoá** để học hành vi mong muốn bằng cách **bootstrap ví dụ minh hoạ** hữu ích trong pipeline. Lấy cảm hứng trực tiếp từ PyTorch, module DSPy được dùng qua đồ thị tính toán **define-by-run**: pipeline được biểu diễn bằng (1) khai báo các module cần dùng và (2) dùng chúng trong bất kỳ luồng điều khiển nào (if, vòng lặp, exception…).
- **Compiler (mục 4).** Tối ưu bất kỳ chương trình DSPy nào để cải thiện chất lượng hoặc chi phí. Input của compiler: chương trình, **vài input train** (nhãn tuỳ chọn), và một **metric** để kiểm tra. Compiler **mô phỏng** các phiên bản của chương trình trên input và **bootstrap trace ví dụ** của từng module để tự cải thiện, dùng chúng để tạo **prompt few-shot** hiệu quả hoặc **fine-tune LM nhỏ** cho các bước của pipeline. Việc tối ưu do **teleprompter** đảm nhận: các chiến lược tối ưu đa dụng quyết định module học từ dữ liệu như thế nào. (Tên *tele-prompter* xuất phát từ ý tưởng tự động hoá việc prompt "từ xa", không cần can thiệp thủ công.)

Đánh giá tập trung vào **vai trò của prompt do chuyên gia viết**: tác giả muốn giảm hoặc loại bỏ vai trò đó bằng module DSPy và teleprompter. Hai case study lớn: **GSM8K** và **HotPotQA**, với CoT, phản tư đa chuỗi, truy hồi nhiều bước, hỏi đáp có truy hồi, và vòng lặp agent.

**Đóng góp.** Mô hình lập trình đầu tiên chuyển kỹ thuật prompting thành **module khai báo có tham số**, cùng một compiler hiệu quả với các chiến lược tối ưu đa dụng. Với DSPy, có thể viết chương trình rất ngắn để bootstrap hệ thống NLP nhiều bước tự cải thiện, dùng LM nhỏ tới llama2-13b-chat và T5-Large (770M). Không cần prompt viết tay, sau vài phút tới vài chục phút compile:

| | GPT-3.5 | llama2-13b-chat |
|---|---|---|
| GSM8K (mục 6) | 33% → **82%** | 9% → **47%** |
| HotPotQA (mục 7) | 32% → **46%** | 22% → **41%** |

---

## 2. Related Work (Công trình liên quan)

- **Cảm hứng.** Vai trò của Torch, Theano, Chainer trong deep learning: cung cấp trừu tượng mạnh. Một chuyển biến tương tự đang diễn ra với pipeline LM; tác giả gọi đó là **lập trình foundation model**. DSPy vay ý tưởng *differentiable programming* nhưng áp dụng cho lời gọi LM, và mượn cú pháp của PyTorch.
- **In-context learning** là cơ chế then chốt. Đặc biệt với instruction tuning, có thể gợi ra hành vi phức tạp chỉ bằng prompt. Các dạng giám sát yếu trước đây cần heuristic thủ công nay có thể do LM làm.
- **Công cụ và thư viện.** LangChain, Semantic Kernel, LlamaIndex cung cấp chain và agent đóng gói sẵn, nhưng **vẫn mắc vấn đề prompt engineering** mà DSPy giải quyết: hành vi theo task được biểu diễn bằng prompt template viết tay (Phụ lục B).
- **Tối ưu prompt.** Đã có nghiên cứu dùng tối ưu rời rạc hoặc RL để tìm prompt, nhưng thường cho **một lời gọi LM**. DSPy tổng quát hoá: tối ưu **pipeline bất kỳ** từ signature khai báo, bằng cách bootstrap ví dụ nhiều bước chất lượng cao có ràng buộc. Teleprompter có thể dùng chọn model (cross-validation), về nguyên tắc cả RL, phản hồi từ LM, hoặc tối ưu siêu tham số kiểu Bayes.

---

## 3. The DSPy Programming Model (Mô hình lập trình DSPy)

DSPy coi LM là **thiết bị sinh văn bản trừu tượng**, và tối ưu cách dùng chúng trong đồ thị tính toán bất kỳ. Chương trình DSPy viết bằng Python: nhận input của task (câu hỏi, paper…) và trả output (câu trả lời, bản tóm tắt…) sau một chuỗi bước.

DSPy đóng góp **ba trừu tượng**:
- **Signature**: trừu tượng hoá hành vi input/output của một module.
- **Module**: thay thế các kỹ thuật prompt thủ công, ghép được thành pipeline bất kỳ.
- **Teleprompter**: tối ưu mọi module trong pipeline để tối đa một metric.

(LM có thể là model prompt được, có in-context learning như GPT-3.5, Llama2-7b, hoặc model nhỏ hơn có thể fine-tune như T5-base.)

### 3.1 Signature ngôn ngữ tự nhiên trừu tượng hoá prompting và fine-tuning

Thay vì prompt dạng chuỗi tự do, chương trình DSPy dùng **signature ngôn ngữ tự nhiên** để giao việc cho LM. Signature là **khai báo kiểu của một hàm bằng ngôn ngữ tự nhiên**: một đặc tả khai báo ngắn cho DSPy biết phép biến đổi văn bản cần làm **gì** (vd. "nhận câu hỏi và trả câu trả lời"), chứ không nói một LM cụ thể **nên được prompt thế nào**.

Hình thức: signature là bộ gồm **trường input**, **trường output**, và **hướng dẫn tuỳ chọn**. Mỗi trường có tên và metadata tuỳ chọn. Thường thì DSPy **suy ra vai trò của trường từ tên trường**: compiler sẽ dùng in-context learning để hiểu `question` khác `answer`, và dần tinh chỉnh cách dùng các trường.

Hai lợi ích so với prompt:
1. Signature có thể được **compile thành prompt hoặc fine-tune tự cải thiện, thích nghi theo pipeline**, chủ yếu bằng cách bootstrap ví dụ minh hoạ cho từng signature.
2. Signature tự xử lý **định dạng và parse** có cấu trúc, giảm thao tác chuỗi dễ vỡ trong chương trình.

Ký hiệu ngắn `question -> answer`: dòng 1 dưới đây là một chương trình DSPy hoàn chỉnh cho hệ hỏi đáp cơ bản:
```python
qa = dspy.Predict("question -> answer")
qa(question="Where is Guaraní spoken?")
# Out: Prediction(answer='Guaraní is spoken mainly in South America.')
```
DSPy parse ký hiệu này và mở rộng tên trường thành hướng dẫn có nghĩa cho LM. Ví dụ `english document -> french translation` sẽ prompt LM dịch Anh–Pháp. Khi cần, có giao diện nâng cao để ràng buộc rõ hơn (Phụ lục A).

### 3.2 Module có tham số và template trừu tượng hoá kỹ thuật prompting

Giống chữ ký kiểu trong ngôn ngữ lập trình, signature chỉ định nghĩa **giao diện**. Để dùng một signature, phải khai báo một **module** với signature đó; khai báo này trả về một hàm có signature đó.

**Module `Predict`.** Module cốt lõi. Bên trong, Predict lưu:
- signature;
- LM tuỳ chọn (mặc định None, nếu đặt thì ghi đè LM mặc định cho module này);
- **danh sách ví dụ minh hoạ** dùng để prompt (ban đầu rỗng).

Giống layer PyTorch, module được gọi như hàm: nhận keyword argument theo các trường input, **định dạng prompt** cho signature (kèm ví dụ phù hợp), gọi LM, và **parse các trường output**. Khi ở **chế độ compile**, Predict còn **ghi lại trace input/output** để teleprompter bootstrap ví dụ.

**Các module có sẵn khác.** `ChainOfThought`, `ProgramOfThought`, `MultiChainComparison`, `ReAct`. Chúng đều dùng được thay thế lẫn nhau cho bất kỳ signature nào. Chỉ cần đổi `Predict` thành `ChainOfThought` trong chương trình trên là có hệ thống **suy nghĩ từng bước** trước khi đưa ra output.

Các module này chỉ vài dòng code: mở rộng signature người dùng và gọi Predict một hoặc nhiều lần. Ví dụ ChainOfThought rút gọn:
```python
class ChainOfThought(dspy.Module):
    def __init__(self, signature):
        # Đổi signature từ `*inputs -> *outputs` thành `*inputs -> rationale, *outputs`.
        rationale_field = dspy.OutputField(prefix="Reasoning: Let's think step by step.")
        signature = dspy.Signature(signature).prepend_output_field(rationale_field)
        # Khai báo một sub-module với signature đã sửa.
        self.predict = dspy.Predict(signature)

    def forward(self, **kwargs):
        # Chuyển tiếp input cho sub-module.
        return self.predict(**kwargs)
```
Đây là một module đầy đủ, có thể học prompt few-shot hiệu quả cho bất kỳ LM hay task nào. Đối lập với các prompt suy luận dài viết tay trong Phụ lục C.

**Tham số hoá.** Điểm độc đáo: DSPy **tham số hoá** các kỹ thuật prompting. Bất kỳ lời gọi LM nào hiện thực một signature đều phải chỉ định các tham số:
1. LM cụ thể sẽ gọi;
2. câu hướng dẫn trong prompt và tiền tố của từng trường;
3. **quan trọng nhất: các ví dụ minh hoạ**, dùng làm prompt few-shot (LM freeze) hoặc làm dữ liệu train (fine-tune).

Paper tập trung chủ yếu vào việc **tự sinh và chọn ví dụ minh hoạ**. Bootstrap ví dụ tốt là cách mạnh để dạy các pipeline LM phức tạp hành vi mới một cách có hệ thống.

**Công cụ.** Module thực thi tính toán, ví dụ `dspy.Retrieve` (hỗ trợ ColBERTv2, Pyserini, Pinecone), thử nghiệm `dspy.SQL` và `dspy.PythonInterpreter` (chạy Python trong sandbox).

**Chương trình.** Module được ghép thành pipeline theo kiểu **define-by-run**: khai báo module trong `__init__` (để DSPy theo dõi khi tối ưu), rồi viết pipeline bằng code bất kỳ trong `forward`. Ví dụ RAG hoàn chỉnh:
```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)
```
Gọi `RAG()("Where is Guaraní spoken?")` để dùng. Đổi signature thành `"context, question -> search_query"` thì có hệ thống **sinh truy vấn tìm kiếm** thay vì câu trả lời.

### 3.3 Teleprompter tự động hoá prompting cho pipeline bất kỳ

Khi compile chương trình DSPy, ta gọi một **teleprompter**: optimizer nhận **chương trình, tập train, metric** và trả về **chương trình mới đã tối ưu**.

- Tập train có thể **rất nhỏ** (vài ví dụ), dù nhiều dữ liệu hơn thì tối ưu mạnh hơn.
- Ví dụ train có thể **không đầy đủ**: chỉ cần input. **Không cần nhãn cho các bước trong pipeline**, trừ khi metric cần. Thường chỉ giả định có nhãn cho **output cuối**. Điều này then chốt cho tính module: xây pipeline mới chỉ cần **compile lại**, không cần gán nhãn dữ liệu riêng cho pipeline mới.
- Metric có thể đơn giản (exact match, F1), hoặc là **cả một chương trình DSPy** cân bằng nhiều tiêu chí.

Ví dụ compile RAG với tập hỏi–đáp và metric exact match:
```python
# Tập train nhỏ, chỉ có câu hỏi và đáp án cuối.
qa_trainset = [dspy.Example(question="What is the capital of France?", answer="Paris")]

# Teleprompter sẽ bootstrap các nhãn thiếu: chuỗi suy luận và context truy hồi.
teleprompter = dspy.BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)
compiled_rag = teleprompter.compile(RAG(), trainset=qa_trainset)
```
`BootstrapFewShot` mô phỏng RAG trên ví dụ train, thu ví dụ minh hoạ của từng module (cặp input–output) mà **cùng nhau dẫn tới output hợp lệ** (đúng signature và metric).

Metric tuỳ biến, ví dụ buộc câu trả lời phải xuất hiện trong context đã truy hồi:
```python
def answer_and_context_match(example, pred, trace=None):
    answer_match = dspy.evaluate.answer_exact_match(example, pred)
    # Câu trả lời có phải chuỗi con của một đoạn văn nào không?
    context_match = any((pred.answer.lower() in c) for c in pred.context)
    return answer_match and context_match
```

**Ghép teleprompter qua chương trình *teacher*.** DSPy lấy ví dụ từ chương trình teacher. Như vậy chương trình đắt (vd. ensemble dùng LM lớn) có thể **giám sát** chương trình rẻ (pipeline đơn giản dùng LM nhỏ). Ví dụ: lấy `compiled_rag` (dùng Llama2-13b-chat) làm teacher để **fine-tune Flan-T5-large**:
```python
# Tập câu hỏi lớn hơn, *không có nhãn*. Nhãn cho mọi bước sẽ được bootstrap.
unlabeled_questions = [dspy.Example(question="What is the capital of Germany?"), ...]

# Vì không có đáp án, dùng `answer_passage_match` để lọc câu trả lời không có căn cứ.
finetuning_teleprompter = BootstrapFinetune(metric=dspy.evaluate.answer_passage_match)

compiled_rag_via_finetune = finetuning_teleprompter.compile(
    RAG(), teacher=compiled_rag, trainset=unlabeled_questions, target='google/flan-t5-large')
```

> 💡 **Liên hệ BetterTogether.** Ở đây teacher (Llama2-13b) và student (T5) là **hai model khác nhau**. BetterTogether hỏi: nếu teacher và student là **cùng một model** thì sao? Đó là "LM tự dạy chính nó".

---

## 4. The DSPy Compiler (Compiler của DSPy)

Sức mạnh then chốt của DSPy là khả năng **compile**, tức tự động tối ưu, bất kỳ chương trình nào. Compile dựa vào một teleprompter, cải thiện chất lượng (hoặc chi phí) của module qua **prompting hoặc fine-tuning**, hai thứ được **hợp nhất** trong DSPy. Teleprompter điển hình đi qua **ba giai đoạn**:

**Giai đoạn 1 — Sinh ứng viên (Candidate Generation).** Compiler tìm (đệ quy) mọi module `Predict` duy nhất (*predictor*) trong chương trình, kể cả lồng trong module khác. Với mỗi predictor, teleprompter sinh **giá trị ứng viên** cho các tham số: câu hướng dẫn, mô tả trường, hoặc **quan trọng nhất là ví dụ minh hoạ** (cặp input–output). Phiên bản này tập trung vào ví dụ minh hoạ, và cách làm đơn giản **kiểu rejection sampling** đã giúp bootstrap hệ thống nhiều bước rất hiệu quả.

Teleprompter đơn giản nhất, `BootstrapFewShot`:
- chạy một **chương trình teacher** (nếu không đặt thì là bản zero-shot của chính chương trình) trên một số input train, có thể nhiều lần với temperature cao;
- ở chế độ compile, **trace nhiều bước** được ghi lại tự động (thread-safe);
- dùng **metric** để lọc các trace nhiều bước giúp pipeline qua được metric;
- như vậy có **nhãn tiềm năng cho mọi signature**: bỏ ví dụ xấu, giữ ví dụ tốt làm ví dụ minh hoạ.

LM có thể rất thiếu ổn định, nhưng lại khá hiệu quả khi **tìm kiếm trong không gian lời giải** cho thiết kế nhiều bước. Một chương trình được chia nhỏ tốt thường tìm được ít nhất vài ví dụ train mà LM vượt qua ràng buộc của signature và metric, nên có thể bootstrap lặp lại nếu cần.

**Giai đoạn 2 — Tối ưu tham số (Parameter Optimization).** Mỗi tham số giờ có một tập ứng viên rời rạc (ví dụ minh hoạ, hướng dẫn…). Có thể dùng các thuật toán tối ưu siêu tham số (random search, Tree-structured Parzen Estimator như HyperOpt và Optuna) để chọn. Phiên bản rút gọn: `BootstrapFewShotWithRandomSearch` và `BootstrapFewShotWithOptuna` (Phụ lục E).

Một kiểu tối ưu khác là **fine-tune** với `BootstrapFinetune`: ví dụ minh hoạ được dùng để **cập nhật trọng số LM** của từng predictor. Thường tối ưu chất lượng trung bình theo metric bằng cross-validation trên tập train hoặc tập validation; áp dụng được kể cả khi **không có nhãn cho bước nào**, tuỳ vào metric.

**Giai đoạn 3 — Tối ưu chương trình bậc cao (Higher-Order Program Optimization).** Sửa **luồng điều khiển** của chương trình. Đơn giản nhất là **ensemble** (dùng trong các case study): bootstrap nhiều bản sao của cùng chương trình, rồi thay bằng một chương trình chạy song song tất cả và gộp dự đoán bằng hàm tuỳ chọn (vd. bỏ phiếu đa số). Sau này có thể mở rộng sang bootstrap lúc test và logic kiểu quay lui.

> 💡 **So sánh với huấn luyện mạng neural.**
> - Giai đoạn 1 ≈ tạo dữ liệu (nhưng dữ liệu tự sinh);
> - Giai đoạn 2 ≈ train (chọn ví dụ hoặc cập nhật trọng số);
> - Giai đoạn 3 ≈ thay đổi kiến trúc (ensemble).

---

## 5. Goals of Evaluation (Mục tiêu đánh giá)

Framework lập trình có thể được đánh giá theo nhiều chiều: hiệu quả tính toán, hiệu quả phát triển, độ trực quan của code và khái niệm… Paper tập trung vào vấn đề cấp bách nhất: **vai trò của prompt viết tay riêng cho task** trong việc đạt hệ thống tốt. Ba giả thuyết:

- **H1.** Với DSPy, có thể thay chuỗi prompt viết tay bằng module ngắn gọn, định nghĩa rõ, **không giảm** chất lượng hay khả năng biểu đạt.
- **H2.** Tham số hoá module và coi prompting là bài toán tối ưu giúp DSPy **thích nghi tốt hơn với LM khác nhau**, và có thể **vượt prompt chuyên gia**.
- **H3.** Tính module giúp khám phá kỹ hơn các pipeline phức tạp có đặc tính hữu ích hoặc phù hợp với metric tinh tế.

Tác giả mong chuyển câu hỏi mơ hồ "các LM so với nhau thế nào trên GSM8K" sang câu hỏi rõ ràng và tái hiện được: "chúng so với nhau thế nào trên GSM8K **với chương trình P, compile bằng chiến lược S**". Mục tiêu cuối: giảm vai trò của việc **viết prompt khéo léo**, thay bằng phát triển **chương trình module và optimizer** ghép được.

---

## 6. Case Study: Math Word Problems (GSM8K)

**Dữ liệu.** Lấy **200** cặp hỏi–đáp cho train và **300** cho dev từ tập train chính thức; đánh giá cuối trên **1.3k** ví dụ test chính thức. So sánh chủ yếu trên dev để tránh overfit tập test. Đo accuracy của **giá trị số cuối cùng** trong output.

**Ba chương trình.**
```python
vanilla = dspy.Predict("question -> answer")          # 1 bước
CoT = dspy.ChainOfThought("question -> answer")       # 2 bước (suy luận rồi trả lời)

class ThoughtReflection(dspy.Module):                 # nhiều bước
    def __init__(self, num_attempts):
        self.predict = dspy.ChainOfThought("question -> answer", n=num_attempts)
        self.compare = dspy.MultiChainComparison('question -> answer', M=num_attempts)

    def forward(self, question):
        completions = self.predict(question=question).completions
        return self.compare(question=question, completions=completions)

reflection = ThoughtReflection(num_attempts=5)
```
Trong `reflection`, LM sinh **5 chuỗi suy luận** (kèm đáp án), rồi module `MultiChainComparison` so sánh chúng và sinh đáp án mới dựa trên mẫu hình của 5 lần thử. Không module nào **riêng cho toán** hay riêng cho một LM.

**Các chiến lược compile.**
- `none`: zero-shot, không compile.
- `fewshot`: `dspy.LabeledFewShot(k=8)` lấy ngẫu nhiên 8 ví dụ (chỉ có câu hỏi và đáp án, không có suy luận). Báo cáo trung bình 3–5 lần chạy.
- `bootstrap`: `BootstrapFewShotWithRandomSearch(metric=gsm8k_accuracy)`, tự sinh chuỗi suy luận cho ví dụ train và chọn tổ hợp ví dụ bằng random search.
- `bootstrap×2`: bootstrap **lồng nhau**, dùng chương trình `bootstrap` đã tối ưu làm **teacher** để bootstrap tiếp. Hữu ích khi chương trình zero-shot ban đầu quá yếu.
- `+ensemble`: ensemble **top-7** chương trình ứng viên từ lần bootstrap, bỏ phiếu đa số.
- `+human CoT`: tập train có thêm chuỗi suy luận do người viết (GSM8K có sẵn).

Compile thường mất **vài phút tới vài chục phút**: kể cả cấu hình đắt nhất cũng chỉ chạy chương trình vài nghìn lần (vd. 10–20 lượt thử trên 150–300 ví dụ validation), và chạy song song được.

### Bảng 1 — GSM8K (accuracy %)

| Chương trình | Compile | Train | GPT-3.5 Dev | GPT-3.5 Test | Llama2-13b Dev | Llama2-13b Test |
|---|---|---|---|---|---|---|
| vanilla | none | n/a | 24.0 | 25.2 | 7.0 | 9.4 |
| | fewshot | trainset | 33.1 | – | 4.3 | – |
| | bootstrap | trainset | 44.0 | – | 28.0 | – |
| | bootstrap×2 | trainset | 64.7 | 61.7 | 37.3 | 36.5 |
| | +ensemble | trainset | 62.7 | 61.9 | 39.0 | 34.6 |
| CoT | none | n/a | 50.0 | – | 26.7 | – |
| | fewshot | trainset | 63.0 | – | 27.3 | – |
| | fewshot | +human CoT | 78.6 | 72.4 | 34.3 | 33.7 |
| | bootstrap | trainset | 80.3 | 72.9 | 43.3 | – |
| | +ensemble | trainset | **88.3** | **81.6** | 43.7 | – |
| reflection | none | n/a | 65.0 | – | 36.7 | – |
| | fewshot | trainset | 71.7 | – | 36.3 | – |
| | bootstrap | trainset | 83.0 | 76.0 | 44.3 | 40.2 |
| | +ensemble | trainset | 86.7 | – | **49.0** | **46.9** |

**Kết quả.**
- **vanilla**: GPT-3.5 và llama2-13b-chat gặp khó khi phải trả lời thẳng không suy luận, nhất là khi không có ví dụ tốt (none, fewshot). Nhưng vanilla được cải thiện mạnh nhờ `bootstrap` và `bootstrap×2`. Xem prompt được bootstrap (Phụ lục F), thấy prompt cho phép LM **dùng trường answer để suy luận trước**, điều được phép vì metric chỉ lấy số cuối cùng.
- **CoT**: chuỗi suy luận của người (+human CoT) giúp rất nhiều, nhưng `bootstrap` **bằng hoặc vượt** nó, ủng hộ giả thuyết DSPy giảm được nhu cầu prompt viết tay.
- **reflection** chỉ dài hơn vài dòng nhưng thắng rõ, dù CoT cũng rất mạnh khi có ensemble.
- Tổng thể, bootstrap tăng mạnh cho **mọi chương trình, với cả hai LM**. Các chương trình đều chỉ ghép 2–4 module và teleprompter. Trong mô hình mới của DSPy, chính việc **ghép đúng module đa dụng**, chứ không phải chỉnh chuỗi prompt, đã nâng các LM từ **4–20%** lên **49–88%**.

**So sánh không chính thức** với các công trình khác:
- Zhang et al. (2022): 48% với text-davinci-002 (gần kết quả llama2-13b-chat của DSPy); 59.4% với codex dùng CoT thủ công, 62.8% với CoT tự động.
- Wang et al. (2022b): 57% với CoT trên PaLM-540B, 74% khi thêm self-consistency.
- Tác giả Llama2: 28.7% (13b), 42.2% (34b), 56.8% (70b). Chương trình DSPy với bản 13b **cạnh tranh với kết quả 34b** dù không dùng suy luận của người.
- Zhao et al. (2023b): 80.8% với CoT trên gpt-3.5-turbo (04/2023).
- Tác giả GPT-4: GPT-3.5 đạt 57.1%, GPT-4 đạt 92%, nhưng GPT-4 đã được pre-train trên một phần tập train GSM8K.

---

## 7. Case Study: Complex Question Answering (HotPotQA)

**Dữ liệu.** HotPotQA thiết lập mở "fullwiki". Truy hồi trên index **Wikipedia 2017 "abstracts"** chính thức của HotPotQA, bằng **ColBERTv2**.
- Tập test của HotPotQA bị ẩn, nên dùng tập validation chính thức làm test (lấy 1000 ví dụ).
- Chia tập train thành 70%/30% train/validation, chỉ giữ ví dụ "hard" (khớp với tập validation/test chính thức).
- Lấy **200** ví dụ train và **300** ví dụ dev.

**Các chương trình.**
- `vanilla`: như ở GSM8K; signature `question -> answer` đủ tổng quát.
- `CoT RAG`: chương trình RAG ở mục 3.2, dùng ChainOfThought. Không giỏi HotPotQA, nên thử thêm hai chương trình nhiều bước.
- `react`: agent dùng công cụ, có sẵn trong DSPy:
  ```python
  react = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)], max_iters=5)
  ```
- `multihop`: mô phỏng luồng thông tin của Baleen và IRRR, giống IRCoT:
  ```python
  class BasicMultiHop(dspy.Module):
      def __init__(self, passages_per_hop):
          self.retrieve = dspy.Retrieve(k=passages_per_hop)
          self.generate_query = dspy.ChainOfThought("context, question -> search_query")
          self.generate_answer = dspy.ChainOfThought("context, question -> answer")

      def forward(self, question):
          context = []
          for hop in range(2):
              query = self.generate_query(context=context, question=question).search_query
              context += self.retrieve(query).passages
          return self.generate_answer(context=context, question=question)

  multihop = BasicMultiHop(passages_per_hop=3)
  ```

**Compile.** Dùng lại các compiler của GSM8K, thêm hai cách ghép:
- với `react`: bootstrap bằng `BootstrapFewShotWithRandomSearch` **bắt đầu từ** một lần bootstrap trước của ReAct;
- với `multihop`: **fine-tune T5-Large** bắt đầu từ bản bootstrap trước:
  ```python
  multihop_t5 = dspy.BootstrapFinetune(metric=answer_exact_match).compile(
      program, teacher=bootstrap, trainset=trainset, target='t5-large')
  ```

### Bảng 2 — HotPotQA (Ans = exact match câu trả lời, Psg = độ chính xác truy hồi cặp đoạn văn)

| Chương trình | Compile | GPT-3.5 Dev Ans | Dev Psg | Test Ans | Test Psg | Llama2-13b Dev Ans | Dev Psg | Test Ans | Test Psg |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | fewshot | 34.3 | n/a | 31.5 | n/a | 27.5 | n/a | 21.8 | n/a |
| CoT RAG | fewshot | 36.4 | 36.0 | 29.8 | 34.4 | 34.5 | 36.0 | 28.0 | 34.4 |
| | bootstrap | 42.3 | 36.0 | – | – | 38.3 | 36.0 | 32.9 | 34.4 |
| react | none | 20.3 | – | – | – | 20.0 | – | – | – |
| | +human r | 33.0 | – | – | – | 28.3 | – | – | – |
| | bootstrap | 31.0 | – | – | – | 24.7 | – | – | – |
| | bootstrap×2 | 39.0 | – | – | – | 40.0 | – | – | – |
| multihop | fewshot | 36.9 | 38.3 | 31.2 | 40.8 | 34.7 | 32.0 | 31.3 | 30.8 |
| | bootstrap | 48.7 | 47.0 | 39.6 | 43.8 | 42.0 | 48.3 | 36.4 | 43.5 |
| | ensemble | **54.7** | – | **45.6\*** | – | **50.0** | – | **41.0** | – |

\* Đánh giá trên 50% tập test vì chi phí. Với HotPotQA, cross-validation dùng trực tiếp tập train (không dùng dev).

**Kết quả.**
- So với few-shot vanilla, `CoT RAG` tự bootstrap tăng EM đáng kể. Nhưng nó phụ thuộc hoàn toàn vào việc ColBERTv2 tìm đúng đoạn văn từ **câu hỏi gốc**, nên recall bị hạn chế.
- `react` và `multihop` khắc phục bằng cách **sinh truy vấn qua nhiều "hop"**. Chương trình `multihop` đơn giản **tốt nhất**, và bootstrap lại rất hiệu quả so với fewshot ở cả hai LM.
- `bootstrap` (và/hoặc `bootstrap×2`) vượt cả few-shot prompting (với multihop) lẫn **suy luận do người viết** (với react, chỉnh nhẹ từ Yao et al. 2022).
- Quan trọng nhất: chỉ cần compile, **llama2-13b-chat cạnh tranh được với GPT-3.5**.

**Fine-tune T5-Large.** Chương trình `multihop_t5` cho ra một model **T5-Large (770M)** đạt **39.3% EM** và **46.0% passage accuracy** trên dev, chỉ dùng **200 input có nhãn** và **800 câu hỏi không nhãn**. Teacher là ensemble (hợp) của hai chương trình `multihop` dùng llama2-13b-chat. Vì cực nhỏ và chạy local được, chương trình này rẻ hơn **nhiều bậc** khi suy luận so với LM độc quyền như GPT-3.5.

**So sánh không chính thức** (phương pháp đánh giá và mẫu test khác nhau giữa các nghiên cứu):
- Si et al. (2022): 25.2% EM với CoT.
- Sun et al. (2022): 26.5% EM với "recite-and-answer" trên PaLM-62B.
- Wang et al. (2022a): 33.8% EM, 44.6% F1 với self-consistency trên PaLM-540B.
- Yao et al. (2022): ReAct 27.4% (PaLM-540B), 30.8% (text-davinci-002) với công cụ tìm kiếm Wikipedia API; 35.1% khi thêm CoT với self-consistency.
- Trivedi et al. (2022): 49% với pipeline dùng code-davinci-002 trên 500 câu hỏi.

---

## 8. Conclusion (Kết luận)

Paper giới thiệu DSPy, một mô hình lập trình mới để thiết kế hệ thống AI bằng pipeline của LM pre-trained và các công cụ khác. Ba khái niệm mới: **signature, module, teleprompter**. Hai case study rất khác nhau cho thấy DSPy hỗ trợ phát triển nhanh các hệ thống hiệu quả cao dùng **LM tương đối nhỏ**.

Tác giả đã duy trì bản mã nguồn mở gần một năm và thấy nhiều chương trình được compile thành hệ thống chất lượng cao, từ trích xuất thông tin tới sinh dữ liệu tổng hợp ít tài nguyên (để dành báo cáo có kiểm soát cho công trình sau). In-context learning đã thay đổi nghiên cứu LM trong 2–3 năm qua, nhưng tác giả cho rằng **sức mạnh biểu đạt thực sự** nằm ở việc xây **đồ thị biến đổi văn bản** phức tạp, nơi module ghép được và optimizer (teleprompter) kết hợp để dùng LM một cách **có hệ thống và đáng tin cậy** hơn.

---

## Phụ lục (tóm tắt)

**A. Signature nâng cao.** Signature có thể viết thành class Python để có hướng dẫn và mô tả trường rõ ràng:
```python
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField(dtype=dspy.SearchQuery)
```
Tính năng đang phát triển (lúc đó): khai báo kiểu cho trường output (`bool`, `int`, `float`, `list`, `dict`), vd. `contexts, question -> answer_found: bool`.

**B. So sánh với LangChain và LlamaIndex.**
- Hai thư viện này giúp lập trình viên ứng dụng bằng thành phần và chain **đóng gói sẵn** (agent, pipeline truy hồi, kết nối cơ sở dữ liệu, bộ nhớ…), nhưng **bên trong vẫn dựa vào prompt viết tay**.
- Khảo sát cuối 09/2023: mã nguồn LangChain có **50 chuỗi dài hơn 1000 ký tự** (chủ yếu là prompt), 12 file `prompts.py` và 42 file `prompt.py`. **DSPy không có prompt viết tay nào** cho bất kỳ task nào.
- Ví dụ: chain Program-Aided Language Model của LangChain dùng template **3982 ký tự** với 8 bài toán mẫu. Prompt SQL cho từng hệ cơ sở dữ liệu trung bình **1058 ký tự**. QA with sources trung bình 1337, Graph QA trung bình 722 ký tự.

**C. Ví dụ prompt lớn.** Liệt kê các prompt dài lấy từ paper và thư viện khác, kèm số từ và số ký tự.

**D. Module.** Pseudo-code rút gọn của `Predict` (D.1) và `ChainOfThought` (D.2).

**E. Teleprompter.** Pseudo-code rút gọn:
- **E.1 `BootstrapFewShot`**: ánh xạ predictor của student và teacher; với từng ví dụ train, chạy teacher ở chế độ compile, lấy trace mọi lời gọi `Predict`; nếu metric đạt thì thêm từng cặp (input, output) làm ví dụ minh hoạ cho predictor tương ứng của student.
  ```python
  for example in trainset:
      with dspy.setting.context(compiling=True):
          prediction = teacher(**example.inputs())
          predicted_traces = dspy.settings.trace
      if self.metric(example, prediction, predicted_traces):
          for predictor, inputs, outputs in predicted_traces:
              d = dspy.Example(automated=True, **inputs, **outputs)
              compiled_program[predictor2name[id(predictor)]].demonstrations.append(d)
  ```
- **E.2 `BootstrapFewShotWithRandomSearch`**: lặp `trials` lần (mặc định 16 trong pseudo-code), mỗi lần xáo trộn tập train theo seed, chạy BootstrapFewShot với số ví dụ ngẫu nhiên, chấm điểm trên valset; trả về chương trình điểm cao nhất.
- **E.3 `BootstrapFewShotWithOptuna`**: dùng Optuna để chọn ví dụ cho từng predictor thay vì random search.

**F. Ví dụ prompt do DSPy tự sinh.** Các prompt sau khi compile cho GSM8K và HotPotQA (xem PDF). Hữu ích để chiếu so sánh "trước và sau compile" khi thuyết trình.
