# Toolformer: Language Models Can Teach Themselves to Use Tools — Summary

**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (Meta AI Research; †Universitat Pompeu Fabra)
**Published:** arXiv:2302.04761, February 2023

---

## 1. What is this paper doing? (Goal & Motivation)

Large language models are great at few-shot/zero-shot tasks but paradoxically **struggle with basic things** that much smaller systems do well:

- arithmetic and precise calculation,
- factual lookup (and a tendency to hallucinate),
- awareness of time / up-to-date information,
- low-resource languages.

Giving LMs access to **external tools** (calculators, search engines, etc.) could fix this, but prior approaches either (a) need large amounts of human annotation, or (b) are hard-coded for one specific task, limiting generality.

**Toolformer's idea:** a language model can **teach itself** to use tools in a **self-supervised** way, needing only a handful of human-written demonstrations per API. The model learns:

- *which* tool to use,
- *when* to use it,
- *what arguments* to pass,
- *how* to incorporate the result into text generation.

The base model is **GPT-J (6.7B parameters)**, and the five tools used are:

| Tool | Implementation | Purpose |
|---|---|---|
| Question Answering | Atlas (retrieval-augmented LM, finetuned on Natural Questions) | factual lookup |
| Calculator | simple 4-operation calculator, results rounded to 2 decimals | arithmetic |
| Wikipedia Search | BM25 retriever over the KILT Wikipedia dump | richer factual context |
| Machine Translation | NLLB 600M (200 languages), fastText for language detection | multilingual text → English |
| Calendar | returns the current date (no input) | temporal awareness |

API calls are inserted inline into text as linearized strings, e.g.:
`Out of 1400 participants, 400 (or [Calculator(400 / 1400) → 0.29] 29%) passed the test.`
(In practice `[ ]`, and `->` are used instead of `<API>`, `</API>`, `→` so no vocabulary change is needed.)

## 2. Method

The approach converts a plain-text dataset **C** (a subset of CCNet) into an augmented dataset **C\*** containing API calls, in **four steps**:

### Step 1 — Sampling API calls
For each API, a prompt `P(x)` is hand-written with a few examples of API calls. For an input text `x = x₁...xₙ`, the model computes the probability of starting an API call at every position `i`:

`pᵢ = p_M(<API> | P(x), x₁:ᵢ₋₁)`

Positions with `pᵢ > τₛ` are kept (top-k max); then up to *m* candidate calls per position are sampled by continuing from `[P(x), x₁...xᵢ₋₁, <API>]`.

### Step 2 — Executing API calls
All generated calls are executed with the real tools, producing a text result `rᵢ` per call.

### Step 3 — Filtering API calls (the key self-supervised signal)
An API call is only kept if it **actually helps the model predict the future tokens** of the text. Using a weighted cross-entropy loss `Lᵢ(z)` over tokens `xᵢ...xₙ` given prefix `z`:

- `L⁺ᵢ = Lᵢ(e(cᵢ, rᵢ))` — loss when the API call **and its result** are provided,
- `L⁻ᵢ = min(Lᵢ(ε), Lᵢ(e(cᵢ, ε)))` — the better of *(i)* no call at all or *(ii)* a call **without** its result.

A call is kept only if:

**`L⁻ᵢ − L⁺ᵢ ≥ τf`** (the call reduces loss by at least threshold `τf`).

A weighting function `w̃ₜ = max(0, 1 − 0.2·t)` favors calls placed close to where their information is needed. The thresholds τₛ/τf are tuned per tool.

### Step 4 — Finetuning
Surviving calls are interleaved back into the original text, giving `C*`. Because `C*` contains the **same original text** (only with API calls inserted), finetuning on it does not distort the model's language modeling distribution. GPT-J is finetuned with a standard LM objective (batch size 128, lr 1e-5, 10% linear warmup).

### Inference
Regular greedy decoding until the model emits `→` (meaning it expects an API response); decoding is paused, the API is called, the result + `</API>` is inserted, and generation continues. Two tricks: allow an API call whenever `<API>` is among the **top-k tokens (k=10)** rather than the single most likely token, and allow **at most one API call per input** to avoid loops.

## 3. Evaluation

**Setup:** prompted **zero-shot** (task described in natural language, *no* in-context examples — deliberately harder than prior few-shot tool-use work).

**Baselines:**
- GPT-J (6.7B) — raw base model
- GPT-J + CC — GPT-J finetuned on plain CCNet subset C (no API calls)
- Toolformer (disabled) — the trained model with API calls turned off
- OPT (66B) and GPT-3/davinci (175B) — models ~10×/25× larger

**Benchmarks:**
| Task family | Datasets | Expected tool | Fairness notes |
|---|---|---|---|
| Fact completion | LAMA (SQuAD, Google-RE, T-REx) | QA | Wikipedia Search disabled (LAMA is built from Wikipedia) |
| Math word problems | ASDiv, SVAMP, MAWPS | Calculator | first predicted number checked |
| Open-domain QA | WebQuestions, Natural Questions, TriviaQA | Wikipedia Search | QA tool disabled (it was trained on NQ) |
| Multilingual QA | MLQA (Es, De, Hi, Vi, Zh, Ar) | Machine Translation | English context, non-English question |
| Temporal reasoning | TEMPLAMA, DATESET (new, template-based) | Calendar | needs current date |
| Language modeling | WikiText, held-out CCNet | — | perplexity must not degrade |
| Scaling | GPT-2 family (124M–1.6B) + GPT-J | QA, Calc, WikiSearch | tool-use ability vs. model size |

## 4. Results

### Fact completion (LAMA)
Accuracy checking whether the correct word appears in the first 5 predicted words:

| Model | SQuAD | Google-RE | T-REx |
|---|---|---|---|
| GPT-J | 17.8 | 4.9 | 31.9 |
| GPT-J + CC | 19.2 | 5.6 | 33.2 |
| Toolformer (disabled) | 22.1 | 6.3 | 34.9 |
| **Toolformer** | **33.8** | **11.5** | **53.5** |
| OPT (66B) | 21.6 | 2.9 | 30.1 |
| GPT-3 (175B) | 26.8 | 7.0 | 39.8 |

Toolformer beats the best equal-sized baseline by **+11.7 / +5.2 / +18.6** points and even beats GPT-3 (175B). It uses the QA tool for **98.1%** of examples.

### Math word problems
| Model | ASDiv | SVAMP | MAWPS |
|---|---|---|---|
| GPT-J | 7.5 | 5.2 | 9.9 |
| GPT-J + CC | 9.6 | 5.0 | 9.3 |
| Toolformer (disabled) | 14.8 | 6.3 | 15.0 |
| **Toolformer** | **40.4** | **29.4** | **44.0** |
| OPT (66B) | 6.0 | 4.9 | 7.9 |
| GPT-3 (175B) | 14.0 | 10.0 | 19.8 |

Enabling the calculator **more than doubles** performance; even the disabled model improves (it internalized math patterns from the training examples). Calculator used for **97.9%** of examples.

### Open-domain QA
| Model | WebQS | NQ | TriviaQA |
|---|---|---|---|
| GPT-J | 18.5 | 12.8 | 43.9 |
| **Toolformer** | **26.3** | **17.7** | **48.8** |
| GPT-3 (175B) | 29.0 | 22.6 | 65.9 |

Clearly beats all GPT-J-sized baselines (via Wikipedia Search, used for 99.3% of examples) but **falls short of GPT-3**, likely because the simple BM25 search engine returns poor snippets and cannot be queried interactively.

### Multilingual QA (MLQA)
The MT tool helps for all languages (used 63.8–94.9% of the time, except Hindi: 7.3%), but Toolformer does **not** consistently beat vanilla GPT-J because finetuning on CCNet shifted the data distribution. OPT and GPT-3 are surprisingly weak here.

### Temporal datasets
| Model | TEMPLAMA | DATESET |
|---|---|---|
| GPT-J | 13.7 | 3.9 |
| **Toolformer** | **16.3** | **27.3** |
| GPT-3 (175B) | 15.5 | 0.8 |

Best on both. On DATESET the gain is fully attributable to the **calendar tool (used 54.8%)**; on TEMPLAMA the calendar is barely used (0.2%) — gains there come from QA/WikiSearch instead.

### Language modeling is not harmed
Perplexity on WikiText / held-out CCNet is identical for GPT-J + CC and Toolformer (disabled) — **adding API-call training costs nothing** in core LM ability.

### Scaling laws
Tool-use ability **emerges around 775M parameters**: smaller GPT-2 models don't benefit from tools. The gap between predictions with vs. without API calls stays large even at 6.7B — bigger models get better both at solving tasks alone *and* at exploiting tools.

### Analysis highlights
- **Decoding:** with k=1 (plain greedy), the model is somewhat calibrated — it calls APIs exactly for the hard cases; raising k to 10 pushes call rates up (e.g., 8.5% → 100% on WebQS) and improves overall accuracy.
- **Data quality:** high `L⁻ᵢ − L⁺ᵢ` scores correlate with intuitively useful API calls; a little residual noise in unfiltered calls is actually useful (prevents the model from blindly trusting tools).

## 5. Limitations (stated by the authors)

1. **No tool chaining** — calls for each tool are generated independently, so the model can't feed one tool's output into another tool.
2. **No interaction** — it can't browse multiple search results or reformulate a failed query.
3. **Prompt sensitivity** — the decision to call an API is sensitive to exact input wording.
4. **Sample inefficiency** — for some tools, >1M documents yield only a few thousand usable examples (e.g., calculator).
5. **Tool cost is ignored** — the model doesn't weigh the computational cost of making an API call.

## 6. Conclusion (one-liner)

A 6.7B-parameter model, finetuned on **its own** API-call annotations that were filtered only by "does this call help me predict the next tokens?", learns to use external tools generically — and with tools enabled, **outperforms the 25× larger GPT-3 (175B) on fact completion, math, and temporal reasoning**, without losing any of its language modeling ability.
