# The Barkeep Protocol — Resources

Organized by act. Work through in order; each resource sits where it's needed.

---

## Act 0 — The Foundation ✅ (kept for reference)

- **NumPy Quickstart** — https://numpy.org/doc/stable/user/quickstart.html
- **PyTorch: Learn the Basics** — https://pytorch.org/tutorials/beginner/basics/intro.html
- **3Blue1Brown: Essence of Linear Algebra** — https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- **Matplotlib Pyplot tutorial** — https://matplotlib.org/stable/tutorials/pyplot.html

---

## Act 1 — The Engine

### 1.1 Bigrams → MLP ✅

- **Karpathy — makemore Parts 1–3** (bigrams, MLP, backprop) https://www.youtube.com/watch?v=PaCmpygFfXo · https://www.youtube.com/watch?v=TCH_1BHY58I · https://www.youtube.com/watch?v=VMj-3S1tku0
- **Bengio et al. 2003 — A Neural Probabilistic Language Model** ✅ read

### 1.2 Attention → the transformer

- **Karpathy — makemore Part 4: BatchNorm** — https://www.youtube.com/watch?v=P6sfmUTpUmc
- **Karpathy — Let's build GPT from scratch** — https://www.youtube.com/watch?v=kCc8FmEb1nY The main event. Watch a section, close it, rebuild from understanding.
- **"Attention Is All You Need"** — Vaswani et al., 2017 — https://arxiv.org/abs/1706.03762 Read **after** building. Sections 1–3 + the architecture figure.
- **Karpathy — minbpe** — https://github.com/karpathy/minbpe Required for D-101 (the tokenizer decision). Read the code before the training day; implement a minimal BPE if that's the path chosen.

Supplementary: 3B1B _But what is a GPT?_ and _Attention in transformers_; Jay Alammar's _Illustrated Transformer_.

### 1.3 Make it modular

- **RoFormer (RoPE)** — Su et al., 2021 — https://arxiv.org/abs/2104.09864 — Sections 1–3 while implementing.
- **Longformer (sliding window)** — Beltagy et al., 2020 — https://arxiv.org/abs/2004.05150 — skim for the concept.
- **GQA** — Ainslie et al., 2023 — https://arxiv.org/abs/2305.13245 — short, clear.
- **Karpathy — nanoGPT** — https://github.com/karpathy/nanoGPT — read after building my own.

### 1.4 The SSM block

Read in this order — the lineage matters:

- **S4 — Efficiently Modeling Long Sequences with Structured State Spaces** — Gu et al., 2021 https://arxiv.org/abs/2111.00396 — first pass only: the state-space view of sequences.
- **The Annotated S4** — Sasha Rush — https://srush.github.io/annotated-s4/ The implementation walkthrough. This replaces a Karpathy video for this phase.
- **Mamba — Linear-Time Sequence Modeling with Selective State Spaces** — Gu & Dao, 2023 https://arxiv.org/abs/2312.00752 — the selectivity argument (Sections 1–3) is the whole point: why input-dependent parameters beat LTI.
- **Mamba: The Hard Way** — Sasha Rush — https://srush.github.io/annotated-mamba/hard.html For when the paper's scan implementation gets opaque.

---

## ☕ Sabbatical A — VAE

- **Auto-Encoding Variational Bayes** — Kingma & Welling, 2013 — https://arxiv.org/abs/1312.6114 Read _after_ a first working implementation (read-after-build holds here too).
- **An Introduction to Variational Autoencoders** — Kingma & Welling, 2019 — https://arxiv.org/abs/1906.02691 The gentler long-form version; good second pass.
- Act 0's KL divergence implementation is the prerequisite. Already done.

---

## Act 2 — The Cast

### 2.1–2.2 Customers + clockwork

- No new theory — the Act 1 codebase plus plain Python. The state machine needs no library; if tempted, don't.
- Corpus selection is the real decision. **Project Gutenberg** — https://www.gutenberg.org/ — for public-domain variety. (Training on copyrighted serials stays a private exercise and constrains what can ever be published or shared — choose corpora with that in mind.)

### 2.3 The bartender

- **SQLite (Python docs)** — https://docs.python.org/3/library/sqlite3.html — the drinks DB is its first deployment; the user profile in Act 5 is its second.
- **RAG — Lewis et al., 2020** — https://arxiv.org/abs/2005.11401 — skim Sections 1–2 for the concept; this version is deliberately simpler.
- Drink data: the IBA official cocktail list as the seed; the histories get written by hand — they're flavor, and writing them is character work.

### 2.4 NPU shipping

- **OpenVINO documentation** — https://docs.openvino.ai/ Model conversion + quantization paths. Expect friction; the friction is the deployment lesson.
- **OpenVINO PyTorch conversion guide** — under "Convert from PyTorch" in the docs. Fixed input shapes are the constraint to design for.

---

## Act 3 — The Soul

### 3.1 The inference engine

Path A (GPT-2):

- **GPT-2 weights** — https://huggingface.co/openai-community/gpt2 — raw safetensors, not the model class.
- **tiktoken** — https://github.com/openai/tiktoken
- **Karpathy — Let's reproduce GPT-2 (124M)** — https://www.youtube.com/watch?v=l8pRSuU81PU
- **GPT-2 paper** — Radford et al., 2019.

Path B (modern ≤3B):

- **Qwen2.5 / Llama 3.2 small-model weights** (Hugging Face) — raw safetensors; architecture details in model cards / tech reports.
- The tokenizer will _not_ be tiktoken — reading the model's tokenizer.json format and writing a loader for it is part of the work.
- The 1.3 implementations (RoPE, GQA, SwiGLU, RMSNorm) are the real reference. That was the point of 1.3.

Reference: **llama.cpp** — https://github.com/ggerganov/llama.cpp — what optimized inference looks like.

### 3.2–3.3 Writing Mara

- **InstructGPT** — Ouyang et al., 2022 — https://arxiv.org/abs/2203.02155 — Sections 1–3.
- **Constitutional AI** — Bai et al., 2022 — https://arxiv.org/abs/2212.08073 — Sections 1–3. Feeds the principle table (4.5) directly.
- **DPO** — Rafailov et al., 2023 — https://arxiv.org/abs/2305.18290 — the feasible-on-this-hardware alignment path, if the heart ever gets fine-tuned.
- **Hugging Face PEFT** — https://huggingface.co/docs/peft — LoRA, same condition.
- **Lilian Weng — Prompt Engineering** — https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
- **GPT-3 paper** — Brown et al., 2020 — Sections 1–2: why in-context learning works at all.

---

## Act 4 — The Mind

### 4.1 The label pipeline

- **Distilling the Knowledge in a Neural Network** — Hinton, Vinyals, Dean, 2015 — https://arxiv.org/abs/1503.02531 The grandfather paper for the whole cheap tier: small models as compressed copies of an expensive judge.

### 4.2 Encoder + heads

- **BERT** — Devlin et al., 2018 — https://arxiv.org/abs/1810.04805 — Sections 1–3: what an encoder representation is for.
- **sentence-transformers** — https://www.sbert.net/ — the wire-don't-train backbone candidates; read how mean-pooled embeddings work, don't reimplement them.

### 4.3 Session-state tracker

- No new resources — the 1.4 block, redeployed. The work is feature design and labels, not architecture.

### 4.4 Associative retrieval

- **Hopfield Networks is All You Need** — Ramsauer et al., 2020 — https://arxiv.org/abs/2008.02217 The update rule ξ' = X·softmax(βXᵀξ) _is_ attention over the store.
- **Hopfield layers blog (ML-JKU)** — https://ml-jku.github.io/hopfield-layers/ — the companion walkthrough.
- Historical anchor: Hopfield 1982 → Ramsauer 2020 is the bridge from associative memory to the attention built in 1.2.

### 4.5 The principle table

- No ML resources — this is design writing. Constitutional AI (above) is the conceptual neighbor; Cyc is the cautionary tale: rules constrain, they don't detect.

---

## Act 5 — The Bar

- **SQLite** — second deployment; the user-profile schema is the new design work.
- **Flask** — https://flask.palletsprojects.com/
- **Gradio** — https://www.gradio.app/ — fallback only; less atmosphere.
- **Jesse Schell — The Art of Game Design** — the chapters on character and player experience.
- **Sherry Turkle — Alone Together** — why people project personality onto machines. (ELIZA, 1964, all over again.)

---

## Ongoing / As-Needed

- **Karpathy — Intro to Large Language Models** — https://www.youtube.com/watch?v=zjkBMFhNj_g
- **Karpathy's blog** — https://karpathy.github.io/ — _The Unreasonable Effectiveness of RNNs_ pairs well with the SSM phase: the recurrence worldview the field left and is now partially returning to.
- **Anthropic Research** — https://www.anthropic.com/research
- **Yannic Kilcher** — https://www.youtube.com/@YannicKilcher — paper walkthroughs when one won't parse alone.

**Long-term vision:**

- Anthropic — _Tracing the thoughts of a large language model_ (NLAs), May 2026
- Anthropic — _The Assistant Axis_, January 2026
- Google — _Nested Learning_, NeurIPS 2025
- _Lifelong Learning of LLM-based Agents: A Roadmap_, 2025 — https://arxiv.org/pdf/2501.07278
- **ContinualAI** — https://www.continualai.org/

---

## The Math Beyond

Not for now — for awareness and course selection. Optimization theory · information theory · statistical learning theory · deeper linear algebra (eigendecomposition, SVD — the SSM phase will make this urgent early: S4's HiPPO matrix is spectral theory at work) · measure-theoretic probability.

## How to Read a Paper

Three passes: (1) abstract/intro/conclusion/figures, ~30 min; (2) the method section carefully, 1–2 hrs; (3) experiments/appendices only if implementing. Speed comes with practice — by paper five it's much faster, and with the SSM and Hopfield phases in the sequence, paper five arrives soon.