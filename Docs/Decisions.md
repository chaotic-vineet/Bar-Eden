# Decisions Log

>Append-only. Entries are never edited — superseded ones get a new entry that references the old ID.

**Format:** each entry records the decision, the options that lost and why, the actual reasoning, and the condition for reopening it.

---

## Open decisions (decide at the marked phase — not before, not after)

|ID|Decision|Decide at|
|---|---|---|
|D-101|WikiText-2 tokenizer: char-level vs minbpe BPE|1.2, before training day|
|D-102|Inference engine target: GPT-2 / modern ≤3B / A-then-B|3.1 start|
|D-103|Heart size: ≤3B current hardware vs 7–8B if 32GB laptop|4.6 start|
|D-104|Bartender voice: template slots vs diegetic senior+apprentice|2.3|
|D-105|Final per-turn vs session-end tracker split|4.6, from measured latency|

---

## Decided

### D-001 · Project timeline: year-scale sequence, not 10-week calendar

**Date:** 2026-06-10 · **Phase:** meta · **Status:** decided

**Decision:** The Barkeep Protocol runs ~1 year at sustainable pace; phases have sizes and gates, not dates.

**Options considered:** Keep the 10-week summer schedule — rejected: can't compress three decades of architectures into ten weeks without trading away understanding, which violates Rule 3.

**Why:** The scope grew (the cast, the SSM block, the mind's auxiliary stack) and the deadline pressure was producing calendar-debt thinking ("days 1–4 on day 4") instead of learning. No external deadline exists after dropping Stardance.

**Revisit if:** Never — though phase sizes get re-estimated freely.

---
### D-002 · Repo visibility and commit style: public, small, messy

**Date:** 2026-06-11 · **Phase:** meta · **Status:** decided

**Decision:** Repo stays public; commit small and often; mess stays in history (branches for genuinely exploratory flailing; unpushed local commits may be reorganized freely).

**Options considered:** Private until done, single polished dump — rejected: a clean codebase materializing at once reads as AI-generated in 2026; loses backup-by-default; prevents commit discipline from developing; and contradicts the project's own thesis (the lived log over the fact table).

**Why:** Commit history is unfakeable provenance of a human learning in sequence — the strongest signal the repo can send.

**Revisit if:** Never.

---
### D-003 · The bar is the project; learning artifacts become the cast

**Date:** 2026-06-11 · **Phase:** meta · **Status:** decided

**Decision:** Customers = from-scratch modular transformers (babble-only, state-machine-scripted, NPU-deployed); bartender = RAG + constrained voice; Mara = the multi-model mind.

**Options considered:** Original framing (Mara as product, bar as aesthetic, learning transformer as discarded exercise) — rejected: left a dangling artifact with no diegetic job.

**Why:** Every model built now has a seat; the bar becomes a difficulty gradient (bartender → customers → Mara) and a rehearsal space for Mara's memory and fact-custody disciplines at low stakes.

**Revisit if:** The cast's maintenance cost ever exceeds its teaching value.

---
### D-004 · Architecture seating: jobs, not résumé lines

**Date:** 2026-06-11 · **Phase:** meta · **Status:** decided

**Decision:** Seated in Mara: Mamba (session-state), modern Hopfield (retrieval), hybrid SSM-attention (one customer), encoder+heads, principle table, borrowed heart. Not seated: VAE, CNN/ResNet, GNN, Neural ODE — standalone sabbaticals instead.

**Options considered:** "Every architecture inside Mara" — rejected: jobless architectures are passengers; "why is there a VAE in your chatbot?" has no good answer, while "here's my standalone VAE study" starts a conversation.

**Why:** Resume value is identical for standalone builds; system integrity and interview defensibility are higher. Each seated architecture survives the question "why is this here?"

**Revisit if:** A genuine job appears (e.g. the bar grows vision → CNN gets a seat).

---
### D-005 · Workspace: vault at repo root, one folder two lenses

**Date:** 2026-06-12 · **Phase:** meta · **Status:** decided

**Decision:** The Obsidian vault root _is_ the repo root. PyCharm and Obsidian are two views of the same folder. `Docs/templates/` is Obsidian's template folder; `notes/daily/` is the private diary (gitignored along with `.obsidian/`); `notes/reading/` is the public reading log (tracked, one note per paper/concept, own-words section mandatory).

**Options considered:** Separate vault in its own subdirectory — rejected: templates would need duplicate copies that drift; Obsidian couldn't see or link to `Docs/`; reading notes couldn't link to checkpoints.

**Why:** One tracked copy of every template, checkpoints draftable in Obsidian with links to the dailies they distill, and the reading life lands in the repo automatically. Privacy is by content (`daily/`), not by tool.

**Revisit if:** Vault performance degrades as the repo grows (excluded-files settings should prevent this).

---
### D-006 · Folder naming: keep "Act 0" / "Act 1" / "Docs"

**Date:** 2026-06-12 · **Phase:** meta · **Status:** decided

**Decision:** Top-level folders keep their human-readable names — capitalized, with spaces — against the lowercase-no-spaces convention.

**Options considered:** `act0/`, `act1/`, `docs/` — rejected for now: the folders contain only notebooks and documents, nothing importable, so the cost is zero and the names read better.

**Why:** Aesthetic preference at zero current cost.

**Revisit if:** A `src/` package ever needs to import from these paths — spaces become illegal in import paths the day that happens.

---
### D-007: No force-pushing history on main

**Date:** 2026-06-12 · **Phase:** meta · **Status:** decided

**Decision:** No `git push -force` a commit on `main` from now on.

**Why:** The whole idea behind Mara is that intelligence is born from lived time. If project commit history is falsified by a `-force` push, then it contradicts the very idea behind it.

**Revisit if:** never.

---
### D-008 · Soft phase targets, hard act deadlines

**Date:** 2026-06-12 · **Phase:** meta · **Status:** decided · *refines D-001*

**Decision:** Every phase has a target date, but those dates are soft — if a phase runs long, its target moves and every later phase target moves with it. Each act also has a "last call" date, and those are hard — they don't move no matter what. The last calls are: Act 1 Jul 25, Act 2 Aug 31, Act 3 Oct 31, Act 4 Jan 31 '27, Act 5 Apr 30 '27. The first real deadline is phase 2.1 (customers built) by Jul 31, 2026. If 1.2–1.4 run long, pure-attention customers are enough to clear 2.1, and the hybrid SSM-attention regular can be added after 1.4 — so the SSM work doesn't put Jul 31 at risk.

**Rejected:** No dates at all (how D-001 originally read) — once Stardance was dropped there's nothing left to push the work forward. Hard dates on every phase — too rigid, and it contradicts letting phases take as long as they take.

**Reasoning:** There's no external deadline anymore, so without some structure I'll drift and not finish. Hard dates on every phase are too strict and fight against actually understanding the material. Soft phase targets plus hard act deadlines is the middle ground: day-to-day stays flexible, but each act still has a real cutoff.

**Revisit if:** A last call slips by more than about 2 weeks (that means I sized the act wrong, not that I'm behind), or an external deadline reappears.

---

### D-009 · 1.2 attention training data layout

**Date:** 2026-06-13 · **Phase:** 1.2 · **Status:** decided

**Decision:** One row per word, with the target being the input shifted by one position. Each word becomes `full = [0] + chars + [0]`, where token 0 (`.`) marks both the start and the end. The input is `full[:-1]` and the target is `full[1:]`. Rows are padded up to `block_size = 8` at the end (the tail) with token 27 (`#`), and cross-entropy uses `ignore_index=27` so padding doesn't count toward the loss. Words longer than 7 characters are dropped before this step (4801 of 32033, about 15%) so every word fits in 8 slots. For the loss, logits come out as `(N,T,V)` and get permuted to `(N,V,T)` to line up against the target's `(N,T)`.

**Rejected:** Padding at the front instead of the back — it needed a separate mask to hide the padding from attention, and a fully-padded row produced all `-inf` scores, which gave NaN. Moving the padding to the tail fixed both, because the causal mask already hides tail positions. Keeping the sliding window from the MLP — it's pointless now that every position is already a prediction target, so it was cut (dropping the row count from 180K to 27K). The zero-padding concat stream — held off until WikiText.

**Reasoning:** Two fixes both worked by removing the cause rather than handling the symptom: drop the sliding window instead of carrying it, and move the padding to the tail so one change kills two separate bugs. Dropping words over 7 characters is just the cost of a fixed 8-slot window on character data, and 15% is fine for a practice dataset.

**Revisit if:** I move to WikiText (where the stream layout is worth reconsidering), or any dataset needs sequences longer than `block_size`.

---
### D-010 · Names dataset: CONTEXT_SIZE = 10, fixed block, no bucketing

**Date:** 2026-06-13 · **Phase:** 1.2 · **Status:** Decided (implemented)

**Decision:** CONTEXT_SIZE = 10. Names with length > 9 are dropped entirely (~1.8% of the corpus). Fixed block size, no bucketing.

**Options considered:** CONTEXT_SIZE=8 / drop >7 chars (15% dropped, original baseline); max-pad to longest name (~16, most rows mostly padding); length-bucketed batches (zero drop).

**Why:** 10 sits at the knee of the length distribution — 98.2% retention vs 8's 85%, at minimal extra pad cost. Max-pad wastes compute quadratically on a rare long tail. Bucketing solves a "can't afford to drop data" problem this 32K-name toy doesn't have — the dropped tail is more-of-the-same phonotactics, not new structure.

**Revisit if:** training on a corpus where the dropped tail carries information not present in the kept distribution (e.g. Act 2.1 customer corpora).

---

### D-011 · Bucketing deferred to Act 2.1

**Date:** 2026-06-13 · **Phase:** 1.2 (impl) / Act 2.1 (revisit) · **Status:** Parked

**Decision:** Bucketing (length-grouped batches, zero drop) is deferred to Act 2.1 customer training. Not relevant to names or WikiText.

**Options considered:** build bucketing now as a skills exercise; build it for WikiText.

**Why:** WikiText is a continuous stream sliced into fixed windows — no per-example length variation exists to bucket. Bucketing only earns its seat where there's a variable-length corpus worth keeping whole.

**Revisit if:** Act 2.1 corpus selection reveals highly variable-length examples where truncation would lose meaningful data.

---

### D-012 · Naive list-of-heads multi-head attention

**Date:** 2026-06-13 · **Phase:** 1.2 · **Status:** Decided (implemented), efficient version parked

**Decision:** MultiheadAttention is the naive list-of-heads implementation — h independent Attention objects, looped, concatenated via `torch.concat`, projected by W_out. The efficient single-matrix-plus-reshape version is not built.

**Options considered:** efficient version — one (d,h⋅dqk)(d, h{\cdot}d_{qk}) (d,h⋅dqk​) projection, reshape to (B,h,T,dqk)(B,h,T,d_{qk}) (B,h,T,dqk​), batched matmul, transpose+reshape to merge.

**Why:** the naive version is the most direct translation of the math, and its manual `parameters()` walk over a list teaches exactly the problem `nn.ModuleList` solves. The efficient version's correctness rests on "concat = reshape," proven only by asserting bit-identical output between the two — not yet done.

**Revisit if:** per-step Python overhead becomes a real bottleneck (single-head 100K steps = 38min vs multi-head h=5 = 60min at equal params, ~5x per-head dispatch overhead) — or as a dedicated "prove the reshape" exercise.

---

### D-013 · d_qk and d_v kept as independent dimensions

**Date:** 2026-06-13 · **Phase:** 1.2 · **Status:** Decided (architectural, ongoing)

**Decision:** dqkd_{qk} dqk​ and dvd_v dv​ are kept as independent constructor parameters throughout. In practice DIM_QK=10, DIM_V=20 (not equal) for the h=5h=5 h=5 runs.

**Options considered:** equate dqk=dv=d/hd_{qk} = d_v = d/h dqk​=dv​=d/h (2017-transformer convention, tidy concat arithmetic, WOW_O WO​ exactly d→dd \to d d→d).

**Why:** the two dimensions answer different questions (dqkd_{qk} dqk​ = resolution of attention matching, dvd_v dv​ = width of information carried out) and are mathematically decoupled — dqkd_{qk} dqk​ vanishes inside the softmax, only dvd_v dv​ determines concat width. Equating them is convention for tidiness, not a requirement.

**Revisit if:** WOW_O WO​'s input width (h⋅dvh \cdot d_v h⋅dv​) stops being a clean relationship to dembd_{emb} demb​ — then the output projection shape needs explicit attention.

---

### D-014 · From-scratch bookkeeping: no nn.Module, no DataLoader, no torch.optim

**Date:** 2026-06-12/13 · **Phase:** 1.2 · **Status:** Decided (pedagogical, ongoing)

**Decision:** No `nn.Module`, no `Dataset`/`DataLoader`, no `torch.optim`. Every layer is a plain class with `__call__`/`parameters()`/`config_dict()`; `parameters()` chains are hand-written and maintained; training uses hand-rolled SGD with a manual cosine schedule.

**Options considered:** `nn.Module` + `nn.ModuleList` for auto-registration; `Dataset`+`DataLoader` for batching; `AdamW`.

**Why:** the manual `parameters()` chain is a deliberate "feel the bookkeeping `nn.Module` automates" exercise — already caught two real bugs (LMHead returning a `Linear` object instead of its parameters; `MhaFfnModel` forgetting `self.ffn.parameters()`) that auto-registration would have prevented, which is the lesson. AdamW was rejected specifically to keep the optimizer fixed across the ablation.

**Revisit if:** WikiText training, where `DataLoader`'s lazy loading earns its keep on a corpus too big for one tensor; or when dropout/`model.train()`/`model.eval()` is needed.

---

### D-015 · Causal mask built once, shared by reference across heads

**Date:** 2026-06-13 · **Phase:** 1.2 · **Status:** Decided (implemented)

**Decision:** the causal mask (T×TT \times T T×T, from `dataset.CONTEXT_SIZE`) is built once in `MultiheadAttention.__init__` and passed by reference into each head's `__call__`. No head builds its own mask.

**Options considered:** each `Attention` head builds its own (T,T)(T,T) (T,T) mask in `__init__` (h identical copies).

**Why:** causality is a layer-level constraint, identical across heads — building it per-head duplicates constant state at the wrong level. Principle: shared non-parameter state lives where the sharing is, not where it's merely consumed.

**Revisit if:** WikiText's document-masking needs a per-position-varying mask — same "build once at the level of sharing" principle applies, but the mask is no longer a single fixed constant.

---

### D-016 · LMHead untied from Embedding

**Date:** 2026-06-13/14 · **Phase:** 1.2 · **Status:** Decided (untied), Act 3 revisit

**Decision:** `LMHead` is a separate, untied `Linear(embedding_dim, vocab_dim)` — not sharing weights with `Embedding`'s table.

**Options considered:** weight-tying (`LMHead = Embedding.T`), as GPT-2 does.

**Why:** vocab=28 makes parameter savings trivial; tying adds a shared-gradient bookkeeping subtlety (the tied tensor must appear exactly once in `parameters()` or get double-stepped) that's an uncontrolled variable inside today's ablation. Untied keeps the model boring and the comparison clean.

**Revisit if:** Act 3.1, when loading GPT-2 weights (which are tied) — building tying once deliberately, on a model you understand, before meeting it in a checkpoint you didn't write.

---

### D-017 · Model returns natural (B,T,V); permute lives in train.py

**Date:** 2026-06-13 · **Phase:** 1.2 · **Status:** Decided (implemented)

**Decision:** `MhaModel`/`MhaFfnModel.__call__` returns logits in natural (B,T,V)(B,T,V) (B,T,V) layout. The permute to (B,V,T)(B,V,T) (B,V,T) for `cross_entropy`'s contract happens in `train.py` at the loss call, not inside `LMHead`.

**Options considered:** `LMHead` internally permutes and returns (B,V,T)(B,V,T) (B,V,T).

**Why:** the permute is loss-contract glue specific to one consumer; baking it into `LMHead` would make the model's output shape wrong for any other consumer (generation, per-position inspection) — caught as a real problem when the causal-mask perturbation test needed to slice the position axis.

**Revisit if:** a second loss function with a different contract is introduced — each consumer adapts independently at its own seam.

---

### D-018 · FFN in series with no residual (superseded)

**Date:** 2026-06-14 · **Phase:** 1.2 · **Status:** Decided (implemented for ablation), superseded by D-024

**Decision:** `MhaFfnModel` applies `FeedForward` in series after `MultiheadAttention` with no residual connection (x=ffn(x)x = \text{ffn}(x) x=ffn(x), not x=x+ffn(x)x = x + \text{ffn}(x) x=x+ffn(x)). No normalization anywhere pre-block.

**Options considered:** add x=x+ffn(x)x = x + \text{ffn}(x) x=x+ffn(x) (and/or x=x+mha(x)x = x + \text{mha}(x) x=x+mha(x)) immediately, since it's trivial to write.

**Why:** residuals are themselves a variable worth isolating — bundling them into the FFN-ablation step would make "add FFN" mean something structurally different from "add multihead" (replacement vs addition-to), confounding two questions in one measurement. Kept as a clean three-point staged ablation, with residual+norm as their own dedicated experiment.

**Revisit if:** the block session (D-024) directly supersedes this. The bare-FFN result (dev 2.1159, gap 0.157, train ~1.96) is the baseline the residual+norm block is compared against.

---

### D-019 · Single-head baseline as parameter-matched h=1 special case

**Date:** 2026-06-14 · **Phase:** 1.2 · **Status:** Decided (implemented)

**Decision:** the single-head baseline is `MhaModel(num_heads=1, dim_qk=50, dim_v=100)` — h=1h=1 h=1 through the same class as multi-head, with dqk/dvd_{qk}/d_v dqk​/dv​ scaled by `NUMBER_OF_HEADS` so total QKV width matches the h=5h=5 h=5 model exactly (both 10,680 params).

**Options considered:** reuse the original hand-rolled notebook's single-head result (9,640 params, fused output-to-vocab matrix).

**Why:** h=1h=1 h=1 through `MhaModel` makes single-head a genuine special case of the same class, and matching total dqk/dvd_{qk}/d_v dqk​/dv​ makes the 11 1-vs-55 5 comparison truly parameter-matched. Result (dev 2.3123, gap 0.013) reproduced the old baseline's ~2.31/merged result almost exactly, confirming the new class structure is sound.

**Revisit if:** never for this ablation — MHA+FFN's extra params (16,810 vs 10,680) remain an acknowledged, unavoidable confound, noted rather than hidden.

---

### D-020 · Identical TrainConfig held fixed across all three ablation runs

**Date:** 2026-06-13/14 · **Phase:** 1.2 · **Status:** Decided (held fixed across ablation)

**Decision:** all three ablation runs use identical `TrainConfig` — hand-rolled SGD (`p.data += -lr * p.grad`), cosine LR 0.5→0.01, batch_size=4096, 100,000 iterations, seed 42, identical batch sequence via a fresh generator inside `train_model`.

**Options considered:** switch to AdamW; change the LR schedule mid-ablation (raised when multi-head's slow convergence at 20K steps looked like it "needed" a different schedule).

**Why:** an ablation requires holding everything except the tested variable fixed. The "cosine isn't good for attention" instinct was a misdiagnosis of a step-budget observation — the curve was still converging as expected, just slowly. Changing optimizer/schedule for one run only would add a second confound on top of architecture.

**Revisit if:** 1.3, when comparing optimizers/schedules becomes the actual experiment rather than incidental to one.

---

### D-021 · runs.jsonl logging with config/metric record types + self-describing config_dict()

**Date:** 2026-06-14 · **Phase:** 1.2 · **Status:** Decided (implemented)

**Decision:** experiment results log to one shared `runs.jsonl`, written incrementally. Two record types: `"config"` (once per run — `model.config_dict()` + `TrainConfig` fields + param count) and `"metric"` (once per `log_interval`). Every model class implements `config_dict()`, chained like `parameters()`, so the model is self-describing.

**Options considered:** Weights & Biases; a single in-memory results dict only (lost entirely when a kernel crashed mid-run on 2026-06-13).

**Why:** JSONL append-per-checkpoint is crash-safe by construction — a kernel death loses at most the steps since the last checkpoint. W&B solves the same problem with a hosted dashboard, but three runs don't need one; building the minimal version first means W&B's value is legible once 1.3's sweeps actually strain a flat file.

**Revisit if:** 1.3, when sweeping positional encoding × attention variant × FFN variant × norm produces enough runs that a flat JSONL becomes unwieldy.

---

### D-022 · results.py as the canonical, committed ablation script

**Date:** 2026-06-14 · **Phase:** 1.2 · **Status:** Decided (implemented)

**Decision:** `results.py` is the canonical, committed script — runs all three ablation models end-to-end, overwrites `runs.jsonl` (single source of truth, no stale/duplicate entries), and saves both the full per-step results dict and trained parameters (`*_weights.pt`) per run. The exploratory testing notebook is not committed.

**Options considered:** commit the testing notebook as-is (not idempotent — re-running would duplicate JSONL entries via both a live run and a hand-backfilled cell); save only results dicts (diagnostics like `weight_histogram`/`activation_saturation` need live model objects, and the training kernel had already been restarted before those ran this round).

**Why:** notebooks are the exploratory lab bench; an unattended script producing a clean, reproducible artifact set (logs, results, weights, plots) is the right committed form for a completed experiment. Saved weights mean deferred diagnostics (gradient distributions, update-ratio plot) don't require retraining.

**Revisit if:** never for this ablation — the pattern (script + saved weights/results + `plots/`) is likely the template for future phase-closing experiment commits.

---

### D-023 · Next target: transformer block (residual + norm), scaling toward ~500K params

**Date:** 2026-06-14 · **Phase:** 1.2 → 1.2 item 3 · **Status:** Open (starting now)

**Decision:** next build target is a transformer block (residual connections + normalization) added to the attention+FFN pipeline, en route to training a small (~500K parameter) transformer on names.txt — up from the current ~10-17K parameter models.

**Options considered:** continue ablating at the current tiny scale; jump straight to WikiText.

**Why:** residuals+norm were deliberately deferred from today's ablation (D-018) specifically so they could be their own clean measurement, with the bare MHA+FFN result (dev 2.1159, gap 0.157) as the baseline to compare against. ~500K params on names.txt is a meaningful scale step while staying on understood infrastructure.

**Revisit if:** this is the live decision being acted on. Positional encoding (1.2 item 4) and the train/val-gap prediction remain open against whatever this block produces.