# Decisions Log

Append-only. Entries are never edited — superseded ones get a new entry that references the old ID.

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