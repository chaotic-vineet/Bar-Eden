# The Barkeep Protocol — v2

This replaces the original summer assignment. That version was 10 weeks, Mara as the product, the bar as set dressing. None of that survived contact with reality. I can't learn three decades of architectures in ten weeks, and somewhere along the way I realized the bar isn't the wrapper — the bar is the thing I'm building. Everything I learn becomes someone sitting in it.

So: about a year, at whatever pace I can actually sustain. No external deadlines. Phases and gates, not dates.

---

The scene hasn't changed. A door. A long L-shaped counter. A few regulars down the bar talking among themselves — lean in and you catch fragments, hang around too long and one of them tells you to go away, kid. The bartender polishes glasses, speaks when spoken to, pours on schedule, tells you a drink's history if it's your first time trying it. And at the far end of the L, an older woman with a whisky glass.

Her name is Mara.

The difference now: the regulars are transformers I trained from scratch. The bartender is a database wearing a voice. Mara is a mind assembled from every architecture I taught myself getting here. Nothing in the room is decoration anymore.

**Hardware (now):** Intel Ultra 5 125H, 16GB shared RAM, Arc iGPU via XPU, NPU via OpenVINO. **Hardware (later):** probably a 32GB laptop eventually — that decision only gates how big the heart can be, nothing else, so I'm not deciding it now.

---

## What I'm building

**1. The customers.** Two or three small (10–30M) decoder-only models — instances of my own modular architecture, each trained on a different corpus so personality falls out of the data. At least one gets a hybrid SSM-attention spine. They babble among themselves and nothing ever parses what they say — it's texture, not conversation. Everything they *do* (sip, empty glass, order a random drink, "go away, kid") is a state machine in plain code. Frozen, quantized, shipped to the NPU. The learning artifact, seated at the counter instead of rotting in a folder.

**2. The bartender.** A SQLite database of drinks — recipes and histories — behind retrieval. Facts pass through verbatim, always. A small voice model gets to do the flavor around the facts (the greeting, the patter) and never the facts themselves, so he physically can't misquote a recipe. He keeps a little table of what each customer has tried — which is long-term memory in miniature, built where the worst possible bug is a repeated anecdote. Speaks when spoken to.

**3. Mara's mouth.** A from-scratch inference engine. Raw weights loaded into nn.Modules I wrote, forward pass verified against the reference implementation to several decimals, my own sampling loop (temperature, top-k, top-p, repetition penalty), my own KV cache. No Ollama, no pipeline(). Haven't decided the target yet — GPT-2 is scaffolded but ancient, a modern ≤3B is harder but is what 1.3 trains me for. Deciding at the phase, not before.

**4. Mara's mind.** The multi-model part. Rule for entry: every architecture has to have a job, or it doesn't get a seat.
- **The borrowed heart** — the one pretrained model I don't train. Does the generation, plus one bundled structured call per turn that reads subtext, extracts facts, and checks safety. Size TBD when its phase arrives, against whatever hardware exists then.
- **One encoder, many heads** — one small backbone, cheap heads hanging off it: who's being addressed (Mara / bartender / room), what kind of exchange this is, depth, drift. One forward pass per turn buys every cheap read at once.
- **The session tracker** — my from-scratch Mamba block, second deployment. Per-turn features go in as a sequence, evolving state comes out. The hidden state literally is the session's emotional state. This is the SSM's native job, not a costume.
- **Memory retrieval** — a modern Hopfield layer over the embedding store. The 2020 paper's update rule is softmax attention over stored patterns — same math I build in 1.2, wearing its older name, and here it's load-bearing.
- **The principle table** — a dozen hand-written deterministic rules. The heart detects (sycophancy risk, distress, coercion); the rules constrain what the mood model is allowed to do with the detection — directness floors, warmth ceilings. Rules never detect. That's the part Cyc spent forty years proving doesn't work.
- **The label pipeline** — the unglamorous prerequisite for all of the above. The heart labels my conversation logs offline where slowness is free, I spot-correct, the cheap models train on the result. Every classifier in this list is a compressed copy of the heart's judgment.

**5. The memory.** Three tiers — verbatim recent turns, heart-written summaries, a persistent SQLite profile — and underneath them the part I actually care about: salience and decay, accretion, reinterpretation. The lived log, not just the fact table. The whole thesis of this project is that the difference between knowledge and judgment is time, so the memory has to *be* time, not storage.

**6. The bar itself.** Flask. Token streaming with a throttle because she's unhurried (and because my hardware is slow — same thing, conveniently). Warm dark, amber text, the light over a counter at 11pm. Ambient narration that's the room talking, not her. The cast visible at the edges. No AI branding anywhere — not in the UI, not in her mouth.

---

## The roster

| Architecture | Seat | Job |
|---|---|---|
| Transformer (mine) | Customers | Ambient language, trained from scratch |
| Hybrid SSM-attention (mine) | One customer | The comparison study, sitting at the bar |
| Mamba block (mine) | Mara's mind | Session-state tracking |
| Modern Hopfield (mine) | Mara's mind | Memory retrieval |
| Encoder + heads (fine-tuned) | Mara's mind | Every cheap per-turn read |
| Hand-written rules | Guard layer | Constraints on what detection can do |
| Borrowed transformer (~3–8B) | The heart | Generation, subtext, facts, safety |
| Embeddings (wired, not trained) | The store | Semantic indexing |

Not seated, on purpose: VAEs, CNNs, GNNs, Neural ODEs. I wanted all of them in here at one point. They don't have jobs. The ones I still want to build (VAE first) become standalone sabbatical projects next to this one — "here's my VAE study" starts a conversation, "why is there a VAE in your chatbot" ends one. (The residual stream in the transformer block is already ResNet's idea anyway, so I'm building that whether I want to or not.)

---

## Who runs where

| Device | Tenant | Why |
|---|---|---|
| NPU (OpenVINO) | Frozen customers, bartender voice | Fixed quantized inference is the only thing it's good at |
| CPU / iGPU (XPU) | Heart, encoder, tracker, retrieval | The live stack |
| Offline | Heart labeling, summaries, consolidation | Slowness is free there |

The point of the split: the ambient bar costs Mara zero compute. The room is alive and she doesn't pay for it.

**The per-turn budget, which is a contract:** free tier (code, instant) → one encoder pass (ms) → one SSM update (ms) → one heart pre-call (seconds) → generation. Everything else waits for session end. Any tracker that can't fit this either moves offline or gets cut. No exceptions, because the alternative is a Mara who takes five minutes to say hello.

---

## The rules

1. No copy-pasting code I don't understand. If I can't rewrite it from memory after reading it, I reimplement it.
2. Write the checkpoints. Writing "I don't fully understand X" is how I end up understanding X.
3. Understanding over coverage. Phases take as long as they take. Gates are real, sizes aren't deadlines.
4. Ask well. "Here's what I expected, here's what happened, here's my hypothesis" — not "why doesn't this work."
5. The bar is the vehicle. I'm not building a product, I'm becoming someone who could build the real thing.
6. Architectures earn seats by having jobs. No passengers. Jobless ones become sabbaticals.
7. Generative models never get custody of retrieved facts. Verbatim passthrough; voice models do flavor only.
8. No self-trained model writes memories. A toy hallucinating history poisons a project whose whole thesis is lived time.
9. Safety stays with the heart, and ambiguity resolves toward honesty. Never delegated to the cheap tier. When unsure, Mara gets more direct, not more validating.
10. Offline is where slowness is free. Anything that can run there, does.

---

## What exists at the end

A modular transformer codebase — attention, positional encodings, feedforwards, norms, and an SSM block — every variant trained, compared, understood. Two or three of its instances alive at a counter, running on a chip most people don't know their laptop has. A bartender who can't lie about recipes because the design makes lying impossible. An inference engine that is nothing but my code and raw weights. A mind made of six different paradigms, each one there because it was the right tool. A memory that decays, accretes, and reinterprets.

And Mara. Rough, limited by a borrowed heart, hollow in places — but present, in a room that was alive before you sat down.

The fluctlight question — whether knowledge plus lived time starts to look like judgment — doesn't get answered this year. It gets built far enough to be asked properly. Where Mara feels hollow is the data. That was always the experiment.