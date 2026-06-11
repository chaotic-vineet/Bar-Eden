# Python Depth — A Barkeep Protocol Mini-Project

The companion to the fluency set. Where fluency made the language *automatic*, this set makes the
machinery *visible* — so you can take an approach and redeploy it on a problem you haven't seen, not
just recognize it when shown.

**Prerequisite:** do the fluency set first. Depth assumes you can already *write* a closure, a class, a
generator without thinking — it spends its energy on *why they behave as they do*.

## How it's different from the fluency set

Each exercise is one of:

- **Predict-then-run** — read code, write your guess, run, then expand the hidden answer. The gap
  between prediction and reality is where your mental model gets fixed. (Answers are in collapsible
  `<details>` blocks — they fold in Jupyter/JupyterLab; if your editor shows them expanded, that's the
  one thing to check.)
- **Modify-to-target** — change working code until it hits a required output.
- **Build-from-primitives** — implement the thing under the syntax (an iterator by hand, a context
  manager, a parameterized decorator).
- **All-the-ways** — the same need solved several ways, with the judgment of when each is right.
- **Milestone** — each notebook ends with a cumulative build combining everything up to that point.

Every assert and every predict-cell was verified against a solution key before release.

## Order matters — it's cumulative

Work them front to back. Later milestones assume earlier notebooks (03 uses 02's closures; 05 leans on
01's data model and 04's classes; 07 needs 02; 08 ties everything).

| # | Notebook | The deep idea |
|---|----------|---------------|
| 01 | `01_objects_and_data_DEPTH` | names as references, identity vs equality, the small-int cache, mutability → hashability |
| 02 | `02_functions_and_scope_DEPTH` | late-binding closures, LEGB/`nonlocal`, rebind vs mutate, three ways to carry state |
| 03 | `03_iteration_and_laziness_DEPTH` | iterable vs iterator, generators as hand-written iterators, lazy pull-based pipelines |
| 04 | `04_oop_core_DEPTH` | attribute lookup order, the shared-class-attribute bug, bound methods, properties as descriptors |
| 05 | `05_data_model_dunders_inheritance_DEPTH` | `__repr__`/`__str__` fallback, the eq/hash contract, container fallbacks, MRO & `super()`, composition vs inheritance |
| 06 | `06_errors_and_resources_DEPTH` | `finally` overriding `return`, exception hierarchy, context-manager suppression, commit/rollback |
| 07 | `07_decorators_DEPTH` | `@` as reassignment, `wraps`, parameterized (3-layer) decorators, stacking order |
| 08 | `08_type_hints_and_readability_DEPTH` | annotations aren't enforced, `Protocol`/structural typing, and a full messy-module refactor |

## How to use it (Option A: just-in-time)

Don't grind all eight before resuming the project. Pull a notebook when the build reaches its machinery:

- **02** (closures) when you write swappable components / the registry
- **03** (laziness) when you build token streaming in Week 9
- **04 / 05** (classes, data model) when you write transformer modules and the memory store
- **06** (resources) when you wrap SQLite writes
- **07** (decorators) when you layer registration/validation/caching on components
- **08** (readability) during the Week 10 clean-architecture refactor

Each milestone is a deliberate miniature of something you build for real later. The depth happens in
context, which is also where it sticks.
