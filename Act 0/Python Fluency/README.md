# Python Fluency — A Barkeep Protocol Mini-Project

A self-contained Python fluency track that sits *under* the Barkeep Protocol. The goal is not new
algorithms — it is making the **language itself** disappear, so that by the time you are writing the
inference engine and Mara's memory system, you are thinking about the problem, never fighting the
syntax.

Scope is deliberately bounded: **no DSA, no standard-library tour, no async, no metaclasses.** The one
allowed import is `functools` (for `reduce` and decorator metadata), because decorators are genuinely
crippled without it — and you'll meet the basic modules properly later anyway.

## How to use it

Work the notebooks **in order** — each leans on the one before (closures in `02` are load-bearing for
decorators in `07`; the container dunders in `05` reappear in the `08` capstone).

For each cell: read the short explanation, fill in the stub where it says `# YOUR CODE HERE`, run the
cell. If the asserts pass you'll see a `passed` line and nothing else. If a cell throws, read the
error against the explanation — the failure *is* the lesson.

Follow **Rule #1 of the protocol**: if you can't reproduce a solution from memory afterward, you don't
understand it yet. Reimplement, don't paste.

## The track

| # | Notebook | Covers |
|---|----------|--------|
| 01 | `01_objects_and_data.ipynb` | identity vs equality, mutability, aliasing, shallow/deep copy, unpacking, hashability |
| 02 | `02_functions_and_scope.ipynb` | `*args`/`**kwargs`, the mutable-default trap, LEGB & `nonlocal`, **closures**, lambda/map/filter/reduce |
| 03 | `03_iteration_and_generators.ipynb` | set/dict comprehensions, generator expressions, `yield`, the iterator protocol by hand, `yield from` |
| 04 | `04_oop_core.ipynb` | instance vs class attributes, instance/class/static methods, `@property` |
| 05 | `05_oop_inheritance_dunders.ipynb` | `__repr__`, `__eq__`/`__hash__`, inheritance & `super()`, duck typing, composition vs inheritance, `__len__`/`__getitem__` |
| 06 | `06_errors_and_context.ipynb` | `try`/`except`/`else`/`finally`, `raise`, custom exceptions, context managers, file handling |
| 07 | `07_decorators.ipynb` | first decorator, `functools.wraps`, memoization, decorators with arguments, stacking |
| 08 | `08_capstone.ipynb` | Barkeep-themed synthesis: mood model, memory store, streaming generator, registry decorator, token vocab |

44 exercises total. Every assert was verified against a solution key before release.

## Why these, for this project

The capstone exercises are intentional miniatures of components you build later:

- **mood model** (`08.1`) → the 4-D mood model in Week 7
- **memory store** (`08.2`) → the SQLite long-term memory in Week 8
- **streaming generator** (`08.3`) → token streaming with delay in Week 9
- **registry decorator** (`08.4`) → swappable attention/sampler components in Acts 1–2
- **token vocabulary** (`08.5`) → the tokenizer vocab in Act 2 (and your BPE detour)

When these feel automatic, the language is no longer the obstacle. That's the whole point.