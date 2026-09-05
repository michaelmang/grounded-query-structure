# grounded-query-structure

`grounded-query-structure` turns a natural-language search question into a
small retrieval plan: concepts, lexical phrases, and a semantic query string.

It does **not** search a corpus, rank passages, or generate answers. Pair it
with [`grounded-corpus`](https://github.com/michaelmang/grounded-corpus) (or any
hybrid retriever) as a first pass before lexical + vector search.

## Install

```bash
pip install 'grounded-query-structure[openai]'
```

Python 3.11 or newer is required. The core package has no runtime dependencies;
the `openai` extra adds `httpx` for Chat Completions.

## Five-minute example

```bash
export OPENAI_API_KEY=sk-...
grounded-query-structure "cult of the saints"
```

```python
from grounded_query_structure import QueryStructurer

structurer = QueryStructurer.from_env(
    domain_context="Early Church Fathers English translations (Schaff et al.).",
)
plan = structurer.structure("What did the Fathers teach about the cult of the saints?")
print(plan.concepts)
print(plan.lexical_or())   # OR'd phrases for FTS
print(plan.semantic_query) # short sentence for the embedder
```

Without `OPENAI_API_KEY`, the same API falls back to a local heuristic that
strips question fluff and keeps the topical words.

## Design guarantees

- Output is structure only: concepts + retrieval forms.
- The model is instructed never to answer or invent citations.
- OpenAI calls use a constrained JSON schema (`strict: true`).
- Provider failure falls back to the heuristic planner by default.
- Identical queries are cached in-process (source becomes `cache`).
- Domain vocabulary belongs in `domain_context` (app policy), not the core.

## Cheap model default

The OpenAI provider defaults to `gpt-4o-mini`. Override with `--model` or
`OpenAIStructurer(model=...)`.

## Sibling packages

| Package | Role |
| --- | --- |
| `grounded-corpus` | Hybrid search engine + local SQLite backend |
| `grounded-corpus-postgres` | Postgres + pgvector backend |
| `grounded-corpus-eval` | Evaluation / qrels / regression gates |
| `grounded-query-structure` | Query concept extraction (this package) |
