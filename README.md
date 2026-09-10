# grounded-query-structure

`grounded-query-structure` owns one slice of hybrid semantic search:

**static topic-map match → bounded LLM extension → expansion fields → expanded semantic queries.**

It does **not** run direct lexical/semantic search, fuse results, or generate
answers. Pair it with [`grounded-corpus`](https://github.com/michaelmang/grounded-corpus)
(or any hybrid retriever) for retrieval + fusion/rerank.

See [ARCHITECTURE.md](ARCHITECTURE.md) for this package’s contract and the
fuller retrieval pipeline.

## Install

```bash
pip install 'grounded-query-structure[openai]'
```

Python 3.11 or newer is required. The core package has no runtime dependencies;
the `openai` extra adds `httpx` for Chat Completions.

## Five-minute example

```bash
export OPENAI_API_KEY=sk-...
grounded-query-structure \
  --topic-map examples/patristic_topic_map.json \
  "cult of the saints"
```

```python
from grounded_query_structure import QueryStructurer, TopicMap

topic_map = TopicMap.from_json_file("examples/patristic_topic_map.json")
structurer = QueryStructurer.from_env(
    topic_map=topic_map,
    domain_context="Early Church Fathers English translations (Schaff et al.).",
)
plan = structurer.structure("What did the Fathers teach about the cult of the saints?")
print(plan.intent)
print(plan.emphasize)
print(plan.historical_expressions)
print(plan.expanded_semantic_queries())
print(plan.lexical_or())  # optional FTS phrases derived from the expansion
```

Without `OPENAI_API_KEY`, the same API falls back to a heuristic that copies
fields from matched topic-map entries (no invented `added_concepts`).

## Output contract

| Field | Role |
| --- | --- |
| `intent` | Short statement of what to retrieve |
| `emphasize` | Topics/phrases that should weigh more |
| `de_emphasize` | Nearby topics to down-rank or avoid conflating |
| `historical_expressions` | Corpus-native / older phrasings |
| `contrasts` | Distinctions that clarify intent |
| `added_concepts` | Optional extras (bounded; empty on heuristic) |
| `matched_topic_ids` | Topic-map ids that grounded the expansion |

`expanded_semantic_queries()` turns those fields into multiple short embedding
queries for the “expanded semantic searches” stage.

## Design guarantees

- Output is expansion structure only — never answers or citations.
- LLM extension is bounded to the matched topic-map subset (+ tiny `added_concepts` budget).
- OpenAI calls use a constrained JSON schema (`strict: true`).
- Provider failure falls back to the map-copying heuristic by default.
- Identical query + map fingerprints are cached in-process (`source` becomes `cache`).
- Corpus vocabulary lives in the topic map (and optional `domain_context`), not hardcoded here.

## Topic map JSON

```json
{
  "entries": [
    {
      "id": "cult_of_saints",
      "labels": ["cult of the saints", "veneration of martyrs"],
      "core_concepts": ["veneration of the martyrs", "relics"],
      "historical_expressions": ["cultus sanctorum", "honor of the martyrs"],
      "contrasts": ["latria versus dulia", "worship due to God alone"],
      "notes": "Honor of saints, not divine worship."
    }
  ]
}
```

## Sibling packages

| Package | Role |
| --- | --- |
| `grounded-corpus` | Hybrid search engine + local SQLite backend |
| `grounded-corpus-postgres` | Postgres + pgvector backend |
| `grounded-corpus-eval` | Evaluation / qrels / regression gates |
| `grounded-query-structure` | Topic-map grounded query expansion (this package) |
