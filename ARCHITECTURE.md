# Architecture

## Fuller hybrid retrieval pipeline

This is the end-to-end architecture this package is designed to support. Boxes
marked **this package** are implemented here; the rest belong in the retriever
/ app layer (for example `grounded-corpus`).

```
original query
   │
   ├──────────────→ direct lexical          (retriever)
   │
   └──────────────→ direct semantic         (retriever)
   │
   ↓
static topic map match                      (this package)
   ↓
bounded LLM extension from that map         (this package)
   ↓
LLM outputs:                                (this package)
- intent
- emphasize
- de-emphasize
- historical expressions
- contrasts
- optional added concepts
   ↓
expanded semantic searches                  (retriever; queries from this package)
   ↓
fusion / rerank                             (retriever)
```

### Stage notes

1. **Direct lexical / direct semantic**  
   Always search the original user query as-is. This preserves recall when the
   topic map misses or the expansion is too narrow.

2. **Static topic map match**  
   A curated map of corpus topics (`labels`, `core_concepts`,
   `historical_expressions`, `contrasts`, optional `notes`). Matching is
   deterministic (label substring / token overlap). No LLM yet.

3. **Bounded LLM extension**  
   The model sees only the **matched subset** of the map (plus optional domain
   context). It may rephrase and lightly extend, but must ground
   emphasize / historical / contrasts in that map. `added_concepts` is capped
   (schema max 4).

4. **Expansion fields → expanded semantic searches**  
   `QueryStructure.expanded_semantic_queries()` derives multiple short embedding
   strings from intent, emphasize, historical expressions, contrasts, and
   added concepts. The retriever runs those as additional semantic searches.

5. **Fusion / rerank**  
   Combine direct channels + expanded channels. Use `de_emphasize` as a soft
   signal when ranking or filtering near-miss topics. This package does not
   implement fusion.

## This package’s slice

```
query
  → TopicMap.match(...)
  → OpenAIStructurer | HeuristicStructurer
  → QueryStructure
  → expanded_semantic_queries() / lexical_phrases()
```

| Responsibility | In scope |
| --- | --- |
| Topic map load + match | yes |
| Bounded LLM / heuristic expansion | yes |
| Expansion field schema | yes |
| Derive expanded semantic query strings | yes |
| Optional lexical phrases from expansion | yes |
| Direct lexical/semantic retrieval | no |
| Fusion / rerank / answering | no |

### Fallback behavior

- **LLM available:** match map → constrained JSON extension.
- **LLM missing or failing:** heuristic copies matched map fields; `added_concepts` stays empty; `de_emphasize` stays empty.
- **No map hits:** heuristic uses stripped topical words as intent/emphasize; LLM is instructed to stay conservative.

### Caching

In-process and optional file caches key on `map fingerprint + domain_context + normalized query` so map edits invalidate stale expansions.

## How an app should wire it

```text
q = user query

hits_lex = lexical_search(q)
hits_sem = semantic_search(q)

plan = QueryStructurer(topic_map=...).structure(q)
for eq in plan.expanded_semantic_queries():
    hits_sem += semantic_search(eq)

# optional: also FTS on plan.lexical_or()
final = fuse_and_rerank(hits_lex, hits_sem, de_emphasize=plan.de_emphasize)
```

Keep the topic map under app/corpus ownership. This library supplies the match
+ bounded expansion machinery, not a universal theology (or domain) ontology.
