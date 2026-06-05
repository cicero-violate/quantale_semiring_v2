# Ongoing Projects

## Internet-as-Data-Model Retrieval Layer

### Status

Proposed active project.

### Core Idea

Treat the internet as the live external data substrate, not as a replacement for local models.

The system should use online sources for fresh world state, then compress useful retrieved data into local semantic vectors, topology nodes, tensor tables, and durable cache entries.

### Architecture

```text
internet/world corpus
  -> retrieval/query layer
  -> trust/filter/judgment layer
  -> local embedding/compression model
  -> topology graph
  -> tensor masks/tables/kernels
  -> GPU hot-path scoring/routing
  -> local cache of distilled structure
```

### Roles

| Component | Role |
|---|---|
| Internet | Live world memory; fresh data, prices, laws, company state, market signals, documents, behavior traces. |
| Local compressed models | Fast translators from text/pages/events into dense vectors and routeable semantic features. |
| Cache | Private distilled memory of repeated or high-value facts, embeddings, source summaries, and extracted structure. |
| Topology graph | Symbolic structure layer: entities, relations, flows, constraints, opportunities, risks. |
| Tensor runtime | Executable layer for ranking, similarity, propagation, masks, closure, projection, and batch scoring. |

### Working Equation

```text
S = J(Q(I, x) + M(x) + C(x))
```

Where:

- `I` = internet / live external corpus
- `Q` = query and retrieval system
- `M` = local compressed model / embedding function
- `C` = local cache of distilled prior structure
- `J` = judgment/filter/trust layer
- `S` = system intelligence for a task

### Retrieval Loop

```text
x -> Q(I)
Q(I) -> filter/trust-rank/deduplicate
filtered evidence -> embed
embeddings -> topology nodes/edges
graph state -> compiled tensors
tensors -> GPU scoring/routing/projection
useful outputs -> cache
```

### Use Internet For

- Fresh prices.
- Laws and regulations.
- News and recent events.
- Company and product data.
- Market demand signals.
- Macro structure mapping.
- Supply chain and infrastructure state.
- Human behavior and public sentiment signals.

### Use Local Compressed Models For

- Semantic similarity.
- Intent detection.
- Query expansion.
- Deduplication.
- Local routing.
- Fast clustering.
- Offline fallback.
- Converting language into tensor-ready input.

### Use GPU Runtime For

- Large-scale similarity scoring.
- Batch ranking.
- Parallel graph updates.
- Tensor propagation.
- Masked routing.
- Closure/projection operations.
- Compiled algebraic execution.

### Minimal Downloadable Model Set

Start with small models that are easy to run locally and convert into tensors:

| Model | Type | Primary Use |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence embedding | General semantic vectors. |
| `BAAI/bge-small-en-v1.5` | Retrieval embedding | Query/document search and ranking. |
| `intfloat/e5-small-v2` | Retrieval embedding | Query/passage matching. |
| `fastText wiki-news-300d-1M` | Word vectors | Deterministic word-level vector table. |
| `GloVe 6B 300d` | Word vectors | Simple static baseline. |

### Deliverable

Build the smallest real-world macro mapping demo:

```text
messy web data
  -> retrieved evidence
  -> filtered source set
  -> embeddings
  -> topology graph
  -> GPU tensor scoring
  -> ranked decision / detected opportunity
```

### Success Condition

A stranger can see:

1. what live data entered the system,
2. what structure was extracted,
3. how the topology graph changed,
4. what tensor/GPU operation scored or routed it,
5. what decision improved.

### Risk

The internet is noisy, slow, adversarial, and unstable. The system must not treat retrieval as truth. It must use source trust, deduplication, timestamping, contradiction tracking, and local cache invalidation.

### Current Priority

High.

This project connects commercial usefulness to the quantale/tensor runtime by turning live world data into executable symbolic and algebraic structure.