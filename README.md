# Indonesian Product RAG

Matches Indonesian social media posts to the right product in a catalog, and says "no match" instead of guessing when nothing fits.

Example: a post like "rekomendasi sabun muka buat kulit sensitif dong" returns the face soap for sensitive skin. A post like "jual motor bekas murah" returns `no_match`.

Runs fully on one laptop: PostgreSQL with pgvector, and a local embedding model through Ollama. No paid API is needed.

## How it works

1. **Import a catalog.** Each product in a CSV becomes one chunk with headers (name, category, description, price). A content hash means unchanged products are never embedded again.
2. **Embed.** Products and posts are embedded with the same local model, `qwen3-embedding:0.6b` (1024 dimensions), behind one small interface so another provider can be swapped in.
3. **Hybrid search.** A SQL function runs two searches and fuses them with Reciprocal Rank Fusion (k = 60):
   - vector search: cosine similarity on an HNSW index
   - keyword search: full-text match on any query word
   It returns the top 3 products.
4. **No-match guard.** If the best cosine similarity in the top 3 is under a threshold (default 0.55), the answer is `no_match`. The guard uses cosine similarity, not the fusion score, because the fusion score measures rank, not fit.
5. **Evaluation.** A runner scores a labeled set of real posts:
   - retrieval: correct product in the top 3 (pass mark 85%)
   - no-match: posts with no fitting product correctly rejected (pass mark 90%)
   - at least 70% of the set must be Indonesian
   A sweep command finds the best no-match threshold.

## Results

Not measured yet. Scores will be added here after the labeled set of real posts is complete. The included smoke file only tests that the code runs; it is not an accuracy result.

## Run it

Requirements: PostgreSQL 17 with pgvector, Ollama with `qwen3-embedding:0.6b`, and `uv`.

```bash
createdb product_rag
uv sync --extra dev
uv run rag-slice init-db
uv run rag-slice import-catalog --csv data/catalog_demo_id.csv
uv run rag-slice search --query "sabun muka untuk kulit sensitif"
```

The default database URL is `postgresql:///product_rag?host=localhost&port=5433`. Set `RAG_DATABASE_URL` to use another one, and `RAG_GUARD_THRESHOLD` to change the no-match threshold.

## Add labeled posts and score them

```bash
uv run rag-slice catalog                 # list product ids
uv run rag-slice add-post                # paste a real post, pick its product or no_match
uv run rag-slice eval  --file evals/real/posts.jsonl
uv run rag-slice sweep --file evals/real/posts.jsonl --min 0.30 --max 0.80 --step 0.05
```

`add-post` never shows the model's guess before you choose the label, so the labels stay independent of the system being tested. Real posts are kept in `evals/real/`, which is not committed; only the scores are published.

## Tests

```bash
uv run pytest
```

## Limits

- The catalog in `data/` is a fictional demo catalog of 20 Indonesian personal care products.
- Single user and local only: no authentication, multi-tenant isolation, or hosted deployment.
- No document parsing: the catalog comes from CSV.
