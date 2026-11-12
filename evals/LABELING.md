# Labeling real posts

Real posts live in `evals/real/posts.jsonl`, which is gitignored. Only scores get
published. Collect 30-50 genuine public social media posts; never fabricate
content and strip personally identifiable information before adding a post.

Add one labeled post at a time:

```bash
uv run rag-slice add-post --text "sabun muka ini lembut banget" --lang id --label DEMO-001
```

Omit `--text` for the interactive mode: paste the post (multi-line, finish with an
empty line), pick the language (`id`, `en`, `id-mixed`; default `id`), then choose
the label from the printed catalog. Repeat until you answer `n`.

Ids are generated (`real-001`, `real-002`, ...) continuing from the file. Empty
text, labels that are not a catalog product id or `no_match`, and duplicate texts
are rejected. To inspect the catalog first:

```bash
uv run rag-slice catalog
```

At least 70% of the set must be Indonesian (including mixed). The catalog is a
fictional demo, so real posts will not name its products. Label a post with the
product id that best answers what the post is asking for (for example, a post
asking for a serum for dull skin gets the vitamin C serum). Use `no_match` when
no product in the catalog fits. Include both matching and non-matching examples.

Run the evaluation:

```bash
uv run rag-slice eval --file evals/real/posts.jsonl
uv run rag-slice sweep --file evals/real/posts.jsonl --min 0.30 --max 0.80 --step 0.05
```

The pilot phase is quality control before the full suite of 100 matching and 50
no-match samples. Keep heldout data separate for final validation.
