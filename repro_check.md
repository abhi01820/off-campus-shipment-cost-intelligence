# Reproducibility Verification Report

## Verification Summary
- **Evaluation Status**:  PASS (100% Identical Outputs)
- **Number of Clean Executions**: 3
- **Environment**: Fixed seed, deterministic sorting, rule-based gate, open-source local embeddings.

## Run Artifacts & Cryptographic SHA-256 Hashes
| Run | File Path | Row Count | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Run 1** | `outputs\repro_run1.csv` | 20 | `b6b869702149bcb7649d25b0074b8c3366b26e23e3462ee1a60fc0c005df912e` |
| **Run 2** | `outputs\repro_run2.csv` | 20 | `b6b869702149bcb7649d25b0074b8c3366b26e23e3462ee1a60fc0c005df912e` |
| **Run 3** | `outputs\repro_run3.csv` | 20 | `b6b869702149bcb7649d25b0074b8c3366b26e23e3462ee1a60fc0c005df912e` |

## Verification Diff
```diff
--- Run 1 vs Run 2
+++ Identical (0 line diffs)

--- Run 2 vs Run 3
+++ Identical (0 line diffs)
```

## Determinism & Zero Look-Ahead Assurances
1. **Zero Randomness in Verdicts**: The decision whether a cost spike is justified is 100% governed by the deterministic `evaluate_candidate_applicability` gate.
2. **Stable In-Memory Retrieval**: Vector cosine similarity over L2-normalized embeddings produces strictly invariant document rankings.
3. **Deterministic Sorting**: Final outputs are explicitly sorted by `(route, week_of)`.
4. **Causal Integrity**: Rolling trailing history strictly excludes current and future weeks.
