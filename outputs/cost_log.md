# Pipeline Cost and Token Consumption Log

- **Run Timestamp (UTC)**: 2026-10-14 21:02:52
- **Local Embedding Vector Invocations**: 30 (Local open-source sentence-transformers)
- **External LLM Invocations**: 0
- **Total Input Tokens**: 0
- **Total Output Tokens**: 0
- **Total Estimated Cost (USD)**: $0.000000

### Accounting Breakdown
| Component | Engine / Model | Calls | Input Tokens | Output Tokens | Published Rate | Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Semantic Retrieval** | `all-MiniLM-L6-v2` | 30 | Local CPU | Local CPU | Open-source ($0/1M) | $0.000000 |
| **Verdict Gate** | Deterministic Code | All | 0 | 0 | Rule-based ($0/1M) | $0.000000 |
| **Explanation Engine** | Template Explainer | 0 | 0 | 0 | Deterministic ($0/1M) | $0.000000 |
| **Total** | -- | **30** | **0** | **0** | -- | **$0.000000** |

> **Audit Note**: In the default production mode, explanations are generated via the deterministic rule-based template engine and local embeddings. Zero tokens are transmitted over external networks and zero API costs are incurred.
