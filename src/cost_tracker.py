"""Token usage and cost tracking logger for pipeline execution."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd


@dataclass
class CallRecord:
    timestamp: str
    route: str
    week_of: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    mode: str


class CostTracker:
    """Tracks token volume, invocation count, and dollar costs across pipeline runs."""

    # Published pricing per 1,000,000 tokens (USD)
    RATES = {
        "gpt-4o-mini": {"input_rate": 0.15, "output_rate": 0.60},
        "gpt-4o": {"input_rate": 2.50, "output_rate": 10.00},
        "all-MiniLM-L6-v2": {"input_rate": 0.00, "output_rate": 0.00},
        "template": {"input_rate": 0.00, "output_rate": 0.00},
    }

    def __init__(self):
        self.records: List[CallRecord] = []
        self.embedding_calls: int = 0
        self.llm_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

    def record_embedding_run(self, doc_count: int, query_count: int, model_name: str = "all-MiniLM-L6-v2"):
        """Record local embedding generation (zero API dollar cost)."""
        self.embedding_calls += (doc_count + query_count)
        self.records.append(CallRecord(
            timestamp=datetime.utcnow().isoformat(),
            route="ALL",
            week_of="BATCH",
            model=model_name,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            mode="local_embedding",
        ))

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        route: str = "N/A",
        week_of: str = "N/A",
        mode: str = "llm",
    ):
        """Record an individual LLM or template call."""
        rate_info = self.RATES.get(model, {"input_rate": 0.0, "output_rate": 0.0})
        cost = (input_tokens * rate_info["input_rate"] + output_tokens * rate_info["output_rate"]) / 1_000_000.0

        self.llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        self.records.append(CallRecord(
            timestamp=datetime.utcnow().isoformat(),
            route=route,
            week_of=week_of,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            mode=mode,
        ))

    def export_csv(self, file_path: str | Path):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            df = pd.DataFrame([{
                "timestamp": datetime.utcnow().isoformat(),
                "route": "NONE",
                "week_of": "NONE",
                "model": "template",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "mode": "template_deterministic",
            }])
        else:
            df = pd.DataFrame([r.__dict__ for r in self.records])
        df.to_csv(path, index=False)

    def export_markdown(self, file_path: str | Path):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        md = f"""# Pipeline Cost and Token Consumption Log

- **Run Timestamp (UTC)**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
- **Local Embedding Vector Invocations**: {self.embedding_calls} (Local open-source sentence-transformers)
- **External LLM Invocations**: {self.llm_calls}
- **Total Input Tokens**: {self.total_input_tokens}
- **Total Output Tokens**: {self.total_output_tokens}
- **Total Estimated Cost (USD)**: ${self.total_cost_usd:.6f}

### Accounting Breakdown
| Component | Engine / Model | Calls | Input Tokens | Output Tokens | Published Rate | Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Semantic Retrieval** | `all-MiniLM-L6-v2` | {self.embedding_calls} | Local CPU | Local CPU | Open-source ($0/1M) | $0.000000 |
| **Verdict Gate** | Deterministic Code | All | 0 | 0 | Rule-based ($0/1M) | $0.000000 |
| **Explanation Engine** | {'gpt-4o-mini' if self.llm_calls > 0 else 'Template Explainer'} | {self.llm_calls} | {self.total_input_tokens} | {self.total_output_tokens} | {'$0.15 / $0.60 per 1M' if self.llm_calls > 0 else 'Deterministic ($0/1M)'} | ${self.total_cost_usd:.6f} |
| **Total** | -- | **{self.embedding_calls + self.llm_calls}** | **{self.total_input_tokens}** | **{self.total_output_tokens}** | -- | **${self.total_cost_usd:.6f}** |

> **Audit Note**: In the default production mode, explanations are generated via the deterministic rule-based template engine and local embeddings. Zero tokens are transmitted over external networks and zero API costs are incurred.
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
