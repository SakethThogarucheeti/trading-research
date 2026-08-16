from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import polars as pl

from research.report_base import SessionReport
from research.session import SessionConfig

# ---------------------------------------------------------------------------
# MonteCarloConfig
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloConfig(SessionConfig):
    """Configuration for a Monte Carlo simulation on a set of trade records."""

    type: Literal["monte_carlo"] = field(default="monte_carlo", init=False)

    n_trials: int = 1000
    method: Literal["shuffle", "bootstrap"] = "bootstrap"
    slippage_sigma: float = 0.0  # additional noise on each trade's PnL
    initial_equity: float = 100_000.0
    seed: int | None = None
    session_id: str = ""


# ---------------------------------------------------------------------------
# MonteCarloReport
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloReport(SessionReport):
    """Results of a Monte Carlo simulation."""

    config: MonteCarloConfig
    return_distribution: pl.Series  # per-trial total return fraction
    drawdown_distribution: pl.Series  # per-trial max drawdown fraction
    probability_of_ruin: float  # fraction of trials with >50% equity loss
    percentile_5_return: float
    percentile_95_return: float
    median_drawdown: float
    n_trials: int

    # SessionReport fields
    session_id: str = ""
    session_type: str = "monte_carlo"
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "n_trials": self.n_trials,
            "method": self.config.method,
            "probability_of_ruin": self.probability_of_ruin,
            "percentile_5_return": self.percentile_5_return,
            "percentile_95_return": self.percentile_95_return,
            "median_drawdown": self.median_drawdown,
            "return_distribution": self.return_distribution.to_list(),
            "drawdown_distribution": self.drawdown_distribution.to_list(),
        }

    def to_html(self) -> str:
        import plotly.graph_objects as go
        from plotly.io import to_html as plotly_to_html

        figs: list[go.Figure] = []
        rets = self.return_distribution.to_list()
        dds = self.drawdown_distribution.to_list()

        # 1 — Return distribution histogram
        ret_fig = go.Figure()
        ret_fig.add_trace(
            go.Histogram(
                x=rets,
                nbinsx=50,
                marker_color="#2196F3",
                name="Return",
            )
        )
        ret_fig.add_vline(
            x=self.percentile_5_return,
            line_dash="dash",
            line_color="#F44336",
            annotation_text="5th pct",
        )
        ret_fig.add_vline(
            x=self.percentile_95_return,
            line_dash="dash",
            line_color="#4CAF50",
            annotation_text="95th pct",
        )
        ret_fig.update_layout(
            title="Return Distribution",
            xaxis_title="Total Return",
            yaxis_title="Count",
            xaxis=dict(tickformat=".1%"),
            template="plotly_white",
        )
        figs.append(ret_fig)

        # 2 — Drawdown distribution histogram
        dd_fig = go.Figure()
        dd_fig.add_trace(
            go.Histogram(
                x=dds,
                nbinsx=50,
                marker_color="#FF9800",
                name="Max Drawdown",
            )
        )
        dd_fig.update_layout(
            title="Max Drawdown Distribution",
            xaxis_title="Max Drawdown",
            yaxis_title="Count",
            xaxis=dict(tickformat=".1%"),
            template="plotly_white",
        )
        figs.append(dd_fig)

        # 3 — Ruin probability gauge
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=self.probability_of_ruin * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Ruin Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#F44336"},
                    "steps": [
                        {"range": [0, 1], "color": "#4CAF50"},
                        {"range": [1, 5], "color": "#FF9800"},
                        {"range": [5, 100], "color": "#FFCDD2"},
                    ],
                },
            )
        )
        figs.append(gauge_fig)

        parts = [
            plotly_to_html(fig, include_plotlyjs=(i == 0), full_html=False)
            for i, fig in enumerate(figs)
        ]
        body = "\n".join(parts)
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Monte Carlo Report ({self.session_id})</title>
  <style>body{{font-family:sans-serif;margin:20px}}</style>
</head>
<body>
  <h1>Monte Carlo Report</h1>
  <p><strong>Trials:</strong> {self.n_trials} &nbsp;
     <strong>Method:</strong> {self.config.method} &nbsp;
     <strong>Session:</strong> {self.session_id}</p>
  <p><strong>5th pct return:</strong> {self.percentile_5_return:.2%} &nbsp;
     <strong>95th pct return:</strong> {self.percentile_95_return:.2%} &nbsp;
     <strong>Median drawdown:</strong> {self.median_drawdown:.2%}</p>
  {body}
</body>
</html>"""
