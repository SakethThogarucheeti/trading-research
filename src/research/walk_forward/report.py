from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import polars as pl

from research.report_base import SessionReport
from research.session import SessionConfig

if TYPE_CHECKING:
    from trading.config.settings import AlgoSettings

    from research.backtesting.data_loader import DataLoader
    from research.backtesting.report import BacktestReport


# ---------------------------------------------------------------------------
# WalkForwardConfig
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig(SessionConfig):
    """Configuration for a walk-forward test."""

    type: Literal["walk_forward"] = field(default="walk_forward", init=False)

    # Required — no default; callers must supply these
    algo: AlgoSettings = field(default=None)  # type: ignore[assignment]
    loader: DataLoader = field(default=None)  # type: ignore[assignment]
    symbols: list[str] = field(default_factory=list)
    intervals: list[str] = field(default_factory=lambda: ["1min"])

    # Window parameters (in bars)
    train_bars: int = 200
    test_bars: int = 50
    step_bars: int = 50

    initial_equity: float = 100_000.0
    session_id: str = ""


# ---------------------------------------------------------------------------
# WalkForwardReport
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardReport(SessionReport):
    """Aggregated results of a walk-forward test across multiple windows."""

    config: WalkForwardConfig
    windows: list[BacktestReport]  # one report per test window
    aggregate_sharpe: float
    aggregate_max_drawdown: float
    aggregate_win_rate: float
    combined_equity_curve: pl.DataFrame  # columns: [date, equity]

    # SessionReport fields
    session_id: str = ""
    session_type: str = "walk_forward"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "n_windows": len(self.windows),
            "aggregate_sharpe": self.aggregate_sharpe,
            "aggregate_max_drawdown": self.aggregate_max_drawdown,
            "aggregate_win_rate": self.aggregate_win_rate,
            "windows": [w.to_dict() for w in self.windows],
            "combined_equity_curve": [
                [row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]), row["equity"]]
                for row in self.combined_equity_curve.iter_rows(named=True)
            ],
        }

    def _build_equity_curve_fig(self):
        """Combined equity curve across all windows."""
        import plotly.graph_objects as go

        eq_fig = go.Figure()
        dates = self.combined_equity_curve["date"].to_list()
        equities = self.combined_equity_curve["equity"].to_list()
        eq_fig.add_trace(
            go.Scatter(
                x=dates,
                y=equities,
                mode="lines",
                name="Combined Equity",
                line=dict(color="#2196F3", width=2),
            )
        )
        eq_fig.update_layout(
            title="Combined Walk-Forward Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity",
            template="plotly_white",
        )
        return eq_fig

    def _build_sharpe_fig(self, window_labels: list[str]):
        """Per-window Sharpe ratio bar chart."""
        import plotly.graph_objects as go

        sharpe_vals = [w.sharpe_ratio for w in self.windows]
        sharpe_fig = go.Figure()
        sharpe_fig.add_trace(
            go.Bar(
                x=window_labels,
                y=sharpe_vals,
                marker_color=["#4CAF50" if v > 0 else "#F44336" for v in sharpe_vals],
                name="Sharpe Ratio",
            )
        )
        sharpe_fig.update_layout(
            title="Per-Window Sharpe Ratio",
            xaxis_title="Window",
            yaxis_title="Sharpe",
            template="plotly_white",
        )
        return sharpe_fig

    def _build_drawdown_fig(self, window_labels: list[str]):
        """Per-window max drawdown bar chart."""
        import plotly.graph_objects as go

        mdd_vals = [w.max_drawdown for w in self.windows]
        mdd_fig = go.Figure()
        mdd_fig.add_trace(
            go.Bar(
                x=window_labels,
                y=mdd_vals,
                marker_color="#FF9800",
                name="Max Drawdown",
            )
        )
        mdd_fig.update_layout(
            title="Per-Window Max Drawdown",
            xaxis_title="Window",
            yaxis_title="Drawdown",
            yaxis=dict(tickformat=".1%"),
            template="plotly_white",
        )
        return mdd_fig

    def _build_metrics_table_fig(self):
        """Aggregate metrics table."""
        import plotly.graph_objects as go

        metrics_fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Metric", "Value"], fill_color="#37474F", font=dict(color="white")
                    ),
                    cells=dict(
                        values=[
                            [
                                "Windows",
                                "Aggregate Sharpe",
                                "Aggregate Max DD",
                                "Aggregate Win Rate",
                            ],
                            [
                                str(len(self.windows)),
                                f"{self.aggregate_sharpe:.3f}",
                                f"{self.aggregate_max_drawdown:.2%}",
                                f"{self.aggregate_win_rate:.2%}",
                            ],
                        ],
                    ),
                )
            ]
        )
        metrics_fig.update_layout(title="Aggregate Metrics", template="plotly_white")
        return metrics_fig

    def to_html(self) -> str:
        from plotly.io import to_html as plotly_to_html

        window_labels = [f"W{i + 1}" for i in range(len(self.windows))]
        figs = [
            self._build_equity_curve_fig(),
            self._build_sharpe_fig(window_labels),
            self._build_drawdown_fig(window_labels),
            self._build_metrics_table_fig(),
        ]

        parts = [
            plotly_to_html(fig, include_plotlyjs=(i == 0), full_html=False)
            for i, fig in enumerate(figs)
        ]
        body = "\n".join(parts)
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Walk-Forward Report ({self.session_id})</title>
  <style>body{{font-family:sans-serif;margin:20px}}</style>
</head>
<body>
  <h1>Walk-Forward Report</h1>
  <p><strong>Windows:</strong> {len(self.windows)} &nbsp;
     <strong>Train:</strong> {self.config.train_bars} bars &nbsp;
     <strong>Test:</strong> {self.config.test_bars} bars &nbsp;
     <strong>Session:</strong> {self.session_id}</p>
  {body}
</body>
</html>"""
