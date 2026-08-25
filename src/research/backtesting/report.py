from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import polars as pl

from research.backtesting.portfolio import TradeRecord
from research.report_base import SessionReport
from research.session import SessionConfig

if TYPE_CHECKING:
    from trading.config.settings import AlgoSettings

    from research.backtesting.data_loader import DataLoader


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig(SessionConfig):
    """Configuration for a single backtest run."""

    type: Literal["backtest"] = field(default="backtest", init=False)

    # Required
    algo: AlgoSettings = field(default_factory=lambda: _missing("algo"))
    start: datetime = field(default_factory=lambda: _missing("start"))
    end: datetime = field(default_factory=lambda: _missing("end"))
    loader: DataLoader = field(default_factory=lambda: _missing("loader"))

    # Optional with sensible defaults
    initial_equity: float = 100_000.0
    slippage_pct: float = 0.05
    partial_fill_prob: float = 0.0
    latency_secs: float = 0.0
    replay_delay_secs: float = 0.0
    session_id: str = ""

    # Hyperparameter overrides — forwarded to make_strategy
    strategy_params: dict = field(default_factory=dict)


def _missing(name: str) -> object:
    raise TypeError(f"BacktestConfig: required field {name!r} was not provided")


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------


@dataclass
class BacktestReport(SessionReport):
    """
    Results of a completed backtest session.

    Attributes
    ----------
    config:
        The config that produced this report.
    equity_curve:
        pl.DataFrame with columns [date, equity].
    trades:
        Completed round-trip trades.
    sharpe_ratio, max_drawdown, ... :
        Pre-computed metric values (see ``backtesting.metrics``).
    """

    config: BacktestConfig
    equity_curve: pl.DataFrame
    trades: list[TradeRecord]
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    win_rate: float
    profit_factor: float
    cagr: float
    calmar_ratio: float
    total_trades: int
    final_equity: float

    # SessionReport fields
    session_id: str = ""
    session_type: str = "backtest"
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # SessionReport ABC
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "algo_name": self.config.algo.name,
            "start": self.config.start.isoformat(),
            "end": self.config.end.isoformat(),
            "initial_equity": self.config.initial_equity,
            "final_equity": self.final_equity,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_secs": self.max_drawdown_duration.total_seconds(),
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "cagr": self.cagr,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": [
                [row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]), row["equity"]]
                for row in self.equity_curve.iter_rows(named=True)
            ],
        }

    def _equity_series(self) -> tuple[list, list, list]:
        """Dates, equity values, and the running max used by both equity-curve charts."""
        dates = self.equity_curve["date"].to_list()
        equities = self.equity_curve["equity"].to_list()
        running_max = self.equity_curve["equity"].cum_max().to_list()
        return dates, equities, running_max

    def _build_equity_curve_fig(self, dates: list, equities: list, running_max: list):
        """Equity curve line chart, with drawdown periods shaded."""
        import plotly.graph_objects as go

        eq_fig = go.Figure()
        eq_fig.add_trace(
            go.Scatter(
                x=dates,
                y=equities,
                mode="lines",
                name="Equity",
                line=dict(color="#2196F3", width=2),
            )
        )
        # Shade drawdown periods
        in_dd = False
        dd_start = None
        for i, (eq, rm) in enumerate(zip(equities, running_max, strict=False)):
            if eq < rm and not in_dd:
                in_dd = True
                dd_start = dates[i]
            elif eq >= rm and in_dd:
                in_dd = False
                eq_fig.add_vrect(
                    x0=dd_start,
                    x1=dates[i],
                    fillcolor="rgba(244,67,54,0.15)",
                    line_width=0,
                )
        if in_dd and dd_start is not None:
            eq_fig.add_vrect(
                x0=dd_start,
                x1=dates[-1],
                fillcolor="rgba(244,67,54,0.15)",
                line_width=0,
            )
        eq_fig.update_layout(
            title=f"Equity Curve — {self.config.algo.name}",
            xaxis_title="Date",
            yaxis_title="Equity",
            template="plotly_white",
        )
        return eq_fig

    def _build_drawdown_fig(self, dates: list, equities: list, running_max: list):
        """Drawdown area chart."""
        import plotly.graph_objects as go

        dd_values = [
            -(rm - eq) / rm if rm > 0 else 0.0
            for eq, rm in zip(equities, running_max, strict=False)
        ]
        dd_fig = go.Figure()
        dd_fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd_values,
                fill="tozeroy",
                fillcolor="rgba(244,67,54,0.3)",
                mode="lines",
                line=dict(color="#F44336"),
                name="Drawdown",
            )
        )
        dd_fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis=dict(tickformat=".1%"),
            template="plotly_white",
        )
        return dd_fig

    def _build_trade_pnl_fig(self):
        """Per-trade P&L bar chart, or None if there are no trades."""
        if not self.trades:
            return None

        import plotly.graph_objects as go

        trade_labels = [f"{t.symbol} {t.side}" for t in self.trades]
        trade_pnls = [t.pnl for t in self.trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in trade_pnls]
        pnl_fig = go.Figure()
        pnl_fig.add_trace(
            go.Bar(
                x=list(range(len(trade_pnls))),
                y=trade_pnls,
                marker_color=colors,
                text=trade_labels,
                name="Trade P&L",
            )
        )
        pnl_fig.update_layout(
            title="Trade P&L",
            xaxis_title="Trade #",
            yaxis_title="P&L",
            template="plotly_white",
        )
        return pnl_fig

    def _build_metrics_table_fig(self):
        """Summary metrics table."""
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
                                "Sharpe Ratio",
                                "Max Drawdown",
                                "Win Rate",
                                "Profit Factor",
                                "CAGR",
                                "Calmar Ratio",
                                "Total Trades",
                                "Final Equity",
                            ],
                            [
                                f"{self.sharpe_ratio:.3f}",
                                f"{self.max_drawdown:.2%}",
                                f"{self.win_rate:.2%}",
                                f"{self.profit_factor:.3f}",
                                f"{self.cagr:.2%}",
                                f"{self.calmar_ratio:.3f}",
                                str(self.total_trades),
                                f"{self.final_equity:,.2f}",
                            ],
                        ],
                        fill_color=["#ECEFF1", "#FAFAFA"],
                    ),
                )
            ]
        )
        metrics_fig.update_layout(title="Performance Metrics", template="plotly_white")
        return metrics_fig

    def to_html(self) -> str:
        """Self-contained HTML with embedded Plotly charts."""
        from plotly.io import to_html as plotly_to_html

        dates, equities, running_max = self._equity_series()

        figs = [
            self._build_equity_curve_fig(dates, equities, running_max),
            self._build_drawdown_fig(dates, equities, running_max),
        ]
        pnl_fig = self._build_trade_pnl_fig()
        if pnl_fig is not None:
            figs.append(pnl_fig)
        figs.append(self._build_metrics_table_fig())

        # Combine into one HTML
        parts = [
            plotly_to_html(fig, include_plotlyjs=("cdn" if i > 0 else True), full_html=False)
            for i, fig in enumerate(figs)
        ]
        # Use offline plotly (embed script) only in the first figure
        body = "\n".join(parts)
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Backtest Report — {self.config.algo.name} ({self.session_id})</title>
  <style>body{{font-family:sans-serif;margin:20px}}</style>
</head>
<body>
  <h1>Backtest Report</h1>
  <p><strong>Algo:</strong> {self.config.algo.name} &nbsp;
     <strong>Period:</strong> {self.config.start.date()} → {self.config.end.date()} &nbsp;
     <strong>Session:</strong> {self.session_id}</p>
  {body}
</body>
</html>"""
