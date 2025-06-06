# apps/smgrid/backtest/backtest_runner.py

import os
import pandas as pd
import plotly.graph_objs as go
from typing import List, Dict, Any
from multiprocessing import Pool

from core.execution.risk_manager import RiskManager
from apps.smgrid.strategy import SMGridStrategy
from core.monitoring.metrics_collector import MetricsCollector


class BacktestResult:
    def __init__(self, symbol: str, variant: Dict[str, Any]):
        self.symbol = symbol
        self.variant = variant
        self.metrics = MetricsCollector()
        self.trades: List[Dict] = []

    def record_trade(self, trade: Dict[str, Any]) -> None:
        self.trades.append(trade)
        self.metrics.record_trade(trade)

    def compute_summary(self) -> Dict[str, Any]:
        summary = self.metrics.compute_summary()
        summary["symbol"] = self.symbol
        summary["variant"] = self.variant
        summary["equity_curve"] = self.metrics.get_equity_curve()
        return summary


class SMGridBacktester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run_all(self) -> List[Dict[str, Any]]:
        variants = self.config["variants"]
        with Pool(processes=min(len(variants), os.cpu_count())) as pool: # type: ignore
            results = pool.map(self._run_variant, variants)
        return results

    def _run_variant(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.read_csv(variant["csv"]).rename(columns=str.lower)
        strategy = SMGridStrategy(variant)
        strategy.load_data(df)
        signals = strategy.generate_signals()

        result = BacktestResult(symbol=variant["symbol"], variant=variant)
        rm = RiskManager(balance=variant["balance"], config={})

        pip_value = variant.get("pip_value", 10)
        max_lot = variant.get("max_lot", 1)
        rr_ratio = variant.get("rr_ratio", 2.0)

        for signal in signals:
            if not signal or "orders" not in signal:
                continue
            for order in signal["orders"]:
                sl = order.get("sl")
                tp = order.get("tp")
                if sl is None or tp is None:
                    continue

                stop_loss_pip = abs(order["price"] - sl)
                lot = rm.compute_lot_size(stop_loss_pips=stop_loss_pip, pip_value=pip_value, max_lot=max_lot)
                pnl = ((tp - order["price"]) if order["direction"] == "buy" else (order["price"] - tp)) * lot * pip_value

                trade = {
                    "entry": order["price"],
                    "tp": tp,
                    "sl": sl,
                    "lot": lot,
                    "pnl": pnl,
                    "r_multiple": abs(pnl / (stop_loss_pip * pip_value * lot)) if stop_loss_pip > 0 else 0,
                    "direction": order["direction"]
                }
                result.record_trade(trade)

        return result.compute_summary()

    def visualize_results(self, summaries: List[Dict[str, Any]]) -> None:
        summary_rows = []

        for summary in summaries:
            equity = summary["equity_curve"]
            if not equity:
                continue
            df = pd.DataFrame(equity)
            df["drawdown"] = df["equity"] - df["equity"].cummax()
            df["drawdown_pct"] = (df["drawdown"] / df["equity"].cummax()) * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["step"], y=df["equity"], name="Equity", line=dict(color='blue')))
            fig.add_trace(go.Scatter(
                x=df["step"], y=df["drawdown_pct"], name="Drawdown",
                yaxis="y2", fill="tozeroy", line=dict(color='red', dash="dot")
            ))
            fig.update_layout(
                title=f"Equity & Drawdown - {summary['symbol']}",
                xaxis_title="Trade",
                yaxis=dict(title="Equity"),
                yaxis2=dict(
                    title="Drawdown (%)",
                    overlaying="y",
                    side="right",
                    range=[-100, 0]
                )
            )
            fig.write_html(f"backtest_{summary['symbol']}.html")

            summary_rows.append({
                "Symbol": summary["symbol"],
                "Trades": summary["total_trades"],
                "Win Rate (%)": round(summary["win_rate"], 2),
                "Net PnL": round(summary["net_pnl"], 2),
                "Avg R": round(summary["avg_r_multiple"], 2),
                "Expectancy": round(summary.get("expectancy", 0), 2),
                "Profit Factor": round(summary.get("profit_factor", 0), 2),
                "Max Drawdown": round(summary.get("max_drawdown", 0), 2),
                "Sharpe Ratio": round(summary.get("sharpe_ratio", 0), 2),
            })

        df_results = pd.DataFrame(summary_rows)
        print("\n📈 Backtest Summary:")
        print(df_results.to_string(index=False))

        with open("summary_table.html", "w") as f:
            f.write("<html><body><h2>Backtest Summary</h2>")
            f.write(df_results.to_html(index=False))
            f.write("</body></html>")
