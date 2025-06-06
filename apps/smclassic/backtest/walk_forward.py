import pandas as pd
from typing import List, Dict, Any
from apps.smclassic.backtest.backtest_runner import SMClassicBacktester


class WalkForwardEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback = config["walk_forward"]["lookback"]
        self.horizon = config["walk_forward"]["horizon"]
        self.top_n = config["walk_forward"]["top_n"]

    def run(self) -> List[Dict[str, Any]]:
        results = []
        grouped = self._group_variants()

        for symbol, variants in grouped.items():
            df = pd.read_csv(variants[0]["csv"])
            start = 0

            while start + self.lookback + self.horizon <= len(df):
                train_df = df.iloc[start: start + self.lookback].copy()
                test_df = df.iloc[start + self.lookback: start + self.lookback + self.horizon].copy()

                train_results = []
                for variant in variants:
                    variant["df"] = train_df
                    result = SMClassicBacktester({"variants": [variant]})._run_variant(variant)
                    train_results.append((result, variant))

                top_variants = sorted(train_results, key=lambda x: x[0]["net_pnl"], reverse=True)[:self.top_n]

                for result, variant in top_variants:
                    variant["df"] = test_df
                    test_result = SMClassicBacktester({"variants": [variant]})._run_variant(variant)
                    results.append(test_result)

                start += self.horizon

        return results

    def _group_variants(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for v in self.config["variants"]:
            grouped.setdefault(v["symbol"], []).append(v)
        return grouped
