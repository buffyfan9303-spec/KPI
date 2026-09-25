"""팩터 전수 검정: 모든 팩터를 문헌 기본값으로 한 번씩(튜닝 없음). 상위 20% 동일가중, 유동성 통과 전 종목.
가격 반전·MAX는 월간, 나머지는 분기 리밸런싱. 학습(2016-04~2020-12)/시험(2021-01~) 나눠 기록."""
import pandas as pd

from research.engine import Config, metrics, run
from research.factors import FACTORS

MONTHLY = {"reversal_1m", "low_max"}

if __name__ == "__main__":
    rows = []
    for name, (_, desc) in FACTORS.items():
        cfg = Config(name=f"F5 {name}", strategy="factor", factor=name, top_n=0, slippage=0.005,
                     rebalance="monthly" if name in MONTHLY else "quarterly", note="팩터 전수(5분위)")
        c = run(cfg)["curve"]
        f, a, o = metrics(c), metrics(pd.Series(c[:"2020-12-31"])), metrics(pd.Series(c["2021-01-01":]))
        rows.append({"factor": name, "desc": desc, "cagr": f["cagr"], "mdd": f["mdd"], "sharpe": f["sharpe"],
                     "is_cagr": a["cagr"], "oos_cagr": o["cagr"], "oos_mdd": o["mdd"]})
        print(f'{name:15s} 전체 {f["cagr"]:+.1%}/{f["mdd"]:+.1%} 샤프 {f["sharpe"]:.2f} | 학습 {a["cagr"]:+.1%} | 시험 {o["cagr"]:+.1%}/{o["mdd"]:+.1%}  ({desc})', flush=True)
    pd.DataFrame(rows).to_csv(__file__.replace("sweep_factors.py", "sweep_factors.csv"), index=False, encoding="utf-8")
