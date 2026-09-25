"""최종 후보 재현 스크립트 (risk-manager 지적 반영: 비용 현실화, 미국 신호 1일 지연 명시, 빌드 과정 저장).

슬리브
- S   : 소형 하위10% + 가치(PER·PBR) + 수익성 상위50%, 분기, KOSPI200 전환(252일) — 슬리피지 1%
- ML  : 그래디언트 부스팅 워크포워드 상위30, 분기 — 슬리피지 1%(보수적), seed 0
- DAA/HAA/PP : 자산배분, 신호=미국 ETF 전일 종가, 매매=한국 상장 ETF
- PEAD: data-engineer 결과가 있으면 포함(research/pead_curves.parquet)
조합 비중과 CPPI 설정은 2026-09-25 사전 선언값 그대로(튜닝 없음).
실행: python -m research.build_final  → research/final_curves.parquet, research/final_table.csv
"""
import json
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd

from research import alloc
from research.combo import blended_cost, combine, cppi
from research.engine import Config, metrics, rotate, run
from research.ml import walk_forward_picks

HERE = Path(__file__).parent
COSTS = {"S": 0.012, "ML": 0.012, "PEAD": 0.012, "DAA": 0.001, "HAA": 0.001, "PP": 0.001}
PLANS = {
    "S40+DAA30+PP30": {"S": .4, "DAA": .3, "PP": .3},
    "S50+DAA50": {"S": .5, "DAA": .5},
    "S70+DAA30": {"S": .7, "DAA": .3},
    "S35+ML35+DAA30": {"S": .35, "ML": .35, "DAA": .3},
    "S30+ML30+DAA20+PP20": {"S": .3, "ML": .3, "DAA": .2, "PP": .2},
}


def sleeves() -> dict[str, pd.Series]:
    us = alloc.us_prices()
    need = sorted({a for v in alloc.NEEDS.values() for a in v} | {"BIL"})
    kr = alloc.kr_prices(need)
    sig = us.shift(1).reindex(kr.index, method="ffill")  # 한국장 마감 시점엔 미국 전일 종가까지만 알 수 있음
    out = {}
    for key, name in (("DAA", "DAA_G12"), ("HAA", "HAA"), ("PP", "영구포트폴리오")):
        cols = alloc.NEEDS[name] + (["BIL"] if "BIL" not in alloc.NEEDS[name] else [])
        out[key] = alloc.backtest(alloc.STRATEGIES[name], sig, pd.DataFrame(kr[cols]), "2016-04-15")
    b = run(Config(name="S", rebalance="quarterly", small_q=0.10, quality_q=0.5), log=False)
    out["S"] = rotate(b["curve"], "S", 252, log=False)["curve"]
    picks = walk_forward_picks(Config(name="ML", rebalance="quarterly"), top_n=30)
    out["ML"] = run(Config(name="ML", strategy="picks", rebalance="quarterly", slippage=0.01, start=min(picks),
                           extra={"picks": picks}), log=False)["curve"]
    pead = HERE / "pead_curves.parquet"
    if pead.exists():
        p = pd.read_parquet(pead)
        col = next((c for c in p.columns if "156" in str(c) or str(c).upper().startswith("PEAD_OP")), p.columns[0])
        out["PEAD"] = p[col].dropna()
    return out


def row(name: str, c: pd.Series) -> dict:
    m, a, o = metrics(c), metrics(pd.Series(c[:"2020-12-31"])), metrics(pd.Series(c["2021-01-01":]))
    return {"name": name, "cagr": m["cagr"], "mdd": m["mdd"], "sharpe": m["sharpe"], "is_cagr": a["cagr"], "is_mdd": a["mdd"],
            "oos_cagr": o["cagr"], "oos_mdd": o["mdd"], "start": str(c.index[0])[:10],
            "yearly": json.dumps(m["yearly"])}


if __name__ == "__main__":
    sl = sleeves()
    start = max(v.index[0] for k, v in sl.items() if k != "PEAD")  # 공통 시작(ML 첫 예측일)
    sl: dict[str, pd.Series] = {k: pd.Series(v[start:]) / float(pd.Series(v[start:]).iloc[0]) for k, v in sl.items()}
    safe = pd.Series(fdr.DataReader("153130", "2014-01-01")["Close"], dtype=float)
    curves: dict[str, pd.Series] = dict(sl)
    rows = [row(k, v) for k, v in sl.items()]
    plans = dict(PLANS)
    if "PEAD" in sl:
        plans["S30+ML20+PEAD20+DAA30"] = {"S": .3, "ML": .2, "PEAD": .2, "DAA": .3}
    for nm, w in plans.items():
        base = combine(sl, w, COSTS)
        protected = cppi(base, safe, cost=blended_cost(w, COSTS))
        for tag, c in (("", base), (" +CPPI", protected)):
            curves[nm + tag] = c
            rows.append(row(nm + tag, c))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "final_table.csv", index=False, encoding="utf-8")
    pd.DataFrame(curves).to_parquet(HERE / "final_curves.parquet")
    pd.set_option("display.width", 200)
    print(t[["name", "cagr", "mdd", "sharpe", "is_cagr", "is_mdd", "oos_cagr", "oos_mdd", "start"]].to_string(index=False))
