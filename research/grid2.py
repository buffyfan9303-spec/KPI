"""2차 검증: 미리 정한 12개 조합 + 워크포워드(학습 2016-04~2020-12 / 시험 2021-01~2026-09).

선정 규칙(결과 보기 전에 고정): 학습 구간에서 MDD ≥ -20%인 조합 중 CAGR 최대. 없으면 Calmar(CAGR/|MDD|) 최대.
합격 기준: 시험 구간 CAGR ≥ 30% 그리고 MDD ≥ -20%, 전체 DSR ≥ 0.95, 슬리피지 2%에서도 CAGR ≥ 25%.
"""
import itertools
import json

import pandas as pd

from research.engine import Config, dual_momentum, etf_returns, metrics, run, vol_target
from research.validate import deflated_sharpe, trial_stats

IS_END, OOS_START = "2020-12-31", "2021-01-01"
COSTS = {"small_value": 0.012, "kospi200": 0.001, "bond": 0.0005}  # 편도: 슬리피지+세금 / ETF는 거래세 면제


def build(stop: float, lookback: int, vt: float, slippage: float = 0.01, log: bool = True) -> dict:
    base = run(Config(name=f"G2 base sl{stop}", rebalance="quarterly", small_q=0.10, quality_q=0.5,
                      stop_loss=stop, slippage=slippage), log=log)
    idx = pd.DatetimeIndex(base["curve"].index)
    assets = {"small_value": base["curve"].pct_change().fillna(0.0),
              "kospi200": etf_returns("069500", idx), "bond": etf_returns("153130", idx)}
    costs = dict(COSTS, small_value=slippage + 0.002)
    dm = dual_momentum(assets, costs, safe="bond", name=f"G2 dual sl{stop} lb{lookback}", lookback=lookback, log=log)
    curve = dm["curve"]
    if vt > 0:
        curve = vol_target(curve, f"G2 vt{vt} sl{stop} lb{lookback}", target=vt, log=log)["curve"]
    return {"curve": curve, "pick": dm["pick"]}


def summarize(curve: pd.Series) -> dict:
    ins, oos = metrics(pd.Series(curve[:IS_END])), metrics(pd.Series(curve[OOS_START:]))
    full = metrics(curve)
    return {"is_cagr": ins["cagr"], "is_mdd": ins["mdd"], "oos_cagr": oos["cagr"], "oos_mdd": oos["mdd"],
            "cagr": full["cagr"], "mdd": full["mdd"], "sharpe": full["sharpe"], "yearly": full["yearly"]}


if __name__ == "__main__":
    rows, curves = [], {}
    for stop, lb, vt in itertools.product([0.0, 0.10, 0.20], [126, 252], [0.0, 0.20]):
        key = f"sl{stop:.2f}_lb{lb}_vt{vt:.2f}"
        out = build(stop, lb, vt)
        curves[key] = out["curve"]
        rows.append({"key": key, **summarize(out["curve"])})
        r = rows[-1]
        print(f'{key}: 학습 {r["is_cagr"]:+.1%}/{r["is_mdd"]:+.1%} | 시험 {r["oos_cagr"]:+.1%}/{r["oos_mdd"]:+.1%} | 전체 {r["cagr"]:+.1%}/{r["mdd"]:+.1%} 샤프 {r["sharpe"]}', flush=True)
    df = pd.DataFrame(rows).set_index("key")
    ok = df[df["is_mdd"] >= -0.20]
    score = pd.Series(ok["is_cagr"]) if len(ok) else pd.Series(df["is_cagr"] / df["is_mdd"].abs())
    chosen = str(score.idxmax())
    c = df.loc[chosen]
    n, var = trial_stats()
    dsr = deflated_sharpe(curves[chosen].pct_change(), n, var)
    stop, lb, vt = float(chosen[2:6]), int(chosen.split("_lb")[1].split("_")[0]), float(chosen.split("_vt")[1])
    hi = summarize(build(stop, lb, vt, slippage=0.02, log=False)["curve"])
    verdict = {
        "시험 CAGR ≥ 30%": bool(c["oos_cagr"] >= 0.30), "시험 MDD ≥ -20%": bool(c["oos_mdd"] >= -0.20),
        "DSR ≥ 0.95": bool(dsr["dsr"] >= 0.95), "슬리피지 2%에서 CAGR ≥ 25%": bool(hi["cagr"] >= 0.25)}
    print("\n선정(학습 구간 규칙):", chosen, "| 학습에서 MDD -20% 이내 조합 수:", len(ok))
    print(f'  시험 구간 CAGR {c["oos_cagr"]:+.1%} MDD {c["oos_mdd"]:+.1%} | 전체 {c["cagr"]:+.1%}/{c["mdd"]:+.1%}')
    print("  DSR", dsr, "\n  슬리피지 2%:", {k: hi[k] for k in ("cagr", "mdd")})
    print("  연도별", c["yearly"])
    print("  합격 판정", json.dumps(verdict, ensure_ascii=False))
    df.to_csv(__file__.replace("grid2.py", "grid2_results.csv"), encoding="utf-8")
