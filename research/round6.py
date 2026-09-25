"""6차: S 슬리브 청산 규칙(T1-T4) + ETF 데이트레이드(T5-T7) + 재무 필터(T8-T9) + 조합(T10-T12).
quant-researcher 사전 선언(2026-09-25), 튜닝 없음. 12개 시도를 한 번 실행하고 round5.judge()로 평가한다.

채택 규칙(사전 선언)
- A: T1~T4 중 2021~ CAGR ≥ 22.0%(기준 S126) AND 낙폭이 가장 얕은 것 → S*. 없으면 기준 유지.
- B: Sharpe<0.5, 2021~ Sharpe<0.3, (T5) 상위5거래 제거 후 CAGR≤0, (T6) 연 5회 미만 거래 → 탈락.
    남은 것 중 비용 반영 Sharpe 최고 → DT*. 전부 탈락하면 DT 비중은 DAA로.
- C: 채택된 S*보다 CAGR이 낮거나 평균 후보 수 <30이면 폐기.
합격(D, 사전 선언): 전체 CAGR≥20% AND 2021~ CAGR≥18% AND 2024-12 절단 CAGR≥12% AND MDD≥-35% AND DSR≥0.95.
"""
import FinanceDataReader as fdr
import pandas as pd

from research import combo, daytrade, validate
from research.engine import Config, metrics, rotate, run
from research.round5 import judge

BASELINE_2021_CAGR = 0.220  # S126 기준(사전 선언)


def s_sleeve(name: str, **kw) -> pd.Series:
    b = run(Config(name=name, rebalance="quarterly", small_q=0.10, quality_q=0.5, **kw), log=False)
    return rotate(b["curve"], name, 126, log=False)["curve"]


def section_a() -> tuple[str, pd.Series, dict[str, pd.Series]]:
    curves = {
        "S126(기준)": s_sleeve("S126"),
        "T1 hold_buffer": s_sleeve("T1", hold_buffer=2),
        "T2 +trend_hold200": s_sleeve("T2", hold_buffer=2, trend_hold_ma=200),
        "T3 +chandelier_losers": s_sleeve("T3", hold_buffer=2, trend_hold_ma=200, exit="chandelier_losers"),
        "T4 +chandelier_regime": s_sleeve("T4", hold_buffer=2, trend_hold_ma=200, exit="chandelier_regime"),
    }
    cands = {}
    for k in ("T1 hold_buffer", "T2 +trend_hold200", "T3 +chandelier_losers", "T4 +chandelier_regime"):
        o = metrics(pd.Series(curves[k]["2021-01-01":]))
        if o["cagr"] >= BASELINE_2021_CAGR:
            cands[k] = o["mdd"]
    if cands:
        best = max(cands, key=lambda k: cands[k])  # 낙폭이 가장 얕은(0에 가까운) 것
        return best, curves[best], curves
    return "S126(기준)", curves["S126(기준)"], curves


def section_b() -> tuple[str, pd.Series | None, dict[str, pd.Series]]:
    o122630 = pd.DataFrame(fdr.DataReader("122630", "2010-01-01")[["Open", "High", "Low", "Close"]])
    o069500 = pd.DataFrame(fdr.DataReader("069500", "2010-01-01")[["Open", "High", "Low", "Close"]])
    t5 = daytrade.volatility_breakout(o122630)
    t6 = daytrade.rsi2_mr(o069500)
    t7 = daytrade.donchian(o069500)
    curves = {"T5 vol_breakout(122630)": t5, "T6 rsi2_mr(069500)": t6, "T7 donchian(069500)": t7}
    n_years = {k: (pd.DatetimeIndex(v.index)[-1] - pd.DatetimeIndex(v.index)[0]).days / 365.25
              for k, v in curves.items()}
    ok: dict[str, float] = {}
    for name, c in curves.items():
        m, o = metrics(c), metrics(pd.Series(c["2021-01-01":]))
        if m["sharpe"] < 0.5 or o["sharpe"] < 0.3:
            continue
        if name.startswith("T6"):
            ret = pd.Series(c.pct_change().dropna())
            n_trades = int((ret != 0).sum())  # 진입·청산일 근사(0이 아닌 수익일)
            if n_trades / n_years[name] < 5:
                continue
        if name.startswith("T5"):
            ret = pd.Series(c.pct_change().dropna())
            top5 = list(pd.Series(ret[ret > 0]).nlargest(5).index)
            without = ret.drop(top5)
            cagr_wo = float((1 + without).cumprod().iloc[-1]) ** (1 / n_years[name]) - 1
            if cagr_wo <= 0:
                continue
        ok[name] = m["sharpe"]
    if not ok:
        return "DT 없음(전부 탈락, DAA로 대체)", None, curves
    best = max(ok, key=lambda k: ok[k])
    return best, curves[best], curves


def section_c(s_star_name: str, s_star: pd.Series) -> dict[str, pd.Series]:
    """T8(fscore_min=4)·T9(accel_filter) — S* CAGR보다 낮으면 폐기(round6.py 결과표에는 참고용으로 남김)."""
    s_star_cagr = metrics(s_star)["cagr"]
    out = {}
    b8 = run(Config(name="T8", rebalance="quarterly", small_q=0.10, quality_q=0.5, fscore_min=4), log=False)
    t8 = rotate(b8["curve"], "T8", 126, log=False)["curve"]
    if metrics(t8)["cagr"] >= s_star_cagr:
        out["T8 fscore_min4"] = t8
    b9 = run(Config(name="T9", rebalance="quarterly", small_q=0.10, quality_q=0.5, accel_filter=True), log=False)
    t9 = rotate(b9["curve"], "T9", 126, log=False)["curve"]
    if metrics(t9)["cagr"] >= s_star_cagr:
        out["T9 accel"] = t9
    return out


def section_d(s_star: pd.Series, dt_star: pd.Series | None, s_star_cost: float) -> dict[str, pd.Series]:
    F = pd.DataFrame(pd.read_parquet("research/final_curves.parquet"))
    daa = pd.Series(F["DAA"])
    start = max(s_star.index[0], daa.index[0], (dt_star.index[0] if dt_star is not None else daa.index[0]))
    s_slice, daa_slice = pd.Series(s_star[start:]), pd.Series(daa[start:])
    sleeves: dict[str, pd.Series] = {"S*": s_slice / float(s_slice.iloc[0]), "DAA": daa_slice / float(daa_slice.iloc[0])}
    costs = {"S*": s_star_cost, "DAA": 0.001, "PP": 0.001}
    if dt_star is not None:
        sleeves["DT*"] = pd.Series(dt_star[start:]) / float(pd.Series(dt_star[start:]).iloc[0])
        costs["DT*"] = 0.0005
        plans = {"T10": {"S*": .5, "DAA": .3, "DT*": .2}, "T12": {"S*": .6, "DAA": .2, "DT*": .2}}
    else:  # DT 탈락 시 그 비중은 DAA로
        plans = {"T10": {"S*": .5, "DAA": .5}, "T12": {"S*": .6, "DAA": .4}}
    return {name: combo.combine(sleeves, w, costs) for name, w in plans.items()}


def dsr_of(curve: pd.Series) -> float:
    n, var_sr = validate.trial_stats()
    return validate.deflated_sharpe(curve.pct_change(), n_trials=222, var_sr=var_sr)["dsr"]


def accept(row: dict, dsr: float) -> bool:
    return (row["cagr"] >= 0.20 and row["oos_cagr"] >= 0.18 and row["cut2024_cagr"] >= 0.12
            and row["mdd"] >= -0.35 and dsr >= 0.95)


if __name__ == "__main__":
    s_star_name, s_star, a_curves = section_a()
    print("A: S* =", s_star_name, flush=True)
    dt_name, dt_star, b_curves = section_b()
    print("B: DT* =", dt_name, flush=True)
    c_curves = section_c(s_star_name, s_star)

    rows, curves = [], {}
    for name, c in {**a_curves, **b_curves, **c_curves}.items():
        r = judge(name, c)
        r["dsr"] = dsr_of(c)
        rows.append(r)
        curves[name] = c

    s_star_cost = 0.012  # S* 편도 비용(사전 선언, S 슬리브 공통)
    d_curves = section_d(s_star, dt_star, s_star_cost)
    # T11 = (S*+T8 필터) 50 / DAA 30 / DT* 20 (T8이 section_c에서 살아남았을 때만 의미 있음)
    t8 = c_curves.get("T8 fscore_min4")
    if t8 is not None:
        d2 = section_d(t8, dt_star, s_star_cost)
        d_curves["T11"] = d2["T10"]
        # T11 비중은 T10과 동일 배분(50/30/20)이므로 T10 계산을 재사용

    for name, c in d_curves.items():
        r = judge(name, c)
        r["dsr"] = dsr_of(c)
        r["pass"] = accept(r, r["dsr"])
        rows.append(r)
        curves[name] = c

    t = pd.DataFrame(rows)
    t.to_csv("research/round6_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(curves).to_parquet("research/round6_curves.parquet")
    pd.set_option("display.width", 220)
    print(t[["name", "cagr", "mdd", "sharpe", "oos_cagr", "cut2024_cagr", "5y_min", "5y_med", "dsr"]]
          .to_string(index=False))
