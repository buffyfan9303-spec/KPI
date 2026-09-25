"""5차: 연평균 30%(5년) 후보 4개 사전 선언 검증 (quant-researcher·fable 명세, 2026-09-25).

합격(사전 선언): 2021~ CAGR ≥ 25% AND MDD ≥ -40% AND 2024-12 절단 CAGR ≥ 15% AND DSR ≥ 0.9.
S 슬리브는 전환 126일로 고정(사후 선택 편향 제거).
"""
import json
import time

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

from research import alloc
from research.build_final import COSTS
from research.combo import _week_last, combine
from research.engine import TRIALS, Config, metrics, rotate, run
from research.leverage import bootstrap_5y, rolling_5y_cagr


def cond_leverage(curve: pd.Series, lmax: float, borrow: float, target_vol: float = 0.25,
                  cost: float = 0.0005) -> pd.Series:
    """#2+#3: 곡선이 200일선 위일 때만 L = min(lmax, 목표변동성/실현변동성(60일)), 아니면 1배. 주 1회 조정."""
    r = curve.pct_change().fillna(0.0)
    vol = r.rolling(60).std() * np.sqrt(252)
    above = curve > curve.rolling(200).mean()
    wk = _week_last(pd.DatetimeIndex(curve.index))
    eq, L, out = 1.0, 1.0, []
    for d in curve.index:
        eq *= 1 + L * r.loc[d] - max(L - 1, 0) * borrow / 252
        eq = max(eq, 0.0)
        if d in wk:
            v = vol.loc[d]
            new = min(lmax, target_vol / v) if (bool(above.loc[d]) and v > 0) else 1.0
            new = max(new, 1.0)
            eq *= 1 - abs(new - L) * cost
            L = new
        out.append(eq)
    return pd.Series(out, index=curve.index)


def global_mom_2x(us: pd.DataFrame, start: str, borrow: float = 0.04, fee: float = 0.009) -> pd.Series:
    """#5: 월말 12개월 수익 최강 1자산(QQQ·SPY·EEM·GLD·TLT). 그 자산이 200일선 위면 2배, 아니면 SHY."""
    assets = ["QQQ", "SPY", "EEM", "GLD", "TLT"]
    p = us[assets + ["SHY"]].ffill()
    r = p.pct_change().fillna(0.0)
    days = pd.DatetimeIndex(p.index[p.index >= pd.Timestamp(start)])
    month_end = set(pd.Series(days, index=days).groupby(days.to_period("M")).max().to_numpy())
    hold, lev, eq, out = "SHY", 1.0, 1.0, []
    for d in days:
        eq *= 1 + lev * r.at[d, hold] - (lev - 1) * (borrow + fee) / 252
        eq = max(eq, 0.0)
        if d in month_end:
            i = int(p.index.get_indexer([d])[0])
            pa = pd.DataFrame(p[assets])
            m12 = pa.iloc[i] / pa.iloc[i - 252] - 1
            best = str(m12.idxmax())
            pb = pd.Series(p[best])
            sma = pb.iloc[i - 199: i + 1].mean()
            new_hold, new_lev = (best, 2.0) if pb.iloc[i] > sma else ("SHY", 1.0)
            if new_hold != hold:
                eq *= 1 - 0.002
            hold, lev = new_hold, new_lev
        out.append(eq)
    return pd.Series(out, index=days)


SECTORS = ["091160", "091180", "091170", "102970", "117700", "117680", "117460", "140710", "098560", "139270"]


def sector_momentum(start: str) -> pd.Series:
    """#13: 월말 12-1개월 모멘텀 상위 2 섹터 동일가중. 절대 모멘텀 < 0인 슬롯은 단기채."""
    px = pd.DataFrame({c: pd.Series(fdr.DataReader(c, "2014-01-01")["Close"], dtype=float) for c in SECTORS + ["153130"]}).ffill()
    r = px.pct_change().fillna(0.0)
    days = pd.DatetimeIndex(px.index[px.index >= pd.Timestamp(start)])
    month_end = set(pd.Series(days, index=days).groupby(days.to_period("M")).max().to_numpy())
    w = pd.Series({"153130": 1.0})
    eq, out = 1.0, []
    for d in days:
        g = float((w * (1 + r.loc[d, w.index])).sum())
        w, eq = w * (1 + r.loc[d, w.index]) / g, eq * g
        if d in month_end:
            i = int(px.index.get_indexer([d])[0])
            ps = pd.DataFrame(px[SECTORS])
            mom = ps.iloc[i - 21] / ps.iloc[i - 252] - 1
            tgt: dict[str, float] = {}
            for c, v in mom.nlargest(2).items():
                k = str(c) if v > 0 else "153130"
                tgt[k] = tgt.get(k, 0) + 0.5
            new = pd.Series(tgt)
            allc = w.index.union(new.index)
            eq *= 1 - float((new.reindex(allc).fillna(0) - w.reindex(allc).fillna(0)).abs().sum()) * 0.001
            w = new
        out.append(eq)
    return pd.Series(out, index=days)


def judge(name: str, c: pd.Series) -> dict:
    m, o = metrics(c), metrics(pd.Series(c["2021-01-01":]))
    cut = metrics(pd.Series(c[:"2024-12-31"]))
    r5 = rolling_5y_cagr(c).dropna()
    bs = bootstrap_5y(c)
    row = {"name": name, "cagr": m["cagr"], "mdd": m["mdd"], "sharpe": m["sharpe"], "oos_cagr": o["cagr"],
           "oos_mdd": o["mdd"], "cut2024_cagr": cut["cagr"], "5y_min": float(r5.min()) if len(r5) else np.nan,
           "5y_med": float(r5.median()) if len(r5) else np.nan, "p5y30": bs["p_cagr30"], "yearly": json.dumps(m["yearly"])}
    pd.DataFrame([{"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "config": json.dumps({"name": "R5 " + name, "note": "5차 사전선언"}, ensure_ascii=False), **m}]
                 ).to_csv(TRIALS, mode="a", header=False, index=False, encoding="utf-8")
    print(f'{name:44s} CAGR {m["cagr"]:+.1%} MDD {m["mdd"]:+.1%} 샤프 {m["sharpe"]:.2f} | 2021~ {o["cagr"]:+.1%}/{o["mdd"]:+.1%} '
          f'| ~2024 {cut["cagr"]:+.1%} | 5년 최저 {row["5y_min"]:+.1%} 중앙 {row["5y_med"]:+.1%} | P(5년≥30%) {bs["p_cagr30"]:.0%}', flush=True)
    return row


if __name__ == "__main__":
    F = pd.read_parquet("research/final_curves.parquet")
    start = F.index[0]
    b = run(Config(name="S", rebalance="quarterly", small_q=0.10, quality_q=0.5), log=False)
    S126 = rotate(b["curve"], "S126", 126, log=False)["curve"][start:]
    sl = {"S126": S126 / S126.iloc[0], "DAA": F["DAA"], "PP": F["PP"]}
    base = combine(sl, {"S126": .4, "DAA": .3, "PP": .3}, {**COSTS, "S126": 0.012})
    rows = [judge("기준: S126_40+DAA30+PP30 x1", base)]
    for lmax in (1.5, 2.0, 2.5):
        for br in (0.04, 0.08):
            rows.append(judge(f"#1 조건부레버리지 Lmax{lmax} 금리{br:.0%}", cond_leverage(base, lmax, br)))
    us = alloc.us_prices()
    g = global_mom_2x(us, "2005-02-01")
    rows.append(judge("#2 글로벌 단일자산 모멘텀 2x (2005~, 미국)", g / g.iloc[0]))
    g2 = pd.Series(g[start:])
    rows.append(judge("  └ 같은 규칙 2017-10~", g2 / float(g2.iloc[0])))
    b10 = run(Config(name="S10", rebalance="quarterly", small_q=0.10, quality_q=0.5, top_n=10), log=False)
    s10 = rotate(b10["curve"], "S10", 126, log=False)["curve"]
    rows.append(judge("#3 소형가치 집중 10종목 + 전환126", s10 / s10.iloc[0]))
    sm = sector_momentum("2016-04-15")
    rows.append(judge("#4 한국 섹터 ETF 모멘텀 상위2", sm / sm.iloc[0]))
    pd.DataFrame(rows).to_csv("research/round5_results.csv", index=False, encoding="utf-8")
