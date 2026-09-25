"""자산배분(TAA/SAA) 전략: 발표된 규칙 그대로(튜닝 없음). 월말 신호 → 다음 거래일 반영, 월중 비중은 가격 따라 표류.

데이터
- 미국 ETF(배당 포함 수정가, yfinance) 2004~: ~/alpha-data/us_etf.parquet. 늦게 상장한 ETF는 비슷한 ETF로 대체(SUB).
- 한국 상장 ETF(FinanceDataReader 종가, 분배금 제외) 2014-07~: 실제 매매용 대용(KR_MAP).
출처: HAA(Keller&Keuning 2023), VAA(2017), DAA(2018), GEM(Antonacci), GTAA(Faber 2007), 영구 포트폴리오, 위험 균형.
"""
from pathlib import Path

import numpy as np
import pandas as pd

US = Path.home() / "alpha-data" / "us_etf.parquet"
SUB = {"BIL": "SHY", "BND": "AGG", "VEA": "EFA", "VWO": "EEM", "HYG": "LQD", "GSG": "DBC", "VGK": "EFA"}
KR_MAP = {"SPY": "143850", "QQQ": "133690", "IWM": "143850", "EFA": "195930", "VEA": "195930", "EEM": "069500",
          "VWO": "069500", "VGK": "195930", "EWJ": "241180", "VNQ": "182480", "DBC": "132030", "GSG": "132030",
          "GLD": "132030", "TLT": "148070", "IEF": "148070", "LQD": "136340", "HYG": "136340", "AGG": "114260",
          "BND": "114260", "TIP": "148070", "SHY": "153130", "BIL": "153130", "KOSPI": "069500"}
COST = 0.001  # 편도 거래비용(ETF, 거래세 면제)


def us_prices() -> pd.DataFrame:
    p = pd.read_parquet(US)
    for new, old in SUB.items():  # 상장 전 구간은 대체 ETF 수익률로 이어 붙임
        r_new, r_old = p[new].pct_change(), p[old].pct_change()
        r = r_new.where(p[new].notna() & p[new].shift(1).notna(), r_old)
        p[new] = (1 + r.fillna(0)).cumprod() * 100
    return p


def kr_prices(tickers: list[str]) -> pd.DataFrame:
    import FinanceDataReader as fdr
    cols = {}
    for t in tickers:
        code = KR_MAP[t]
        cols[t] = pd.Series(fdr.DataReader(code, "2014-01-01")["Close"], dtype=float)
    return pd.DataFrame(cols).ffill()


def _ret(p: pd.DataFrame, i: int, n: int) -> pd.Series:
    return p.iloc[i] / p.iloc[i - n] - 1


def mom13612w(p, i):
    return 12 * _ret(p, i, 21) + 4 * _ret(p, i, 63) + 2 * _ret(p, i, 126) + _ret(p, i, 252)


def mom_avg(p, i):
    return (_ret(p, i, 21) + _ret(p, i, 63) + _ret(p, i, 126) + _ret(p, i, 252)) / 4


def sma_ratio(p, i, n=210):  # 10개월 ≈ 210거래일
    return p.iloc[i] / p.iloc[i - n + 1: i + 1].mean() - 1


# ── 전략: (가격, 인덱스 i) → {자산: 비중} ──
def haa(p, i):
    off = ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"]
    m = mom_avg(p, i)
    defensive = max(["BIL", "IEF"], key=lambda a: m[a])
    if m["TIP"] <= 0:
        return {defensive: 1.0}
    top = m[off].nlargest(4)
    w: dict[str, float] = {}
    for a, v in top.items():
        k = str(a) if v > 0 else defensive
        w[k] = w.get(k, 0) + 0.25
    return w


def vaa_g4(p, i):
    m = mom13612w(p, i)
    off, de = ["SPY", "EFA", "EEM", "AGG"], ["LQD", "IEF", "SHY"]
    return {str(m[off].idxmax()): 1.0} if (m[off] > 0).all() else {str(m[de].idxmax()): 1.0}


def daa_g12(p, i):
    m = mom13612w(p, i)
    off = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD"]
    de = ["SHY", "IEF", "LQD"]
    b = int((m[["VWO", "BND"]] < 0).sum())
    cash = b / 2
    w: dict[str, float] = {}
    if cash > 0:
        w[str(m[de].idxmax())] = cash
    if cash < 1:
        for a in m[off].nlargest(6).index:
            w[str(a)] = w.get(str(a), 0) + (1 - cash) / 6
    return w


def gem(p, i):
    r12 = _ret(p, i, 252)
    if r12["SPY"] > r12["BIL"]:
        return {"SPY": 1.0} if r12["SPY"] >= r12["EFA"] else {"EFA": 1.0}
    return {"AGG": 1.0}


def gtaa5(p, i):
    assets = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
    s = sma_ratio(p, i)
    w: dict[str, float] = {}
    for a in assets:
        k = a if s[a] > 0 else "BIL"
        w[k] = w.get(k, 0) + 0.2
    return w


def permanent(p, i):
    return {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "BIL": 0.25}


def sixty_forty(p, i):
    return {"SPY": 0.6, "IEF": 0.4}


def risk_parity(p, i):
    assets = ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]
    vol = p[assets].iloc[i - 59: i + 1].pct_change().std()
    inv = 1 / vol
    return {str(a): float(v) for a, v in (inv / inv.sum()).items()}


STRATEGIES = {"HAA": haa, "VAA_G4": vaa_g4, "DAA_G12": daa_g12, "GEM": gem, "GTAA5": gtaa5,
              "영구포트폴리오": permanent, "60/40": sixty_forty, "위험균형": risk_parity}
NEEDS = {"HAA": ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT", "TIP", "BIL"],
         "VAA_G4": ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"],
         "DAA_G12": ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD", "SHY", "IEF", "BND"],
         "GEM": ["SPY", "EFA", "BIL", "AGG"], "GTAA5": ["SPY", "EFA", "IEF", "VNQ", "DBC", "BIL"],
         "영구포트폴리오": ["SPY", "TLT", "GLD", "BIL"], "60/40": ["SPY", "IEF"], "위험균형": ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]}


def backtest(rule, signal_px: pd.DataFrame, trade_px: pd.DataFrame, start: str, end: str | None = None) -> pd.Series:
    """signal_px로 월말 신호, trade_px 수익률로 운용(같으면 순수 미국 버전, 다르면 한국 ETF로 매매)."""
    sp = signal_px.ffill()
    tr = trade_px.ffill().pct_change().fillna(0.0)
    days = tr.index[(tr.index >= pd.Timestamp(start)) & ((end is None) | (tr.index <= pd.Timestamp(end or "2100")))]
    days = pd.DatetimeIndex(days)
    months = pd.Series(days, index=days).groupby(days.to_period("M")).max()
    rebal = set(months.to_numpy())
    w = pd.Series(dtype=float)
    eq, curve = 1.0, []
    for d in days:
        if len(w):
            rr = 1 + tr.loc[d, w.index]
            g = float((w * rr).sum())
            w, eq = w * rr / g, eq * g
        if d in rebal:
            i = int(sp.index.get_indexer([d], method="ffill")[0])
            tgt = pd.Series(rule(sp, i), dtype=float)
            tgt = tgt.groupby(level=0).sum()
            allc = w.index.union(tgt.index)
            turn = float((pd.Series(tgt.reindex(allc)).fillna(0) - pd.Series(w.reindex(allc)).fillna(0)).abs().sum())
            eq *= 1 - turn * COST
            w = tgt
        curve.append(eq)
    return pd.Series(curve, index=days)
