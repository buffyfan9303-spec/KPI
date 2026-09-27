"""8차: 국내외 문헌 조사 후 미검증 후보 7계열 사전 선언 검증 (quant-researcher fable, 2026-09-27).

출처·주장·근거 등급은 research/round8_sources.md. 규칙은 결과를 보기 전에 여기 고정했다.

데이터
- 미국 ETF 배당 포함 수정가(~/alpha-data/us_etf.parquet, 2004~) + 추가(us_extra.parquet: ^IRX 3개월 T-bill, 실제 레버리지 ETF TQQQ/QLD/SSO/UPRO, SCZ, USD/KRW).
- 한국 ETF 종가(FinanceDataReader, 분배금 제외): KODEX200 069500, KODEX 레버리지 122630, KODEX 단기채권 153130, TIGER 나스닥100 133690.
- 한국 개별주: research/data.py(marcap+DART) 그대로.

후보(사전 선언)
 T1 레버리지 ETF + 200일선(Gayed & Bilello 2016 "Leverage for the Long Run"): 지수 종가 > 200일 SMA면 L배 ETF, 아니면 단기채.
    합성 L배 일수익 = L·r − (L−1)·(T-bill+0.5%)/252 − 0.95%/252. 실제 TQQQ/QLD/SSO/UPRO와 겹치는 구간에서 추적 오차를 보고.
    한국판: KOSPI200(069500 신호) → KODEX 레버리지(122630, 2배) / 단기채.
 T2 변동성 목표 + 추세(Moreira-Muir 2017 + T1): 레버리지 ETF 비중 = min(1, 목표변동성 / (L × 지수 60일 실현변동성)), 200일선 아래면 0. 주 1회 조정.
 T3 가속 듀얼 모멘텀(Engineered Portfolio 2018): SPY·SCZ의 (1+3+6개월 수익) 비교, 둘 다 ≤0이면 TLT/TIP 중 1개월 수익 높은 쪽. 월말.
 T4 BAA(Keller 2022): 카나리아(SPY,VWO,VEA,BND) 13612W 하나라도 ≤0 → 방어(TIP,DBC,BIL,IEF,TLT,LQD,BND) SMA12 상위3(BIL보다 낮으면 BIL),
    아니면 공격 G4(QQQ,VWO,VEA,BND) 상위1 / G12 상위6. 월말.
 T5 GTAA13 AGG3(Faber 2013): 13자산 평균(1,3,6,12개월) 모멘텀 상위 3, 각각 10개월 SMA 위일 때만 보유, 아니면 BIL. 월말.
 T6 한국 월말·월초 효과 + 레버리지(윤주영·김동영 2014, Kim & Kim 2022): 월 마지막 거래일(-1)부터 +3일까지 4거래일만 KODEX 레버리지, 나머지는 단기채.
    (3차에 1배 KODEX200 버전 4.3%가 있음 → 레버리지+채권 주차는 새 규칙.)
 T7 한국 주주환원(자사주 순매입) + 수익성: 시총 상위 500·거래대금 ≥3억·흑자(순이익>0, 영업이익>0) 중
    자사주 순매입 수익률(최근 252일 상장주식수 감소율, 기준가 조정일 제외) 순위 + 영업이익/자산 순위 합산 상위 30, 분기, 슬리피지 0.5%.
    배당 자료(pykrx)는 KRX 로그인 필요로 받지 못해 배당은 뺐다(주주수익률의 절반만 검증).
비용·세금
- 미국 ETF: 매매 편도 0.25%(KIS 기본 온라인 해외 수수료) , 환전 0.2%(진입 시 1회), 원화 환산(USD/KRW, 환헤지 없음).
  양도세: 자본 1억 원 기준 연간 실현이익 − 250만 원 공제 × 22%, 이익은 그해 전부 실현으로 가정(보수적), 손실 이월 없음.
- 한국 ETF: 편도 0.05%(거래세 없음). 한국 개별주: 슬리피지 0.5% + 매도세(연도별).
- 샤프는 무위험 수익률을 뺀 값: 미국 ^IRX(3개월 T-bill), 한국 KODEX 단기채권(153130) 수익률.
검증: 학습 = 시작~2020-12(미국은 2004/2007~), 시험 = 2021~, 공통 학습 2016-04~2020-12, 슬리피지 2배, 상위 2년 손익 비중, DSR(누적 시도), PBO(CSCV).
조합(사전 선언, 최대 4개): 기존 최선 S40+DAA30+PP30 대비
  C1 S40+X30+PP30 (X = 학습 구간 샤프 최고 자산배분 T3~T5), C2 S40+DAA30+LETF30, C3 S30+DAA30+PP20+LETF20, C4 C2 + 변동성 목표 20%.
  LETF = T1 중 학습 구간 샤프 최고(원화·세후).
합격(6·7차와 동일): 전체 ≥20%, 2021~ ≥18%, 2024 절단 ≥12%, MDD ≥ -35%, DSR ≥ 0.95.
실행: python -m research.round8 → research/round8_results.csv, research/round8_curves.parquet
"""
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from research import alloc, combo, data, validate
from research.engine import EXCLUDE, TRIALS, Config, _accounts_at, _rebalance_dates, metrics, run
from research.leverage import rolling_5y_cagr
from research.round7 import pbo_cscv

HERE = Path(__file__).parent
DATA = Path.home() / "alpha-data"
START_KR, END = "2016-04-15", "2026-09-23"
IS_END, OOS_START = "2020-12-31", "2021-01-01"
US_COST, US_FX_COST, KR_ETF_COST = 0.0025, 0.002, 0.0005
LETF_ER, LETF_SPREAD = 0.0095, 0.005
CAPITAL, TAX_FREE, TAX = 1e8, 2.5e6, 0.22
TOP_N7, MIN_AMT7, SLIP7 = 30, 3e8, 0.005


# ---------- 데이터 ----------
@lru_cache(maxsize=1)
def us() -> dict:
    p = alloc.us_prices()
    x = pd.read_parquet(DATA / "us_extra.parquet")
    rf = (x["^IRX"] / 100).reindex(p.index).ffill().fillna(0.0)  # 연율
    return dict(p=p, real=x[["TQQQ", "QLD", "SSO", "UPRO"]].reindex(p.index), scz=x["SCZ"].reindex(p.index),
                fx=x["KRW=X"].reindex(p.index).ffill(), rf=rf)


@lru_cache(maxsize=1)
def kr_etf() -> pd.DataFrame:
    import FinanceDataReader as fdr
    out = {}
    for code in ("069500", "122630", "153130", "133690"):
        out[code] = pd.Series(fdr.DataReader(code, "2014-01-01")["Close"], dtype=float)
    return pd.DataFrame(out).ffill()


def kr_rf() -> pd.Series:
    return kr_etf()["153130"].pct_change().fillna(0.0)


# ---------- T1 / T2 ----------
def synthetic_letf(idx_px: pd.Series, rf: pd.Series, L: float) -> pd.Series:
    """합성 L배 일수익: L·r − (L−1)·(T-bill+스프레드)/252 − 운용보수/252."""
    r = idx_px.pct_change().fillna(0.0)
    return L * r - (L - 1) * (rf.reindex(r.index).fillna(0.0) + LETF_SPREAD) / 252 - LETF_ER / 252


def trend_signal(idx_px: pd.Series, sma: int = 200) -> pd.Series:
    """t 종가 > SMA(t) → t+1부터 보유(shift 1). 앞부분 SMA 미완성은 미보유."""
    on = idx_px > idx_px.rolling(sma).mean()
    return on.shift(1).fillna(False).astype(bool)


def switch_curve(risk_r: pd.Series, cash_r: pd.Series, on: pd.Series, cost: float) -> pd.Series:
    r = np.where(on, risk_r, cash_r)
    sw = on.astype(int).diff().abs().fillna(0.0)
    return pd.Series((1 + r - sw * cost).cumprod(), index=risk_r.index)


def vol_target_letf(idx_px: pd.Series, letf_r: pd.Series, cash_r: pd.Series, L: float, target: float,
                    on: pd.Series | None = None, window: int = 60, cost: float = US_COST) -> pd.Series:
    """레버리지 ETF 비중 w = min(1, target/(L·실현변동성)), 매주 마지막 거래일 판단 → 다음 날 반영. on=False면 0."""
    r1 = idx_px.pct_change()
    vol = r1.rolling(window).std() * np.sqrt(252)
    raw = (target / (L * vol)).clip(upper=1.0)
    ix = pd.DatetimeIndex(idx_px.index)
    wk_last = pd.Series(ix, index=ix).groupby(ix.to_period("W")).transform("max") == ix
    w = raw.where(wk_last).ffill().fillna(0.0)
    if on is not None:
        w = w.where(on.reindex(ix).fillna(False), 0.0)  # on은 이미 shift된 신호
    w = w.shift(1).fillna(0.0)
    ret = w * letf_r + (1 - w) * cash_r - w.diff().abs().fillna(0.0) * cost
    return pd.Series((1 + ret).cumprod(), index=ix)


# ---------- T3 / T4 / T5 (월말 자산배분, alloc.backtest 규칙 형식) ----------
def adm(p: pd.DataFrame, i: int) -> dict:
    m = {a: (alloc._ret(p, i, 21) + alloc._ret(p, i, 63) + alloc._ret(p, i, 126))[a] for a in ("SPY", "SCZ")}
    best = max(m, key=m.get)
    if m[best] > 0:
        return {best: 1.0}
    return {max(("TLT", "TIP"), key=lambda a: alloc._ret(p, i, 21)[a]): 1.0}


def _sma12(p: pd.DataFrame, i: int) -> pd.Series:
    m = p.iloc[i - 252: i + 1].resample("ME").last()  # 최근 13개월 월말(마지막은 현재)
    return p.iloc[i] / m.tail(13).mean() - 1


def _baa(p: pd.DataFrame, i: int, offensive: list[str], top: int) -> dict:
    canary = alloc.mom13612w(p, i)[["SPY", "VWO", "VEA", "BND"]]
    s = _sma12(p, i)
    if (canary <= 0).any():
        de = ["TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "BND"]
        w: dict[str, float] = {}
        for a in s[de].nlargest(3).index:
            k = str(a) if s[a] > s["BIL"] else "BIL"
            w[k] = w.get(k, 0) + 1 / 3
        return w
    return {str(a): 1 / top for a in s[offensive].nlargest(top).index}


def baa_g4(p, i):
    return _baa(p, i, ["QQQ", "VWO", "VEA", "BND"], 1)


def baa_g12(p, i):
    return _baa(p, i, ["SPY", "QQQ", "IWM", "VGK", "EWJ", "VWO", "VNQ", "DBC", "GLD", "TLT", "HYG", "LQD"], 6)


GTAA13 = ["SPY", "IWM", "QQQ", "EFA", "EEM", "VNQ", "DBC", "GLD", "TLT", "IEF", "LQD", "HYG", "TIP"]


def gtaa_agg3(p, i):
    m = alloc.mom_avg(p, i)[GTAA13]
    s = alloc.sma_ratio(p, i)
    w: dict[str, float] = {}
    for a in m.nlargest(3).index:
        k = str(a) if s[a] > 0 else "BIL"
        w[k] = w.get(k, 0) + 1 / 3
    return w


ALLOC_RULES = {"ADM": adm, "BAA_G4": baa_g4, "BAA_G12": baa_g12, "GTAA13_AGG3": gtaa_agg3}


def alloc_curve(name: str, start: str) -> pd.Series:
    d = us()
    p = d["p"].copy()
    p["SCZ"] = d["scz"]
    p = p.dropna(subset=["SCZ"]) if name == "ADM" else p
    alloc.COST = US_COST  # 미국 계좌 매매 비용(편도)
    return alloc.backtest(ALLOC_RULES[name], p, p, start)


# ---------- T6 한국 월말·월초 + 레버리지 ----------
def tom_mask(idx: pd.DatetimeIndex, before: int = 1, after: int = 3) -> pd.Series:
    """수익을 취하는 날 = 각 달의 마지막 before 거래일 + 다음 달 처음 after 거래일. (매수는 그 전날 종가.)"""
    ix = pd.DatetimeIndex(idx)
    pos = pd.Series(np.arange(len(ix)), index=ix)
    last = pos.groupby(ix.to_period("M")).max()
    on = pd.Series(False, index=ix)
    for j in last.to_numpy():
        on.iloc[max(0, j - before + 1): j + after + 1] = True
    return on


def tom_curve(risk: str, before: int = 1, after: int = 3) -> pd.Series:
    e = kr_etf().loc[START_KR:END]
    on = tom_mask(pd.DatetimeIndex(e.index), before, after)
    return switch_curve(e[risk].pct_change().fillna(0.0), e["153130"].pct_change().fillna(0.0), on, KR_ETF_COST * 2)


# ---------- T7 자사주 순매입(주식수 감소) + 수익성 ----------
@lru_cache(maxsize=1)
def buyback_yield() -> pd.DataFrame:
    """날짜×종목: 최근 252거래일 상장주식수 감소율(= −Σ Δlog 주식수). 기준가 조정일(액면분할·증자 등: 원가 등락 ≠ KRX 등락률)은 0 처리."""
    px = data.prices()
    px = px[px["Code"].str.endswith("0") & ~px["Name"].str.contains(EXCLUDE, na=False, regex=True)]
    close = px.pivot_table(index="Date", columns="Code", values="Close", aggfunc="last")
    mcap = px.pivot_table(index="Date", columns="Code", values="Marcap", aggfunc="last")
    chg = px.pivot_table(index="Date", columns="Code", values="ChangesRatio", aggfunc="last") / 100
    shares = (mcap / close).replace([np.inf, -np.inf], np.nan)
    raw = close / close.shift(1)
    adjusted = ((raw / (1 + chg)) - 1).abs() > 0.02
    dlog = np.log(shares).diff().where(~adjusted, 0.0)
    return -dlog.rolling(252, min_periods=200).sum()


def buyback_picks(dates: list[pd.Timestamp], mode: str = "bb+q") -> dict:
    px = data.prices()
    amt20 = px.pivot_table(index="Date", columns="Code", values="Amount", aggfunc="last").rolling(20, min_periods=10).mean()
    by = buyback_yield()
    out = {}
    for d in dates:
        snap = px[px["Date"] == d].set_index("Code")
        u = snap[snap.index.str.endswith("0") & ~snap["Name"].str.contains(EXCLUDE, na=False, regex=True)
                 & (snap["Volume"] > 0) & (amt20.loc[d].reindex(snap.index).fillna(0) >= MIN_AMT7)]
        u = u.nlargest(500, columns="Marcap")
        acc = _accounts_at(d)[["net", "op", "assets"]]
        df = u.join(acc, how="inner")
        df = df[(df["net"] > 0) & (df["op"] > 0)]
        b = by.loc[d].reindex(df.index)
        df = df.assign(bb=b, opa=df["op"] / df["assets"]).dropna(subset=["bb"])
        score = df["bb"].rank(ascending=False)
        if mode == "bb+q":
            score = score + df["opa"].rank(ascending=False)
        elif mode == "bb+q+v":
            score = score + df["opa"].rank(ascending=False) + (df["Marcap"] / df["net"]).rank()
        out[str(d.date())] = [str(c) for c in score.nsmallest(TOP_N7).index]
    return out


def buyback_curve(mode: str, slip: float = SLIP7) -> pd.Series:
    idx = pd.DatetimeIndex(data.returns_wide(data.prices()).loc[START_KR:END].index)
    cfg = Config(name=f"R8 buyback {mode}", rebalance="quarterly", start=START_KR, end=END)
    pk = buyback_picks(_rebalance_dates(idx, cfg), mode)
    cfg = Config(name=cfg.name, rebalance="quarterly", start=START_KR, end=END, strategy="picks", top_n=TOP_N7,
                 min_amount=MIN_AMT7, slippage=slip, extra={"picks": pk})
    return run(cfg, log=False)["curve"]


# ---------- 원화 환산·세금 ----------
def to_krw(curve_usd: pd.Series) -> pd.Series:
    fx = us()["fx"].reindex(curve_usd.index).ffill()
    return curve_usd * fx / float(fx.iloc[0]) * (1 - US_FX_COST)


def after_tax(curve_krw: pd.Series, capital: float = CAPITAL, rate: float = TAX, allowance: float = TAX_FREE) -> pd.Series:
    """매년 말 (그해 이익 − 공제) × 세율 차감. 이익은 그해 전부 실현으로 본다(보수적). 손실 이월 없음.
    미국 상장 ETF: 22%·250만 공제. 국내 상장 해외지수 ETF(133690 등): 배당소득세 15.4%·공제 없음."""
    r = curve_krw.pct_change().fillna(0.0)
    eq, out = capital, []
    for _, seg in r.groupby(pd.DatetimeIndex(r.index).year):
        start_eq = eq
        vals = start_eq * (1 + seg).cumprod()
        gain = float(vals.iloc[-1]) - start_eq
        vals.iloc[-1] -= rate * max(0.0, gain - allowance)
        eq = float(vals.iloc[-1])
        out.append(vals)
    s = pd.concat(out)
    return s / float(s.iloc[0])


# ---------- 평가 ----------
def sharpe_ex(curve: pd.Series, rf_daily: pd.Series) -> float:
    r = curve.pct_change().dropna()
    ex = r - rf_daily.reindex(r.index).fillna(0.0)
    return float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0


def row_of(name: str, c: pd.Series, rf: pd.Series, **extra) -> dict:
    c = c.dropna()
    m, i, o = metrics(c), metrics(pd.Series(c[:IS_END])), metrics(pd.Series(c[OOS_START:]))
    ci = pd.Series(c["2016-04-15":IS_END])
    cut = metrics(pd.Series(c[:"2024-12-31"]))
    lg = pd.Series({k: np.log1p(v) for k, v in m["yearly"].items() if v > -1})
    top2 = float(lg.nlargest(2).sum() / lg.sum()) if lg.sum() > 0 else np.nan
    r5 = rolling_5y_cagr(c).dropna()
    return {"name": name, **extra, "start": str(c.index[0].date()), "cagr": m["cagr"], "mdd": m["mdd"],
            "sharpe_ex": round(sharpe_ex(c, rf), 3), "is_cagr": i["cagr"], "is_sharpe_ex": round(sharpe_ex(pd.Series(c[:IS_END]), rf), 3),
            "is16_cagr": metrics(ci)["cagr"] if len(ci) > 300 else np.nan,
            "oos_cagr": o["cagr"], "oos_mdd": o["mdd"], "cut2024_cagr": cut["cagr"],
            "5y_min": float(r5.min()) if len(r5) else np.nan, "5y_med": float(r5.median()) if len(r5) else np.nan,
            "top2yr_share": round(top2, 2) if pd.notna(top2) else np.nan, "yearly": json.dumps(m["yearly"])}


def trial_stats_trim(lo: float = -2.0, hi: float = 3.0) -> tuple[int, float]:
    import csv
    with open(TRIALS, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]
    sr = np.array([float(r[4]) for r in rows if "진단" not in json.loads(r[1]).get("note", "")])
    sr = sr[(sr > lo) & (sr < hi)]
    return len(sr), float(np.var(sr / np.sqrt(252), ddof=1))


def accept(r: dict, dsr: float) -> bool:
    return (r["cagr"] >= 0.20 and r["oos_cagr"] >= 0.18 and r["cut2024_cagr"] >= 0.12 and r["mdd"] >= -0.35 and dsr >= 0.95)


def _log(name: str, m: dict, note: str) -> None:
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": json.dumps({"name": name, "note": note}, ensure_ascii=False), **m}
    pd.DataFrame([row]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False, encoding="utf-8")


def _pr(r: dict) -> None:
    print(f"{r['name']:44s} {r['start']} CAGR {r['cagr']:+.1%} MDD {r['mdd']:+.1%} SR {r['sharpe_ex']:.2f} "
          f"IS {r['is_cagr']:+.1%} OOS {r['oos_cagr']:+.1%} cut24 {r['cut2024_cagr']:+.1%} top2 {r['top2yr_share']}", flush=True)


if __name__ == "__main__":
    d = us()
    p, rf_us, rf_kr = d["p"], d["rf"], kr_rf()
    rf_us_d = rf_us / 252
    cash_us = p["BIL"].pct_change().fillna(0.0)
    rows, curves = [], {}

    def add(name, c, rf, note, **kw):
        curves[name] = c
        _log("R8 " + name, metrics(c.dropna()), note)
        rows.append(row_of(name, c, rf, **kw))
        _pr(rows[-1])
        return c

    # T1 레버리지 + 200일선 (달러, 세전) / 실제 ETF 추적 오차
    track = {}
    for sig_name, L, real in (("SPY", 2, "SSO"), ("SPY", 3, "UPRO"), ("QQQ", 2, "QLD"), ("QQQ", 3, "TQQQ")):
        syn = synthetic_letf(p[sig_name], rf_us, L)
        rr = d["real"][real].pct_change()
        both = pd.concat([syn, rr], axis=1).dropna()
        track[real] = {"days": len(both), "syn_cagr": float((1 + both.iloc[:, 0]).prod() ** (252 / len(both)) - 1),
                       "real_cagr": float((1 + both.iloc[:, 1]).prod() ** (252 / len(both)) - 1),
                       "corr": float(both.corr().iloc[0, 1])}
        on = trend_signal(p[sig_name])
        c = switch_curve(syn, cash_us, on, US_COST)
        add(f"T1 {sig_name}x{L} +SMA200 (USD)", c, rf_us_d, "8차 LETF 추세", family="T1", ccy="USD", tax="pre")
        add(f"T1 {sig_name}x{L} +SMA200 (KRW 세후)", after_tax(to_krw(c)), rf_kr, "8차 LETF 추세 세후", family="T1", ccy="KRW", tax="post")
        add(f"T1 {sig_name}x{L} +SMA200 slip x2", switch_curve(syn, cash_us, on, US_COST * 2), rf_us_d, "8차 LETF 슬리피지2배", family="T1", ccy="USD", tax="pre")
        add(f"T1 {sig_name}x{L} 보유(USD)", pd.Series((1 + syn).cumprod(), index=p.index), rf_us_d, "8차 LETF 보유 비교", family="T1ref", ccy="USD", tax="pre")
    print("추적 오차(합성 vs 실제):", json.dumps(track, indent=0))
    e = kr_etf().loc[START_KR:END]
    on_kr = trend_signal(kr_etf()["069500"]).reindex(e.index).fillna(False)
    add("T1 KOSPI200x2(122630) +SMA200", switch_curve(e["122630"].pct_change().fillna(0.0), e["153130"].pct_change().fillna(0.0), on_kr, KR_ETF_COST),
        rf_kr, "8차 한국 레버리지 추세", family="T1", ccy="KRW", tax="none")
    c_nq = switch_curve(e["133690"].pct_change().fillna(0.0), e["153130"].pct_change().fillna(0.0),
                        trend_signal(kr_etf()["133690"]).reindex(e.index).fillna(False), KR_ETF_COST)
    add("T1 나스닥100(133690) +SMA200 (세전)", c_nq, rf_kr, "8차 한국 나스닥 추세", family="T1", ccy="KRW", tax="pre")
    add("T1 나스닥100(133690) +SMA200 (세후 15.4%)", after_tax(c_nq, rate=0.154, allowance=0.0), rf_kr, "8차 한국 나스닥 추세 세후", family="T1", ccy="KRW", tax="post")

    # T2 변동성 목표 + 추세
    for sig_name, L, tgt in (("QQQ", 3, 0.25), ("QQQ", 2, 0.20), ("SPY", 2, 0.15)):
        syn = synthetic_letf(p[sig_name], rf_us, L)
        c = vol_target_letf(p[sig_name], syn, cash_us, L, tgt, on=trend_signal(p[sig_name]))
        add(f"T2 {sig_name}x{L} vol{int(tgt*100)} +SMA200 (USD)", c, rf_us_d, "8차 변동성목표 추세", family="T2", ccy="USD", tax="pre")
        add(f"T2 {sig_name}x{L} vol{int(tgt*100)} +SMA200 (KRW 세후)", after_tax(to_krw(c)), rf_kr, "8차 변동성목표 세후", family="T2", ccy="KRW", tax="post")

    # T3~T5 자산배분 (달러 세전 + 원화 세후)
    for name in ALLOC_RULES:
        start = "2008-12-31" if name == "ADM" else "2008-01-02"
        c = alloc_curve(name, start)
        add(f"{name} (USD)", c, rf_us_d, "8차 자산배분", family="alloc", ccy="USD", tax="pre")
        add(f"{name} (KRW 세후)", after_tax(to_krw(c)), rf_kr, "8차 자산배분 세후", family="alloc", ccy="KRW", tax="post")

    # T6 월말·월초 + 레버리지
    for risk, b, a in (("122630", 1, 3), ("122630", 4, 3), ("069500", 1, 3)):
        add(f"T6 TOM(-{b},+{a}) {risk}", tom_curve(risk, b, a), rf_kr, "8차 월말월초", family="T6", ccy="KRW", tax="none")

    # T7 자사주 순매입 + 수익성
    for mode in ("bb+q", "bb", "bb+q+v"):
        add(f"T7 buyback {mode}", buyback_curve(mode), rf_kr, "8차 자사주", family="T7", ccy="KRW", tax="none")
    add("T7 buyback bb+q slip x2", buyback_curve("bb+q", SLIP7 * 2), rf_kr, "8차 자사주 슬리피지2배", family="T7", ccy="KRW", tax="none")

    # 조합: 기존 S·DAA·PP + 새 슬리브 (학습 구간 샤프로만 선택)
    T = pd.DataFrame(rows)
    x_name = T[(T["family"] == "alloc") & (T["ccy"] == "KRW")].nlargest(1, "is_sharpe_ex")["name"].iloc[0]
    letf_name = T[(T["family"] == "T1") & (T["ccy"] == "KRW") & (T["tax"] == "post")].nlargest(1, "is_sharpe_ex")["name"].iloc[0]
    print("X =", x_name, "| LETF =", letf_name, flush=True)
    F = pd.DataFrame(pd.read_parquet(HERE / "final_curves.parquet"))
    start = F.index[0]
    slv = {k: pd.Series(F[k][start:]) / float(F[k][start:].iloc[0]) for k in ("S", "DAA", "PP", "S40+DAA30+PP30")}
    for key, nm in (("X", x_name), ("LETF", letf_name)):
        s = pd.Series(curves[nm]).reindex(slv["S"].index).ffill()
        slv[key] = s / float(s.iloc[0])
    costs = {"S": 0.012, "DAA": 0.001, "PP": 0.001, "X": US_COST, "LETF": US_COST}
    s3 = pd.Series(curves["T1 QQQx3 +SMA200 (KRW 세후)"]).reindex(slv["S"].index).ffill()
    slv["L3"] = s3 / float(s3.iloc[0])
    costs["L3"] = US_COST
    plans = {f"C1 S40+X30+PP30 [{x_name}]": {"S": .4, "X": .3, "PP": .3},
             f"C2 S40+DAA30+LETF30 [{letf_name}]": {"S": .4, "DAA": .3, "LETF": .3},
             f"C3 S30+DAA30+PP20+LETF20 [{letf_name}]": {"S": .3, "DAA": .3, "PP": .2, "LETF": .2},
             "C2b S40+DAA30+L3 30 [QQQx3 +SMA200 KRW 세후, 참고]": {"S": .4, "DAA": .3, "L3": .3}}
    add("기준 S40+DAA30+PP30", slv["S40+DAA30+PP30"], rf_kr, "8차 기준 재계산", family="combo", ccy="KRW", tax="mixed")
    for name, w in plans.items():
        add(name, combo.combine(slv, w, costs), rf_kr, "8차 조합", family="combo", ccy="KRW", tax="mixed")
    from research.engine import vol_target
    c2 = curves[list(plans)[1]]
    add("C4 C2 + vol20", vol_target(c2, "C4", 0.20, log=False)["curve"], rf_kr, "8차 조합 변동성목표", family="combo", ccy="KRW", tax="mixed")

    # DSR · PBO · 표
    T = pd.DataFrame(rows)
    n_tr, var_sr = validate.trial_stats()
    T["dsr"] = [validate.deflated_sharpe(curves[n].dropna().pct_change(), n_tr, var_sr)["dsr"] for n in T["name"]]
    n_tr2, var2 = trial_stats_trim()  # 샤프 ≤-2·≥3 인 파산 수준 시도(7차 급등·급락 이벤트 등)를 뺀 분산: 분산 인플레이션 민감도
    T["dsr_trim"] = [validate.deflated_sharpe(curves[n].dropna().pct_change(), n_tr2, var2)["dsr"] for n in T["name"]]
    T["sr0_annual_trim"] = round(validate.expected_max_sharpe(n_tr2, var2) * np.sqrt(252), 2)
    T["pass"] = [accept(r, ds) for r, ds in zip(T.to_dict("records"), T["dsr"])]
    cand = pd.DataFrame({n: curves[n].pct_change() for n in T[T["family"].isin(["T1", "T2", "alloc", "T6", "T7"])]["name"]}).loc[START_KR:]
    T["pbo_group"] = pbo_cscv(cand.fillna(0.0))
    T["n_trials"] = n_tr
    T = T.sort_values("sharpe_ex", ascending=False).reset_index(drop=True)
    T.to_csv(HERE / "round8_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(curves).to_parquet(HERE / "round8_curves.parquet")
    (HERE / "round8_tracking.json").write_text(json.dumps(track, indent=1), encoding="utf-8")
    pd.set_option("display.width", 260)
    print(f"\n시도 수 {n_tr}, PBO(8차 {len(cand.columns)}개) {T['pbo_group'].iloc[0]:.2f}")
    print(T[["name", "start", "cagr", "mdd", "sharpe_ex", "is_cagr", "oos_cagr", "cut2024_cagr", "5y_min", "top2yr_share", "dsr", "dsr_trim", "pass"]].to_string(index=False))
