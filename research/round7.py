"""7차: 코스피+코스닥 전체(상장폐지 포함) × 규모 4단계 × 급등·급락·모멘텀 스타일 (quant-researcher, 2026-09-25).

데이터: research/data.py 그대로(FinanceData/marcap 2015~, 상폐 599종목 포함, DART 재무). 새로 받은 것 없음.
규칙(사전 선언, 결과 보기 전 고정)
- 유니버스: 보통주(코드 끝 0), 스팩·리츠·선박 제외, 20일 평균 거래대금 ≥ 3억, 당일 거래 있음.
- 규모: 시장(ALL/KOSPI/KOSDAQ) 안에서 시총 상위 비율 large 0~10% / mid 10~30% / small 30~60% / micro 60~100%.
- 월별 슬리브(20종목, 리밸런싱일 종가 선택 → 다음 날부터 수익, engine.run picks 방식):
  mom 12-1개월 / h52 52주 신고가 근접 / brk 20일 신고가+거래량비 (급등 계열)
  rev 1개월 반전 / dd5 5일 낙폭 최대 / low52 52주 저가 근접 / ma120 120일선 아래 이격 최대 (급락 계열)
- 이벤트(일별 겹침 포트폴리오): 신호일 t 종가 → t+1 시가 매수 → t+hold 종가 매도. t+1 시가 없으면 미체결.
  하한가(-29.5% 이하)·거래정지일엔 못 팔아 다음 거래 가능일로 미룸(최대 5일 뒤 강제 청산).
  정지 상태로 상장폐지되면 그날 -100%. 포지션 비중 1/max(활성 수, 10). 갭(신호 종가→다음 날 시가)은 따로 기록.
  급등: 당일 +15~+30%, 거래량 ≥ 20일 평균×3.  급락: 당일 -10/-15/-20 ~ -30%(하한가 포함), 거래량 급증 유/무.
  급락+재무: 급락 -15%+거래량 종목 중 PBR 0~1 & 직전 사업보고서 흑자(공시 후 자료만) vs 나머지.
- 급락 회피 필터: 최근 60일 안에 -15% 이상 일간 하락 또는 거래정지일이 있으면 제외. 가치(S 원판, 분기)·모멘텀(M*) 슬리브에 적용.
- 비용: 편도 슬리피지 large 0.5% / mid 1.0% / small 1.2% / micro 1.5%, 코스닥 +0.3%p, 이벤트 1.5%. 매도세 연도별. 슬리피지 2배 민감도.
- 일간 수익률 ±30%(가격제한폭)로 자름. 30% 초과 879건(정지 후 재상장·감자)은 신호에서도 제외.
- 조합용 M* = 급등 계열 슬리브 중 학습 구간(2016-04~2020-12) 샤프 최고, C* = 급락 계열(슬리브+이벤트) 중 학습 구간 샤프 최고.
  시험 구간(2021~) 성과는 선택에 쓰지 않는다.
- 합격(6차와 동일): 전체 ≥20%, 2021~ ≥18%, 2024 절단 ≥12%, MDD ≥ -35%, DSR ≥ 0.95.
실행: python -m research.round7 → research/round7_results.csv, research/round7_curves.parquet
"""
import itertools
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from research import combo, data, validate
from research.engine import EXCLUDE, TRIALS, Config, _accounts_at, _rebalance_dates, metrics, rank_value, regime_signal, run, sell_tax
from research.leverage import rolling_5y_cagr

START, END = "2016-04-15", "2026-09-23"
IS_END, OOS_START = "2020-12-31", "2021-01-01"
MIN_AMT = 3e8
LIMIT = 0.30
LIMIT_DOWN = -0.295
TOP_N = 20
SLIP = {"large": 0.005, "mid": 0.010, "small": 0.012, "micro": 0.015}
KOSDAQ_EXTRA = 0.003
EVENT_SLIP = 0.015
TIERS = {"large": (0.0, 0.10), "mid": (0.10, 0.30), "small": (0.30, 0.60), "micro": (0.60, 1.0)}
MARKETS = ["ALL", "KOSPI", "KOSDAQ"]
SURGE_STYLES = ["mom", "h52", "brk"]
CRASH_STYLES = ["rev", "dd5", "low52", "ma120"]
STYLES = SURGE_STYLES + CRASH_STYLES
PARTICIPATION = 0.05  # 운용 가능 금액: 종목별 일 거래대금의 5%까지
AVOID_DROP, AVOID_WINDOW = -0.15, 60


@lru_cache(maxsize=1)
def panel() -> dict:
    px = data.prices()
    px = px[px["Code"].str.endswith("0") & ~px["Name"].str.contains(EXCLUDE, na=False, regex=True)]
    Rraw = data.returns_wide(px)
    R = Rraw.clip(-LIMIT, LIMIT)
    P = (1 + R.fillna(0.0)).cumprod().where(R.notna().cummax())

    def piv(col):
        return px.pivot_table(index="Date", columns="Code", values=col, aggfunc="last").reindex(index=R.index, columns=R.columns)

    vol, amt = piv("Volume"), piv("Amount")
    last = px.groupby("Code").tail(1).set_index("Code")
    frozen = last[(last["Volume"] == 0) & (last["Date"] < px["Date"].max())]["Date"]  # 정지 상태로 사라진 종목
    pos = R.index.get_indexer(frozen) + 1
    dead_next = pd.Series(R.index[pos[pos < len(R)]], index=frozen.index[pos < len(R)])
    return dict(Rraw=Rraw, R=R, P=P, vol=vol, vol20=vol.shift(1).rolling(20, min_periods=10).mean(),
                amt20=amt.rolling(20, min_periods=10).mean(), mcap=piv("Marcap"), open=piv("Open"), close=piv("Close"),
                market=px.groupby("Code")["Market"].last(), dead_next=dead_next, delisted=set(last[last["Date"] < px["Date"].max()].index))


def eligible(pn: dict, d: pd.Timestamp, market: str) -> pd.Index:
    ok = (pn["vol"].loc[d] > 0) & (pn["amt20"].loc[d] >= MIN_AMT) & pn["mcap"].loc[d].notna()
    if market != "ALL":
        ok &= pn["market"].reindex(ok.index) == market
    return ok[ok].index


def tier_codes(pn: dict, d: pd.Timestamp, market: str, tier: str) -> pd.Index:
    mc = pn["mcap"].loc[d, eligible(pn, d, market)].sort_values(ascending=False)
    lo, hi = TIERS[tier]
    return mc.index[int(lo * len(mc)): int(hi * len(mc))]


@lru_cache(maxsize=4)
def tier_matrix(market: str) -> pd.DataFrame:
    """날짜×종목: 그날 시장 안 유동성 통과 종목의 시총 상위 비율(0=최대). 이벤트를 규모별로 나눌 때 사용."""
    pn = panel()
    ok = (pn["vol"] > 0) & (pn["amt20"] >= MIN_AMT) & pn["mcap"].notna()
    if market != "ALL":
        ok &= pn["market"].reindex(ok.columns).eq(market).to_numpy()[None, :]
    return pn["mcap"].where(ok).rank(axis=1, ascending=False, pct=True)


def score(pn: dict, d: pd.Timestamp, codes: pd.Index, style: str) -> pd.Series:
    """d 종가까지 자료만 사용. 값이 클수록 좋음."""
    P = pn["P"]
    i = int(P.index.get_loc(d))
    if i < 252:
        return pd.Series(dtype=float)
    if style == "mom":
        s = P.iloc[i - 21] / P.iloc[i - 252] - 1
    elif style == "h52":
        s = P.iloc[i] / P.iloc[i - 251: i + 1].max()
    elif style == "brk":  # 20일 신고가에 있는 종목만, 거래량/20일 평균 순
        at_high = P.iloc[i] >= P.iloc[i - 19: i + 1].max() * 0.999
        s = (pn["vol"].iloc[i] / pn["vol20"].iloc[i]).where(at_high)
    elif style == "rev":
        s = -(P.iloc[i] / P.iloc[i - 21] - 1)
    elif style == "dd5":
        s = -(P.iloc[i] / P.iloc[i - 5] - 1)
    elif style == "low52":
        s = -(P.iloc[i] / P.iloc[i - 251: i + 1].min())
    elif style == "ma120":
        s = -(P.iloc[i] / P.iloc[i - 119: i + 1].mean() - 1)
    else:
        raise ValueError(style)
    return pd.Series(s).reindex(codes).replace([np.inf, -np.inf], np.nan).dropna()


def avoid_matrix(pn: dict) -> pd.DataFrame:
    """급락 회피: 최근 60일에 -15% 이하 일간 하락 또는 (상장 중) 거래정지일이 있으면 True. 관리종목 지정 자료는 없어 대용."""
    crash = pn["Rraw"].rolling(AVOID_WINDOW, min_periods=1).min() <= AVOID_DROP
    halt = ((pn["vol"] == 0) & pn["P"].notna()).rolling(AVOID_WINDOW, min_periods=1).max() > 0
    return (crash | halt).fillna(False)


def picks(pn: dict, dates: list[pd.Timestamp], market: str, tier: str, style: str, n: int = TOP_N,
          avoid: pd.DataFrame | None = None) -> tuple[dict, float]:
    """리밸런싱일 → 종목 목록. 두 번째 값은 운용 가능 금액(억원): 종목 수 × 선정 종목 거래대금 중앙값 × 참여율."""
    out, caps = {}, []
    for d in dates:
        codes = tier_codes(pn, d, market, tier)
        if avoid is not None:
            codes = codes[~avoid.loc[d].reindex(codes).fillna(False).to_numpy()]
        s = score(pn, d, codes, style)
        sel = [str(c) for c in s.nlargest(n).index]
        out[str(d.date())] = sel
        if sel:
            caps.append(float(pn["amt20"].loc[d, sel].median()) * len(sel) * PARTICIPATION / 1e8)
    return out, float(np.median(caps)) if caps else np.nan


def slip_of(market: str, tier: str) -> float:
    return SLIP[tier] + (KOSDAQ_EXTRA if market == "KOSDAQ" else 0.0)


def _run_picks(name: str, pk: dict, slip: float, rebalance: str, note: str) -> pd.Series:
    cfg = Config(name=name, start=START, end=END, rebalance=rebalance, strategy="picks", top_n=TOP_N, min_amount=MIN_AMT,
                 slippage=slip, extra={"picks": pk})
    out = run(cfg, log=False)  # picks 사전은 크니 trials.csv에는 요약만 남긴다
    _log(name, out["metrics"], note)
    return out["curve"]


def sleeve(market: str, tier: str, style: str, slip_mult: float = 1.0, avoid: pd.DataFrame | None = None) -> tuple[pd.Series, float]:
    pn = panel()
    idx = pd.DatetimeIndex(pn["R"].loc[START:END].index)
    pk, cap = picks(pn, _rebalance_dates(idx, Config(rebalance="monthly", start=START, end=END)), market, tier, style, avoid=avoid)
    name = f"R7 {market}-{tier}-{style}" + (f" slip x{slip_mult:g}" if slip_mult != 1 else "") + (" -avoid" if avoid is not None else "")
    return _run_picks(name, pk, slip_of(market, tier) * slip_mult, "monthly", f"7차 {style} {tier} {market}"), cap


def value_sleeve(avoid: pd.DataFrame | None = None) -> pd.Series:
    """S 원판(소형 하위10%+PER·PBR+수익성 상위50%, 분기, 전환 없음)을 engine.rank_value로 고르고 급락 회피 필터만 얹는다."""
    pn = panel()
    px = data.prices()
    cfg = Config(rebalance="quarterly", small_q=0.10, quality_q=0.5, start=START, end=END)
    amt20 = px.pivot_table(index="Date", columns="Code", values="Amount", aggfunc="last").rolling(20, min_periods=10).mean()
    pk = {}
    for d in _rebalance_dates(pd.DatetimeIndex(pn["R"].loc[START:END].index), cfg):
        snap = px[px["Date"] == d].set_index("Code")
        s = rank_value(d, snap, amt20.loc[d], cfg)
        if avoid is not None:
            s = s[~avoid.loc[d].reindex(s.index).fillna(False).to_numpy()]
        pk[str(d.date())] = [str(c) for c in s.nsmallest(TOP_N).index]
    name = "R7 S(value) " + ("-avoid" if avoid is not None else "base")
    return _run_picks(name, pk, 0.01, "quarterly", "7차 가치+급락회피")


def _log(name: str, m: dict, note: str) -> None:
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": json.dumps({"name": name, "note": note}, ensure_ascii=False), **m}
    pd.DataFrame([row]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False, encoding="utf-8")


# ---------- 급등·급락·돌파 이벤트(일별 겹침 포트폴리오) ----------

def _market_mask(pn: dict, sig: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "ALL":
        return sig
    return sig & pn["market"].reindex(sig.columns).eq(market).to_numpy()[None, :]


def surge_signal(pn: dict, market: str = "ALL", lo: float = 0.15, hi: float = LIMIT + 0.001, vol_mult: float = 3.0) -> pd.DataFrame:
    """t 종가 기준: 당일 등락률 lo~hi, 거래량 ≥ 직전 20일 평균×vol_mult, 유동성 통과. hi 초과는 자료 오류·재상장으로 보고 제외."""
    sig = (pn["Rraw"] >= lo) & (pn["Rraw"] <= hi) & (pn["vol"] >= vol_mult * pn["vol20"]) & (pn["amt20"] >= MIN_AMT)
    return _market_mask(pn, sig, market).fillna(False)


def crash_signal(pn: dict, market: str = "ALL", drop: float = -0.15, vol_mult: float | None = 3.0,
                 tier: str | None = None) -> pd.DataFrame:
    """t 종가 기준: 당일 등락률 -30%(하한가 포함)~drop, (선택) 거래량 급증, 유동성 통과. 규모 구간은 그날 시총 순위로."""
    sig = (pn["Rraw"] <= drop) & (pn["Rraw"] >= -LIMIT - 0.001) & (pn["amt20"] >= MIN_AMT)
    if vol_mult:
        sig &= pn["vol"] >= vol_mult * pn["vol20"]
    if tier:
        lo, hi = TIERS[tier]
        pct = tier_matrix(market)
        sig &= (pct > lo) & (pct <= hi)
    return _market_mask(pn, sig, market).fillna(False)


def breakout_signal(pn: dict, window: int = 60, vol_mult: float = 2.0) -> pd.DataFrame:
    """t 종가가 window일 신고가(연속 신고가는 첫날만) + 거래량 ≥ 20일 평균×vol_mult."""
    P = pn["P"]
    new_high = P >= P.rolling(window, min_periods=window).max()
    fresh = new_high & ~new_high.shift(1, fill_value=False)
    return (fresh & (pn["vol"] >= vol_mult * pn["vol20"]) & (pn["amt20"] >= MIN_AMT)).fillna(False)


@lru_cache(maxsize=1)
def fundamentals_good() -> pd.DataFrame:
    """날짜×종목: 그 달 첫 거래일에 공개돼 있던 최신 사업보고서로 PBR 0~1 & 흑자. 월 단위 갱신(보수적, 미래 참조 없음)."""
    pn = panel()
    idx = pn["R"].index
    firsts = pd.Series(idx, index=idx).groupby(idx.to_period("M")).min()
    eq, net = {}, {}
    for d in firsts:
        a = _accounts_at(d)
        eq[d], net[d] = a["equity"], a["net"]
    eq = pd.DataFrame(eq).T.reindex(index=idx, columns=pn["R"].columns).ffill()
    net = pd.DataFrame(net).T.reindex(index=idx, columns=pn["R"].columns).ffill()
    pbr = pn["mcap"] / eq
    return ((pbr > 0) & (pbr < 1) & (net > 0)).fillna(False)


def event_curve(pn: dict, sig: pd.DataFrame, hold: int, slip: float = EVENT_SLIP, max_pos: int = 10, max_ext: int = 5) -> tuple[pd.Series, dict]:
    """신호 t → t+1 시가 매수(같은 날 시가→종가 수익은 원가 비율) → 이후 일간 수익률 → t+hold 종가 매도.
    t+hold에 하한가·정지면 다음 거래 가능일까지 보유(최대 max_ext일, 그 뒤 강제 청산). 정지 후 상장폐지는 -100%.
    포지션 비중 1/max(활성 수, max_pos): 10개 미만이면 나머지 현금(일별 균등 재배분 근사, Jegadeesh-Titman 겹침 포트폴리오)."""
    R, op, cl = pn["R"], pn["open"], pn["close"]
    entry = (cl / op - 1).clip(-LIMIT, LIMIT).where(op > 0)
    gap = (op / cl.shift(1) - 1).clip(-LIMIT, LIMIT).where(op > 0)
    sig = sig.loc[START:END].reindex(index=R.index, fill_value=False)
    ent = sig & entry.shift(-1).notna()  # 다음 날 시가가 없으면(정지) 체결 안 됨
    R_ev = R.fillna(0.0)
    for code, nd in pn["dead_next"].items():
        if code in R_ev.columns:
            R_ev.at[nd, code] = -1.0
    stuck = ((pn["Rraw"] <= LIMIT_DOWN) | (pn["vol"] == 0) | pn["Rraw"].isna()).to_numpy()  # 그날 못 판다
    E, Rv, En = ent.to_numpy(), R_ev.to_numpy(np.float32), np.nan_to_num(entry.to_numpy(np.float32))
    T = len(R)
    tax = np.array([sell_tax(d) for d in R.index], dtype=np.float32)[:, None]

    def sh(M, k):  # k일 뒤로 밀기(위쪽은 False)
        out = np.zeros_like(M)
        out[k:] = M[:-k]
        return out

    tot, n_act, ev_log = np.zeros(T), np.zeros(T), np.zeros((T, R.shape[1]), dtype=np.float32)
    dead_hits, A = 0, None
    K = hold + max_ext
    for k in range(1, K + 1):
        A = sh(E, k) if k <= hold else (sh(A, 1) & sh(stuck, 1))
        if not A.any():
            break
        r = np.where(A, (En - slip) if k == 1 else Rv, 0.0).astype(np.float32)
        if k >= hold:
            sell = A & (~stuck | (k == K))
            r = r - np.where(sell, slip + tax, 0.0)
        tot += r.sum(1)
        n_act += A.sum(1)
        ev_log[:T - k] += np.log1p(np.clip(r, -0.999, None))[k:]
        dead_hits += int((A & (Rv <= -0.999)).sum())
    port = pd.Series(tot / np.maximum(n_act, max_pos), index=R.index)
    curve = (1 + port.loc[START:END]).cumprod()
    ev_ret = pd.DataFrame(np.expm1(ev_log), index=R.index, columns=R.columns).where(ent).loc[START:END].stack()
    g = gap.shift(-1).where(ent).loc[START:END].stack()
    delisted_share = float(pd.Series(ev_ret.index.get_level_values(1)).isin(pn["delisted"]).mean()) if len(ev_ret) else np.nan
    info = {"n_events": int(len(ev_ret)), "gap_mean": round(float(g.mean()), 4), "ev_ret_mean": round(float(ev_ret.mean()), 4),
            "hit_rate": round(float((ev_ret > 0).mean()), 3), "invested": round(float((n_act[R.index.get_indexer(curve.index)] > 0).mean()), 2),
            "delisted_in_hold": dead_hits, "delisted_share": round(delisted_share, 3)}
    return curve, info


# ---------- 평가 ----------

def apply_timing(curve: pd.Series, slip: float) -> pd.Series:
    """KODEX200 > 200일선(월말 판단, 다음 날 적용)일 때만 보유, 아니면 현금. 전환마다 왕복 슬리피지+매도세."""
    bull = regime_signal(pd.DatetimeIndex(curve.index))
    r = curve.pct_change().fillna(0.0)
    sw = bull.astype(float).diff().abs().fillna(0.0)
    out = r.where(bull, 0.0) - sw * (2 * slip + 0.002)
    return (1 + out).cumprod()


def row_of(name: str, c: pd.Series, **extra) -> dict:
    m, i, o = metrics(c), metrics(pd.Series(c[:IS_END])), metrics(pd.Series(c[OOS_START:]))
    cut = metrics(pd.Series(c[:"2024-12-31"]))
    lg = pd.Series({k: np.log1p(v) for k, v in m["yearly"].items() if v > -1})
    top2 = float(lg.nlargest(2).sum() / lg.sum()) if lg.sum() > 0 else np.nan
    r5 = rolling_5y_cagr(c).dropna()
    return {"name": name, **extra, "cagr": m["cagr"], "mdd": m["mdd"], "sharpe": m["sharpe"],
            "is_cagr": i["cagr"], "is_sharpe": i["sharpe"], "oos_cagr": o["cagr"], "oos_mdd": o["mdd"],
            "cut2024_cagr": cut["cagr"], "5y_min": float(r5.min()) if len(r5) else np.nan,
            "top2yr_share": round(top2, 2) if pd.notna(top2) else np.nan,
            "concentrated": bool(pd.notna(top2) and top2 > 0.7), "yearly": json.dumps(m["yearly"])}


def pbo_cscv(rets: pd.DataFrame, n_blocks: int = 16) -> float:
    """Bailey et al.(2017) CSCV: 블록 절반을 학습으로 써서 고른 최고 전략이 나머지 절반에서 중앙값 이하일 확률."""
    X = rets.dropna().to_numpy()
    blocks = np.array_split(np.arange(len(X)), n_blocks)
    s1 = np.stack([X[b].sum(0) for b in blocks])
    s2 = np.stack([(X[b] ** 2).sum(0) for b in blocks])
    n = np.array([len(b) for b in blocks], dtype=float)

    def sharpe(ix):
        nn = n[ix].sum()
        mu = s1[ix].sum(0) / nn
        return mu / np.sqrt(np.maximum(s2[ix].sum(0) / nn - mu ** 2, 1e-18))

    lam = []
    for comb in itertools.combinations(range(n_blocks), n_blocks // 2):
        ins = np.array(comb)
        oos = np.setdiff1d(np.arange(n_blocks), ins)
        best = int(np.argmax(sharpe(ins)))
        so = sharpe(oos)
        w = (np.sum(so < so[best]) + 0.5) / (len(so) + 1)  # 시험 구간 순위(0~1)
        lam.append(np.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


def accept(r: dict, dsr: float) -> bool:
    return (r["cagr"] >= 0.20 and r["oos_cagr"] >= 0.18 and r["cut2024_cagr"] >= 0.12 and r["mdd"] >= -0.35 and dsr >= 0.95)


def _pr(r: dict, extra: str = "") -> None:
    print(f"{r['name']:40s} CAGR {r['cagr']:+.1%} MDD {r['mdd']:+.1%} 샤프 {r['sharpe']:.2f} IS {r['is_cagr']:+.1%} "
          f"OOS {r['oos_cagr']:+.1%} {extra}", flush=True)


if __name__ == "__main__":
    pn = panel()
    rows, curves = [], {}

    def add(name, c, **kw):
        curves[name] = c
        rows.append(row_of(name, c, **kw))
        return rows[-1]

    # A. 규모 × 스타일 × 시장 (월별 엔진 슬리브)
    for market, tier, style in itertools.product(MARKETS, TIERS, STYLES):
        c, cap = sleeve(market, tier, style)
        r = add(f"{market}-{tier}-{style}", c, kind="sleeve", family="surge" if style in SURGE_STYLES else "crash",
                market=market, tier=tier, style=style, hold=21, slip=slip_of(market, tier), capacity_ukwon=round(cap, 1))
        _pr(r, f"cap {cap:.0f}억")

    # B. 이벤트: 급등 / 돌파 / 급락(문턱·거래량·보유) / 급락 규모별 / 급락+재무
    events: dict[str, tuple[pd.DataFrame, int, str, str, str]] = {}
    for market in MARKETS:
        for hold in (1, 5, 20):
            events[f"surge15_{market}_h{hold}"] = (surge_signal(pn, market), hold, "surge", market, "all")
    for window in (20, 60):
        for hold in (10, 20):
            events[f"breakout{window}_ALL_h{hold}"] = (breakout_signal(pn, window), hold, "surge", "ALL", "all")
    for drop in (-0.10, -0.15, -0.20):
        for vm in (3.0, None):
            for hold in (1, 5, 20):
                events[f"crash{int(drop*100)}{'v' if vm else ''}_ALL_h{hold}"] = (crash_signal(pn, "ALL", drop, vm), hold, "crash", "ALL", "all")
    for market in ("KOSPI", "KOSDAQ"):
        for hold in (1, 5, 20):
            events[f"crash-15v_{market}_h{hold}"] = (crash_signal(pn, market, -0.15, 3.0), hold, "crash", market, "all")
    for tier in TIERS:
        events[f"crash-15v_ALL-{tier}_h5"] = (crash_signal(pn, "ALL", -0.15, 3.0, tier), 5, "crash", "ALL", tier)
    good = fundamentals_good()
    base_crash = crash_signal(pn, "ALL", -0.15, 3.0)
    for hold in (5, 20):
        events[f"crash-15v+good(PBR<1,흑자)_ALL_h{hold}"] = (base_crash & good, hold, "crash", "ALL", "all")
        events[f"crash-15v+rest_ALL_h{hold}"] = (base_crash & ~good, hold, "crash", "ALL", "all")
    for name, (sig, hold, fam, market, tier) in events.items():
        c, info = event_curve(pn, sig, hold)
        _log(f"R7 event {name}", metrics(c), "7차 이벤트")
        c2, _ = event_curve(pn, sig, hold, EVENT_SLIP * 2)
        _log(f"R7 event {name} slip x2", metrics(c2), "7차 이벤트")
        r = add(name, c, kind="event", family=fam, market=market, tier=tier, style=name.split("_")[0], hold=hold,
                slip=EVENT_SLIP, cagr_slip2x=metrics(c2)["cagr"], **info)
        _pr(r, f"x2 {r['cagr_slip2x']:+.1%} 건 {info['n_events']} 갭 {info['gap_mean']:+.2%} 건당 {info['ev_ret_mean']:+.2%} 승률 {info['hit_rate']:.0%}")

    # C. 급락 회피 필터: 가치(S 원판) + 모멘텀(M*)
    T = pd.DataFrame(rows)
    sl = T[T["kind"] == "sleeve"]
    m_star = sl[sl["family"] == "surge"].nlargest(1, "is_sharpe")["name"].iloc[0]
    c_star = T[T["family"] == "crash"].nlargest(1, "is_sharpe")["name"].iloc[0]
    print("M* =", m_star, "| C* =", c_star, flush=True)
    avoid = avoid_matrix(pn)
    add("S(value) base", value_sleeve(), kind="filter", family="value", market="ALL", tier="small", style="value", hold=63, slip=0.01)
    add("S(value) -avoid", value_sleeve(avoid), kind="filter", family="value", market="ALL", tier="small", style="value -avoid", hold=63, slip=0.01)
    mk, tr, st = m_star.split("-")
    c, _ = sleeve(mk, tr, st, avoid=avoid)
    add(m_star + " -avoid", c, kind="filter", family="surge", market=mk, tier=tr, style=st + " -avoid", hold=21, slip=slip_of(mk, tr))
    for r in rows[-3:]:
        _pr(r)

    # D. 슬리피지 2배: 전체 샤프 상위 5 슬리브 재실행
    for name in sl.nlargest(5, "sharpe")["name"]:
        mk, tr, st = name.split("-")
        c2, _ = sleeve(mk, tr, st, slip_mult=2.0)
        for r in rows:
            if r["name"] == name:
                r["cagr_slip2x"] = metrics(c2)["cagr"]

    # E. 시장 타이밍(200일선) — M*, C*
    for name in (m_star, c_star):
        c = apply_timing(curves[name], EVENT_SLIP)
        _log(f"R7 {name} +timing200", metrics(c), "7차 타이밍")
        add(name + " +timing200", c, kind="timing", family="timing", market="ALL", tier="mix", style="timing", hold=21, slip=EVENT_SLIP)
        _pr(rows[-1])

    # F. 조합: M*·C*(사전 선언) + 기존 S·DAA·PP(final_curves), 월말 리밸런싱
    F = pd.DataFrame(pd.read_parquet("research/final_curves.parquet"))
    start = F.index[0]
    slv = {k: pd.Series(F[k][start:]) / float(F[k][start:].iloc[0]) for k in ("S", "DAA", "PP")}
    for key, name in (("M*", m_star), ("C*", c_star), ("M*t", m_star + " +timing200")):
        s = pd.Series(curves[name][start:])
        slv[key] = s / float(s.iloc[0])
    costs = {"S": 0.012, "DAA": 0.001, "PP": 0.001, "M*": EVENT_SLIP + 0.002, "C*": EVENT_SLIP + 0.002, "M*t": EVENT_SLIP + 0.002}
    plans = {f"M*40+DAA30+PP30 [{m_star}]": {"M*": .4, "DAA": .3, "PP": .3},
             f"C*40+DAA30+PP30 [{c_star}]": {"C*": .4, "DAA": .3, "PP": .3},
             f"M*25+S25+DAA25+PP25 [{m_star}]": {"M*": .25, "S": .25, "DAA": .25, "PP": .25},
             f"M*20+C*20+S20+DAA20+PP20": {"M*": .2, "C*": .2, "S": .2, "DAA": .2, "PP": .2},
             f"M*t40+DAA30+PP30 [{m_star}+timing]": {"M*t": .4, "DAA": .3, "PP": .3}}
    for name, w in plans.items():
        c = combo.combine(slv, w, costs)
        _log("R7 " + name, metrics(c), "7차 조합")
        add(name, c, kind="combo", family="combo", market="ALL", tier="mix", style="combo", hold=21, slip=EVENT_SLIP)
        _pr(rows[-1])

    # G. DSR(전체 시도 수) · PBO(7차 슬리브+이벤트 CSCV) · 순위표
    T = pd.DataFrame(rows)
    n_tr, var_sr = validate.trial_stats()
    T["dsr"] = [validate.deflated_sharpe(curves[n].pct_change(), n_tr, var_sr)["dsr"] for n in T["name"]]
    T["pass"] = [accept(r, d) for r, d in zip(T.to_dict("records"), T["dsr"])]
    cand = pd.DataFrame({n: curves[n].pct_change() for n in T[T["kind"].isin(["sleeve", "event"])]["name"]})
    T["pbo_group"] = pbo_cscv(cand)
    T["n_trials"] = n_tr
    T = T.sort_values("sharpe", ascending=False).reset_index(drop=True)
    T.to_csv("research/round7_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(curves).to_parquet("research/round7_curves.parquet")
    pd.set_option("display.width", 250)
    print(f"\n시도 수 {n_tr}, PBO(7차 {len(cand.columns)}개) {T['pbo_group'].iloc[0]:.2f}")
    print(T[["name", "kind", "cagr", "mdd", "sharpe", "is_cagr", "oos_cagr", "cut2024_cagr", "cagr_slip2x", "capacity_ukwon",
             "top2yr_share", "dsr", "pass"]].head(40).to_string(index=False))
