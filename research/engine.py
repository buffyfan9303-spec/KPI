"""포트폴리오 백테스트 엔진 (1단계 A 소형 가치주 + 3단계 타이밍).

규칙(미래 참조 방지)
- 재무는 사용 가능일(avail) 이후에만 쓴다. avail = min(DART 접수일, 이듬해 4월 15일)
  → 정정공시로 접수일이 늦어진 경우 원공시가 기한 내 제출됐다고 본다(정정 수치 사용은 약한 미래 참조, 기록해 둠).
- 리밸런싱일 종가 기준으로 고르고 다음 거래일 수익률부터 반영한다.
- 수익률은 KRX 등락률(배당 제외 → 보수적). 거래정지일 수익률은 0, 상장폐지 후엔 마지막 값이 현금으로 남는다.
- 비용: 매수·매도 슬리피지 + 매도 거래세(연도별 실제 세율).
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from research import data

TRIALS = Path(__file__).with_name("trials.csv")
EXCLUDE = r"스팩|리츠|선박|\d+호$"  # 스팩·리츠·선박투자회사(예: 코리아01호)는 영업회사가 아님

# 매도 증권거래세(농특세 포함, 코스피·코스닥 동일 취급). 출처: 국세청·언론 정리(2019-06 인하, 2021·2023·2024·2025 단계 인하, 2026 환원)
TAX = [("2015-01-01", 0.0030), ("2019-06-03", 0.0025), ("2021-01-01", 0.0023), ("2023-01-01", 0.0020),
       ("2024-01-01", 0.0018), ("2025-01-01", 0.0015), ("2026-01-01", 0.0020)]


def sell_tax(d: pd.Timestamp) -> float:
    rate = TAX[0][1]
    for start, r in TAX:
        if d >= pd.Timestamp(start):
            rate = r
    return rate


@dataclass
class Config:
    name: str = "1A"
    start: str = "2016-04-15"
    end: str = "2026-09-24"
    rebalance: str = "annual"       # annual(4월) / semi(4·10월) / quarterly(1·4·7·10월) / monthly
    small_q: float = 0.20           # 시가총액 하위 비율
    top_n: int = 20
    value: str = "per_pbr"          # per_pbr(PDF) / per_pbr_psr(가치 3지표)
    quality_q: float = 0.0          # 0이면 끔. 예: 0.5 → 자산 대비 영업이익(OP/A) 상위 50%만
    min_amount: float = 1e8         # 20일 평균 거래대금 하한(원). 너무 안 거래되는 종목 제외
    slippage: float = 0.01          # 편도
    strategy: str = "A"             # A(소형 가치) / B(대형 모멘텀, EPS 상향의 대용) / AB(반반) / factor(research.factors)
    factor: str = ""                # strategy="factor"일 때 research.factors.FACTORS 이름
    universe: str = "all"           # factor: all(유동성 통과 전체) / small(시총 하위 small_q)
    b_top: int = 200                # B: 시가총액 상위 몇 종목에서 고를지
    mom_lb: int = 252               # B: 모멘텀 계산 기간(거래일)
    mom_skip: int = 21              # B: 최근 1개월 제외(단기 반전 효과 회피)
    stop_loss: float = 0.0          # 0이면 끔. 예: 0.10 → 매수 후 고점 대비 -10%면 다음 리밸런싱까지 현금(PDF 13쪽 손절)
    timing: str = "none"            # none / KQ11 / KS11  (3단계 Hysteresis 현금 전환)
    ma: int = 60
    band: float = 0.01
    fscore_min: int = 0             # 0이면 끔. 라운드6 T8: 5점 F-score-lite 최소 점수(rank_value에서 적용)
    accel_filter: bool = False      # 라운드6 T9: 영업이익 가속((ΔOP_FY-ΔOP_FY-1)/자산>0), 2018-04부터 적용
    hold_buffer: int = 0            # 0이면 끔. 라운드6 T1: 기존 보유는 순위 ≤ hold_buffer*top_n이면 유지
    trend_hold_ma: int = 0          # 0이면 끔. 라운드6 T2: 버퍼 밖이어도 종가>이 기간 SMA면 최대 trend_hold_max_q분기 더 유지
    trend_hold_max_q: int = 4
    exit: str = "none"              # none / chandelier_losers(T3) / chandelier_regime(T4)
    entry_basis: bool = False       # True: 계속 보유하는 종목은 '처음 산 가격' 기준 손익 유지(검증 지적 버그 수정, 2026-09-25)
    stop_grace: int = 0             # 매수 후 이 거래일 수 동안은 샹들리에 손절 유예
    note: str = ""
    extra: dict = field(default_factory=dict)


def _rebalance_dates(idx: pd.DatetimeIndex, cfg: Config) -> list[pd.Timestamp]:
    months = {"annual": [4], "semi": [4, 10], "quarterly": [1, 4, 7, 10], "monthly": list(range(1, 13))}[cfg.rebalance]
    out = []
    years = sorted({t.year for t in idx})
    for y in range(years[0], years[-1] + 1):
        for m in months:
            later = idx[idx >= pd.Timestamp(y, m, 15)]
            if len(later) and pd.Timestamp(cfg.start) <= later[0] <= pd.Timestamp(cfg.end):
                out.append(later[0])
    return sorted(set(out))


def _accounts_at(d: pd.Timestamp, lag: int = 0) -> pd.DataFrame:
    """d 시점에 공개돼 있던 가장 최근 사업보고서(종목별)."""
    frames = []
    for fy in (d.year - 1 - lag, d.year - 2 - lag):
        if fy < 2015:
            continue
        a = data.annual_accounts(fy).copy()
        avail = a["filed"].where(a["filed"] < pd.Timestamp(fy + 1, 4, 15), pd.Timestamp(fy + 1, 4, 15))
        frames.append(a[avail <= d].assign(fy=fy))
    if not frames:  # 2015년 이전 재무는 DART API에 없음
        return pd.DataFrame({c: pd.Series(dtype=float) for c in ("net", "equity", "sales", "op", "assets", "fy")})
    acc = pd.concat(frames)
    return acc.sort_values("fy", ascending=False).groupby(level=0).head(1)


def rank_value(d: pd.Timestamp, snap: pd.DataFrame, amt20: pd.Series, cfg: Config) -> pd.Series:
    """소형 가치 유니버스 전체에 대한 점수(낮을수록 좋음). select()의 순위를 라운드6 T1 hold_buffer가 쓸 수 있게 분리."""
    u = pd.DataFrame(snap[snap.index.str.endswith("0") & ~snap["Name"].str.contains(EXCLUDE, na=False, regex=True)
                          & (snap["Volume"] > 0) & (amt20.reindex(snap.index).fillna(0) >= cfg.min_amount)])
    small = pd.DataFrame(u[u["Marcap"] <= u["Marcap"].quantile(cfg.small_q)])
    df = small.join(_accounts_at(d)[["net", "equity", "sales", "op", "assets"]], how="inner")
    df = df.assign(per=df["Marcap"] / df["net"], pbr=df["Marcap"] / df["equity"], psr=df["Marcap"] / df["sales"],
                   opa=df["op"] / df["assets"])
    df = pd.DataFrame(df[(df["per"] > 0) & (df["pbr"] > 0)])
    if cfg.quality_q > 0:  # 수익성 하위 종목(가치 함정) 제거
        df = pd.DataFrame(df[df["opa"] >= df["opa"].quantile(1 - cfg.quality_q)])
    if cfg.fscore_min > 0:  # 라운드6 T8: 5점 F-score-lite
        from research import fscore
        df = pd.DataFrame(df[fscore.fscore(d, df.index).reindex(df.index).fillna(0) >= cfg.fscore_min])
    if cfg.accel_filter:  # 라운드6 T9: 영업이익 가속
        from research import fscore
        df = pd.DataFrame(df[fscore.accel_ok(d, df.index).reindex(df.index).fillna(False)])
    score = pd.Series(df["per"]).rank() + pd.Series(df["pbr"]).rank()
    if cfg.value == "per_pbr_psr":
        psr = pd.Series(df["psr"])
        score = score + psr.where(psr > 0).rank()
    return score


def select(d: pd.Timestamp, snap: pd.DataFrame, amt20: pd.Series, cfg: Config) -> list[str]:
    return [str(c) for c in rank_value(d, snap, amt20, cfg).nsmallest(cfg.top_n).index]


def _hold_buffer_target(d: pd.Timestamp, snap: pd.DataFrame, amt20: pd.Series, cfg: Config, current: pd.Series,
                        extra_q: dict[str, int], P: pd.DataFrame, i: int) -> tuple[pd.Series, dict[str, int]]:
    """라운드6 T1(+T2). 기존 보유는 순위 ≤ hold_buffer*top_n이면 드리프트된 비중 그대로 유지,
    빈자리만 상위 순위에서 채운다(새 편입분은 남는 비중을 균등 배분). T2: 추세(SMA) 예외로 최대 N분기 더 유지."""
    score = rank_value(d, snap, amt20, cfg)
    if len(score) == 0:
        return pd.Series(dtype=float), {}
    ranks = score.rank(method="first")
    buffer_n = cfg.hold_buffer * cfg.top_n
    kept: dict[str, float] = {}
    new_extra_q: dict[str, int] = {}
    for c, wgt in current.items():
        c = str(c)
        if c not in ranks.index:
            continue  # 유니버스 이탈(거래정지 등) → 자연 매도
        if ranks[c] <= buffer_n:
            kept[c] = float(wgt)
        elif cfg.trend_hold_ma > 0 and c in P.columns:
            lo = max(0, i - cfg.trend_hold_ma + 1)
            sma = float(P[c].iloc[lo: i + 1].mean())
            q = extra_q.get(c, cfg.trend_hold_max_q)
            if q > 0 and pd.notna(P[c].iloc[i]) and pd.notna(sma) and float(P[c].iloc[i]) > sma:
                kept[c] = float(wgt)
                new_extra_q[c] = q - 1
    n_new = cfg.top_n - len(kept)
    if n_new > 0:
        cands = [str(c) for c in ranks.sort_values().index if str(c) not in kept][:n_new]
        freed = max(0.0, 1.0 - sum(kept.values()))
        for c in cands:
            kept[c] = freed / len(cands) if cands else 0.0
    return pd.Series(kept), new_extra_q


def select_b(d: pd.Timestamp, snap: pd.DataFrame, amt20: pd.Series, P: pd.DataFrame, cfg: Config) -> list[str]:
    """1단계 B 대용: 대형주 중 12-1개월 가격 모멘텀 상위. (EPS 추정치 상향 과거 데이터가 무료로 없어서)"""
    u = pd.DataFrame(snap[snap.index.str.endswith("0") & ~snap["Name"].str.contains(EXCLUDE, na=False, regex=True)
                          & (snap["Volume"] > 0)])
    big = u.nlargest(cfg.b_top, columns="Marcap")
    i = int(P.index.get_indexer([d])[0])
    if i < cfg.mom_lb:
        return []
    mom = (P.iloc[i - cfg.mom_skip] / P.iloc[i - cfg.mom_lb] - 1).reindex(big.index).dropna()
    return mom.nlargest(cfg.top_n).index.tolist()


def select_factor(d, snap, amt20, P, R, cfg: Config) -> list[str]:
    from research import factors
    u = pd.DataFrame(snap[snap.index.str.endswith("0") & ~snap["Name"].str.contains(EXCLUDE, na=False, regex=True)
                          & (snap["Volume"] > 0) & (amt20.reindex(snap.index).fillna(0) >= cfg.min_amount)])
    if cfg.universe == "small":
        u = pd.DataFrame(u[u["Marcap"] <= u["Marcap"].quantile(cfg.small_q)])
    acc = _accounts_at(d)
    prev = _accounts_at(d, lag=1)
    # 직전 연도 재무: 같은 종목의 한 해 전 사업보고서
    prev = pd.DataFrame(prev[prev["fy"].reindex(prev.index) < acc["fy"].reindex(prev.index).fillna(9999)])
    f = factors.features(d, u, P, R, acc, prev)
    spec, _ = factors.FACTORS[cfg.factor]
    return factors.pick(f, spec, cfg.top_n)


def _weights(d, snap, amt20, P, cfg: Config, R=None) -> pd.Series:
    parts = []
    if cfg.strategy == "factor":
        parts.append(select_factor(d, snap, amt20, P, R, cfg))
    if cfg.strategy == "picks":  # 외부(예: 머신러닝)에서 미리 고른 종목
        parts.append(cfg.extra["picks"].get(str(d.date()), []))
    if cfg.strategy in ("A", "AB"):
        parts.append(select(d, snap, amt20, cfg))
    if cfg.strategy in ("B", "AB"):
        parts.append(select_b(d, snap, amt20, P, cfg))
    w = pd.Series(dtype=float)
    for names in parts:
        if names:
            w = w.add(pd.Series(1 / (len(parts) * len(names)), index=names), fill_value=0)
    return w / w.sum() if len(w) else w


def _timing(cfg: Config, idx: pd.DatetimeIndex) -> pd.Series:
    if cfg.timing == "none":
        return pd.Series(1.0, index=idx)
    import FinanceDataReader as fdr
    from app.timing import hysteresis_position
    close = pd.Series(fdr.DataReader(cfg.timing, "2014-01-01")["Close"], dtype=float)
    pos = hysteresis_position(close, pd.Series(close.rolling(cfg.ma).mean()), cfg.band)
    return pos.reindex(idx).ffill().fillna(0.0)


PARK_ETF = "069500"     # KODEX 200: T3 손실 종목·T4 강세장 청산분 대피처
PARK_SAFE = "153130"    # KODEX 단기채권: T4 약세장 전체 청산분 대피처


def chandelier_stops(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일별 (종가, 22일 최고고가-3xATR22) 패널. High/Low는 marcap 원가(액면분할 조정 미반영, 22일 창이라 영향 미미 — 문서화된 단순화)."""
    high = px.pivot_table(index="Date", columns="Code", values="High", aggfunc="last")
    low = px.pivot_table(index="Date", columns="Code", values="Low", aggfunc="last")
    close = px.pivot_table(index="Date", columns="Code", values="Close", aggfunc="last")
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()]).groupby(level=0).max()
    atr22 = tr.rolling(22, min_periods=11).mean()
    hh22 = high.rolling(22, min_periods=11).max()
    return close, hh22 - 3 * atr22


def _chandelier_hits(w: pd.Series, val: pd.Series, d: pd.Timestamp, chand_close: pd.DataFrame,
                     chand_stop: pd.DataFrame, bear: bool) -> list[str]:
    """T3(bear=False): 손실 중인(val<1) 보유만 대상. T4 약세장(bear=True): 전 보유 대상."""
    cand = [str(c) for c in w.index] if bear else [str(c) for c in w.index if float(val[c] if c in val.index else 1.0) < 1.0]
    return [c for c in cand if c in chand_stop.columns and pd.notna(chand_stop.at[d, c])
            and c in chand_close.columns and float(chand_close.at[d, c]) < float(chand_stop.at[d, c])]


def regime_signal(idx: pd.DatetimeIndex, etf: str = PARK_ETF, ma: int = 200) -> pd.Series:
    """월말(마지막 거래일) 판단, 다음 날부터 적용: ETF 종가 > 200일선이면 강세(True)."""
    import FinanceDataReader as fdr
    close = pd.Series(fdr.DataReader(etf, "2010-01-01")["Close"], dtype=float)
    bull = close > close.rolling(ma).mean()
    ci = pd.DatetimeIndex(idx)
    month_end = ci.to_series().groupby(ci.to_period("M")).transform("max") == ci
    sig = pd.Series(pd.NA, index=ci, dtype="boolean")
    sig[month_end] = bull.reindex(ci)[month_end]
    return pd.Series(sig.ffill().fillna(True).shift(1).fillna(True), dtype=bool)


def run(cfg: Config, log: bool = True) -> dict:
    px = data.prices()
    R = data.returns_wide(px)
    P = (1 + R.fillna(0.0)).cumprod().where(R.notna().cummax())  # 수정 가격 지수, 상장 전은 NaN
    R_full = R
    R = pd.DataFrame(R[(R.index >= pd.Timestamp(cfg.start)) & (R.index <= pd.Timestamp(cfg.end))]).copy()
    amt = px.pivot_table(index="Date", columns="Code", values="Amount", aggfunc="last")
    amt20 = amt.rolling(20, min_periods=10).mean()
    idx = pd.DatetimeIndex(R.index)
    rebal = set(_rebalance_dates(idx, cfg))
    last = px.groupby("Code").tail(1).set_index("Code")
    frozen_exit = pd.Series(last[(last["Volume"] == 0) & (last["Date"] < px["Date"].max())]["Date"])  # 정지 상태로 사라진 종목
    expo = _timing(cfg, idx).shift(1).fillna(0.0)  # 오늘 신호 → 내일 반영

    px_by_date = {k: g.set_index("Code") for k, g in px[px["Date"] >= pd.Timestamp(cfg.start)].groupby("Date")}
    w = pd.Series(dtype=float)       # 주식 비중(합 ≤ 1), 나머지는 현금
    target = pd.Series(dtype=float)
    val, peak = pd.Series(dtype=float), pd.Series(dtype=float)  # 손절/샹들리에용: 매수 후 누적값과 그 고점
    stopped: set[str] = set()
    eq, curve, turn_total, picks = 1.0, [], 0.0, {}
    prev_e = 0.0
    track_val = cfg.stop_loss > 0 or cfg.exit in ("chandelier_losers", "chandelier_regime")
    age = pd.Series(dtype=float)  # 매수 후 경과 거래일(stop_grace용)
    extra_q: dict[str, int] = {}      # 라운드6 T2: 추세 예외로 더 들고 있는 분기 수 카운트
    chand_close, chand_stop = pd.DataFrame(), pd.DataFrame()
    bull: pd.Series = pd.Series(dtype=bool)
    if cfg.exit in ("chandelier_losers", "chandelier_regime"):
        chand_close, chand_stop = chandelier_stops(px)
        for etf_code in (PARK_ETF, PARK_SAFE):  # 대피 ETF도 보유 종목처럼 R에 합쳐 같은 매커니즘으로 수익 반영
            R[etf_code] = etf_returns(etf_code, idx)
        bull = regime_signal(idx) if cfg.exit == "chandelier_regime" else pd.Series(True, index=idx)
    for d in idx:
        r = R.loc[d]
        # 1) 오늘 수익 반영(정지·폐지 NaN → 0)
        if len(w):
            rr = 1 + r.reindex(w.index).fillna(0.0)
            gross = (w * rr).sum() + (1 - w.sum())
            w = w * rr / gross
            eq *= gross
            gone = [c for c in w.index if c in frozen_exit.index and d > frozen_exit[c]]
            if gone:  # 거래정지 상태로 상장폐지 → 회수 불가로 0원 처리(보수적)
                eq *= 1 - float(w[gone].sum())
                w = w.drop(gone)
            if track_val:
                val = (val.reindex(w.index).fillna(1.0)) * rr
                peak = pd.concat([peak.reindex(w.index).fillna(1.0), val], axis=1).max(axis=1)
                age = age.reindex(w.index).fillna(0.0) + 1
            if cfg.stop_loss > 0:
                hit = [c for c in w.index if val[c] / peak[c] - 1 <= -cfg.stop_loss]
                if hit:
                    sold = float(w[hit].sum())
                    eq *= 1 - sold * (cfg.slippage + sell_tax(d))
                    turn_total += sold
                    w = w.drop(hit)
                    stopped.update(hit)
            if cfg.exit in ("chandelier_losers", "chandelier_regime") and d in chand_close.index:
                bear = not bool(bull.loc[d])
                hit = _chandelier_hits(w, val, d, chand_close, chand_stop, bear)
                if cfg.stop_grace > 0:
                    hit = [c for c in hit if float(age.get(c, 0.0)) >= cfg.stop_grace]
                if hit:
                    sold = float(w[hit].sum())
                    eq *= 1 - sold * (cfg.slippage + sell_tax(d))
                    turn_total += sold
                    w = w.drop(hit)
                    stopped.update(hit)
                    dest = PARK_SAFE if bear else PARK_ETF
                    eq *= 1 - sold * 0.0005  # ETF 매수 슬리피지(세금 없음, T3/T4)
                    w = w.add(pd.Series({dest: sold}), fill_value=0.0)
        # 2) 종가에 리밸런싱·타이밍 반영(비용은 오늘 차감, 수익은 내일부터)
        if d in rebal:
            snap = px_by_date[d]
            if cfg.hold_buffer > 0 and cfg.strategy == "A":
                i = int(P.index.get_indexer([d])[0])
                target, extra_q = _hold_buffer_target(d, snap, pd.DataFrame(amt20).loc[d], cfg, w, extra_q, P, i)
            else:
                target = _weights(d, snap, pd.DataFrame(amt20).loc[d], P, cfg, R_full)
            stopped = set()
            if cfg.entry_basis:
                held_now = [c for c in target.index if c in w.index]
                val = pd.Series(1.0, index=target.index).where(~target.index.isin(held_now), val.reindex(target.index))
                peak = pd.Series(1.0, index=target.index).where(~target.index.isin(held_now), peak.reindex(target.index))
                age = pd.Series(0.0, index=target.index).where(~target.index.isin(held_now), age.reindex(target.index))
                val, peak, age = val.fillna(1.0), peak.fillna(1.0), age.fillna(0.0)
            else:
                val, peak = pd.Series(1.0, index=target.index), pd.Series(1.0, index=target.index)
            picks[str(d.date())] = target.index.tolist()
        e = float(expo.loc[d])
        if len(target) and (d in rebal or e != prev_e):
            keep = target.drop([c for c in stopped if c in target.index])
            tgt = keep * e if e > 0 else pd.Series(dtype=float)
            if len(w):  # 오늘 거래가 없는(정지) 보유 종목은 팔 수 없으니 그대로 두고 나머지를 줄인다
                today = px_by_date.get(d)
                stuck = [c for c in w.index if c not in (PARK_ETF, PARK_SAFE)
                        and (today is None or c not in today.index or today.at[c, "Volume"] == 0)]
                if stuck:
                    held = pd.Series(w[stuck])
                    rest = pd.Series(tgt.drop([c for c in stuck if c in tgt.index]))
                    room = max(0.0, e - float(held.sum()))
                    tgt = pd.concat([held, rest / rest.sum() * room if rest.sum() > 0 else rest])
            allc = w.index.union(tgt.index)
            diff = tgt.reindex(allc).fillna(0) - w.reindex(allc).fillna(0)
            buy, sell = diff.clip(lower=0).sum(), (-diff).clip(lower=0).sum()
            cost = (buy + sell) * cfg.slippage + sell * sell_tax(d)
            eq *= 1 - cost
            turn_total += buy + sell
            w = tgt
        prev_e = e
        curve.append(eq)
    s = pd.Series(curve, index=idx)
    m = metrics(s)
    m.update(turnover_per_year=round(turn_total / max((idx[-1] - idx[0]).days / 365.25, 1e-9), 2))
    if log:
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": json.dumps(asdict(cfg), ensure_ascii=False), **m}
        pd.DataFrame([row]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False, encoding="utf-8")
    return {"metrics": m, "curve": s, "picks": picks}


def metrics(s: pd.Series) -> dict:
    d = s.pct_change().dropna()
    ix = pd.DatetimeIndex(s.index)
    years = (ix[-1] - ix[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    mdd = float((s / s.cummax() - 1).min())
    sharpe = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
    yearly = s.resample("YE").last().pct_change()
    yearly.iloc[0] = s.resample("YE").last().iloc[0] / s.iloc[0] - 1
    return {"cagr": round(float(cagr), 4), "mdd": round(mdd, 4), "sharpe": round(sharpe, 3),
            "vol": round(float(d.std() * np.sqrt(252)), 4), "days": len(d),
            "yearly": {str(k.year): round(float(v), 3) for k, v in yearly.items()}}


def rotate(curve: pd.Series, name: str, lookback: int = 252, bench: str = "069500",
           switch_cost: float = 0.025, log: bool = True) -> dict:
    """팩터 모멘텀 전환: 매월 말, 전략의 최근 lookback일 수익이 KOSPI200 ETF보다 낮으면 ETF로 갈아탄다.
    근거: Ehsani & Linnainmaa(2022, JF) 팩터 수익의 자기상관. 전환 비용은 편도 합산(switch_cost)."""
    import FinanceDataReader as fdr
    rs = curve.pct_change().fillna(0.0)
    etf = pd.Series(fdr.DataReader(bench, "2014-01-01")["Close"], dtype=float)  # ETF는 marcap에 없어 따로 받음(분배금 제외)
    rb = etf.pct_change().reindex(curve.index).fillna(0.0)
    bcurve = (1 + rb).cumprod()
    mom_s, mom_b = curve / curve.shift(lookback) - 1, bcurve / bcurve.shift(lookback) - 1
    ci = pd.DatetimeIndex(curve.index)
    month_end = ci.to_series().groupby(ci.to_period("M")).transform("max") == ci
    sig = pd.Series(np.nan, index=curve.index)
    sig[month_end] = (mom_s >= mom_b)[month_end].astype(float)
    sig = sig.ffill().fillna(1.0).shift(1).fillna(1.0)  # 월말 판단 → 다음 날부터
    r = np.where(sig == 1, rs, rb)
    r = r - sig.diff().abs().fillna(0.0).to_numpy() * switch_cost
    s = pd.Series((1 + r).cumprod(), index=curve.index)
    m = metrics(s)
    m.update(switches=int(sig.diff().abs().sum()), in_strategy=round(float(sig.mean()), 2))
    if log:
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "config": json.dumps({"name": name, "rotate_lookback": lookback, "bench": bench}, ensure_ascii=False), **m}
        pd.DataFrame([row]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False, encoding="utf-8")
    return {"metrics": m, "curve": s}


def vol_target(curve: pd.Series, name: str, target: float = 0.20, window: int = 60, cost: float = 0.01,
               log: bool = True) -> dict:
    """변동성 목표화(Moreira & Muir 2017): 최근 window일 변동성이 높으면 주식 비중을 줄인다.
    비중 = min(1, target / 실현변동성), 매주 금요일에만 조정(잦은 매매 방지), 비중 변화분에 비용."""
    r = curve.pct_change().fillna(0.0)
    vol = r.rolling(window).std() * np.sqrt(252)
    raw = (target / vol).clip(upper=1.0).fillna(1.0)
    fri = [pd.Timestamp(str(t)).weekday() == 4 for t in curve.index]
    w = raw.where(fri).ffill().fillna(1.0).shift(1).fillna(1.0)
    out = w * r - w.diff().abs().fillna(0.0) * cost
    s = (1 + out).cumprod()
    m = metrics(s)
    m.update(avg_exposure=round(float(w.mean()), 2))
    if log:
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "config": json.dumps({"name": name, "vol_target": target, "window": window}, ensure_ascii=False), **m}
        pd.DataFrame([row]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False, encoding="utf-8")
    return {"metrics": m, "curve": s}


def etf_returns(code: str, idx: pd.DatetimeIndex) -> pd.Series:
    import FinanceDataReader as fdr
    c = pd.Series(fdr.DataReader(code, "2014-01-01")["Close"], dtype=float)
    return c.pct_change().reindex(idx).fillna(0.0)


def dual_momentum(assets: dict[str, pd.Series], costs: dict[str, float], safe: str, name: str,
                  lookback: int = 252, log: bool = True) -> dict:
    """듀얼 모멘텀(Antonacci): 매월 말 최근 lookback일 수익이 가장 높은 자산을 고르되(상대 모멘텀),
    그 수익이 안전자산보다 낮으면 안전자산으로(절대 모멘텀). assets는 같은 날짜의 일간 수익률.
    전환 비용 = 파는 자산 비용 + 사는 자산 비용."""
    df = pd.DataFrame(assets).fillna(0.0)
    ci = pd.DatetimeIndex(df.index)
    growth = (1 + df).cumprod()
    mom = growth / growth.shift(lookback) - 1
    month_end = ci.to_series().groupby(ci.to_period("M")).transform("max") == ci
    pick = pd.Series(np.nan, index=ci, dtype=object)
    for d in ci[pd.Series(month_end).to_numpy(dtype=bool)]:
        m = mom.loc[d].dropna()
        if len(m) == 0:
            continue
        best = str(m.idxmax())
        pick[d] = best if m[best] > m.get(safe, -np.inf) else safe
    pick = pick.ffill().shift(1).fillna(safe)  # 판단 다음 날부터
    r = pd.Series([df.at[d, a] for d, a in zip(ci, pick)], index=ci)
    sw = pick != pick.shift(1)
    sw.iloc[0] = False
    cost = pd.Series([costs[a] + costs[b] if s_ else 0.0 for s_, a, b in zip(sw, pick.shift(1).fillna(safe), pick)], index=ci)
    s = (1 + r - cost).cumprod()
    m = metrics(s)
    m.update(switches=int(sw.sum()), share={k: round(float((pick == k).mean()), 2) for k in assets})
    if log:
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "config": json.dumps({"name": name, "dual_lookback": lookback, "assets": list(assets)}, ensure_ascii=False), **m}
        pd.DataFrame([row]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False, encoding="utf-8")
    return {"metrics": m, "curve": s, "pick": pick}
