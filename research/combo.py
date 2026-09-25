"""전략 묶음(슬리브) 조합 + 포트폴리오 낙폭 제어(CPPI).

슬리브 곡선들을 월말마다 정해진 비중으로 다시 맞춘다. 슬리브마다 편도 비용이 다르다(소형주 슬리브 ≈1.2%, ETF ≈0.1%).
CPPI(Grossman-Zhou): 위험자산 비중 = min(1, m × (자산 − 바닥)/자산), 바닥 = 고점 × floor. 나머지는 단기채.
주의: 바닥은 목표일 뿐 보장이 아니다(주 1회 조정, 갭 하락 시 뚫릴 수 있음 — risk-manager 검토 2026-09-25).
"""
import pandas as pd

DEFAULT_COST = 0.002


def _week_last(ix: pd.DatetimeIndex) -> set:
    """각 주의 마지막 거래일(금요일 휴장 주도 빠지지 않게)."""
    s = pd.Series(ix, index=ix)
    return set(s.groupby(ix.to_period("W")).max().to_numpy())


def combine(curves: dict[str, pd.Series], weights: dict[str, float],
            costs: dict[str, float] | None = None) -> pd.Series:
    costs = costs or {}
    df = pd.DataFrame({k: curves[k] for k in weights}).dropna()
    r = df.pct_change().fillna(0.0)
    ix = pd.DatetimeIndex(df.index)
    month_end = set(pd.Series(ix, index=ix).groupby(ix.to_period("M")).max().to_numpy())
    tgt = pd.Series(weights, dtype=float)
    c = pd.Series({k: costs.get(k, DEFAULT_COST) for k in weights}, dtype=float)
    w, eq, out = tgt.copy(), 1.0, []
    for d in ix:
        g = float((w * (1 + r.loc[d])).sum())
        w, eq = w * (1 + r.loc[d]) / g, eq * g
        if d in month_end:
            eq *= 1 - float(((w - tgt).abs() * c).sum())
            w = tgt.copy()
        out.append(eq)
    return pd.Series(out, index=ix)


def blended_cost(weights: dict[str, float], costs: dict[str, float]) -> float:
    """위험 묶음 전체를 사고팔 때의 가중 평균 편도 비용."""
    tot = sum(weights.values())
    return sum(w * costs.get(k, DEFAULT_COST) for k, w in weights.items()) / tot


def cppi(curve: pd.Series, safe: pd.Series, floor: float = 0.8, m: float = 4.0, cost: float = DEFAULT_COST) -> pd.Series:
    """매주 마지막 거래일에 위험 비중을 다시 계산. 오늘 판단 → 다음 날 반영."""
    r = curve.pct_change().fillna(0.0)
    rs = safe.reindex(curve.index).ffill().pct_change().fillna(0.0)
    week_last = _week_last(pd.DatetimeIndex(curve.index))
    eq, peak, w, out = 1.0, 1.0, 1.0, []
    for d in curve.index:
        eq *= 1 + w * r.loc[d] + (1 - w) * rs.loc[d]
        peak = max(peak, eq)
        if d in week_last:
            new_w = min(1.0, max(0.0, m * (eq - floor * peak) / eq))
            eq *= 1 - abs(new_w - w) * cost
            w = new_w
        out.append(eq)
    return pd.Series(out, index=curve.index)
