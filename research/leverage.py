"""레버리지 실험 (사전 선언, 2026-09-25): 위험 대비 수익이 가장 좋은 조합에 일정 배율을 건다.

- 매일 목표 배율 L로 맞춤(선물로 노출을 조절하는 방식과 비슷). 일간 수익 = L·r − (L−1)·차입금리/252 − 조정 비용.
- 차입금리 2가지: 3.5%(KOSPI200 선물 내재 금리 수준 가정) / 8.0%(증권사 신용융자 수준 가정).
- 평가: 전체 CAGR·MDD, 5년 이동 구간 CAGR 분포, 블록 부트스트랩(6개월 블록, 1만 회)으로 5년 CAGR ≥ 30% 확률.
주의: 소형주 슬리브는 선물로 레버리지할 수 없어 실제로는 신용융자(8%)가 현실적이다.
"""
import numpy as np
import pandas as pd

ADJ_COST = 0.0005  # 배율 유지용 조정 거래 비용(조정 비중당)


def lever(curve: pd.Series, L: float, borrow: float) -> pd.Series:
    r = curve.pct_change().fillna(0.0).to_numpy()
    eq, out = 1.0, []
    for x in r:
        gross = 1 + L * x - (L - 1) * borrow / 252
        # 하루 수익 뒤 실제 배율이 L에서 벗어난 만큼 다시 맞추는 비용
        drift = abs(L * (1 + x) / (1 + L * x) - L) if (1 + L * x) > 0 else L
        eq = max(eq * gross * (1 - drift * ADJ_COST), 0.0)  # 0 밑으로는 못 감(파산)
        out.append(eq)
    return pd.Series(out, index=curve.index)


def rolling_5y_cagr(curve: pd.Series) -> pd.Series:
    m = curve.resample("ME").last()
    return (m / m.shift(60)) ** (1 / 5) - 1


def bootstrap_5y(curve: pd.Series, n: int = 10000, block: int = 6, seed: int = 0) -> dict:
    m = curve.resample("ME").last().pct_change().dropna().to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(m) - block + 1, size=(n, 60 // block))
    paths = np.stack([m[s:s + block] for s in starts.ravel()]).reshape(n, 60)
    growth = np.prod(1 + paths, axis=1)
    cagr = growth ** (1 / 5) - 1
    dd = np.min(np.cumprod(1 + paths, axis=1) / np.maximum.accumulate(np.cumprod(1 + paths, axis=1), axis=1) - 1, axis=1)
    return {"p_cagr30": float((cagr >= 0.30).mean()), "p_loss": float((cagr < 0).mean()),
            "median_cagr": float(np.median(cagr)), "p5_cagr": float(np.percentile(cagr, 5)),
            "median_mdd": float(np.median(dd)), "p5_mdd": float(np.percentile(dd, 5))}
