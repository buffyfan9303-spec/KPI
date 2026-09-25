"""과최적화 검증: Deflated Sharpe Ratio (Bailey & López de Prado, 2014, SSRN 2460551).

여러 설정을 시도하면 가장 좋은 샤프는 운으로도 높게 나온다. 시도 횟수 N과 시도들 간 샤프 분산으로
'운으로 기대되는 최대 샤프'(SR*)를 구하고, 실제 샤프가 그보다 유의하게 큰지 확률로 본다.
DSR ≥ 0.95 이면 통과(5% 유의수준).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from research.engine import TRIALS

EULER = 0.5772156649


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """시도 n번, 샤프 분산 var_sr일 때 운만으로 기대되는 최대 샤프(일간 단위)."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, var_sr: float) -> dict:
    r = returns.dropna()
    sr = float(r.mean() / r.std())                   # 일간 샤프
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_sr)
    z = (sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return {"sharpe_annual": round(sr * np.sqrt(252), 3), "sr0_annual": round(sr0 * np.sqrt(252), 3),
            "dsr": round(float(stats.norm.cdf(z)), 4), "n_trials": n_trials, "skew": round(sk, 2), "kurt": round(ku, 2)}


def trial_stats() -> tuple[int, float]:
    """trials.csv에서 진단용을 뺀 실제 후보 시도 수와 (일간) 샤프 분산."""
    import csv
    with open(TRIALS, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]  # 시도 종류마다 열 수가 달라 csv로 읽고 샤프(5번째 열)만 쓴다
    sharpes = [float(r[4]) for r in rows if "진단" not in json.loads(r[1]).get("note", "")]
    daily = np.array(sharpes) / np.sqrt(252)
    return len(sharpes), float(daily.var(ddof=1))


if __name__ == "__main__":
    # 자체 점검: 순수 잡음 전략 100개 중 최고는 DSR을 통과하면 안 된다
    rng = np.random.default_rng(0)
    rs = [pd.Series(rng.normal(0, 0.01, 2500)) for _ in range(100)]
    srs = [float(r.mean() / r.std()) for r in rs]
    best = rs[int(np.argmax(srs))]
    out = deflated_sharpe(best, 100, float(np.var(srs, ddof=1)))
    assert out["dsr"] < 0.95, out
    print("잡음 100개 중 최고 전략 DSR:", out["dsr"], "→ 통과 못함(정상)")
