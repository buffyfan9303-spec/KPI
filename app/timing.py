import pandas as pd


def hysteresis_position(close: pd.Series, ma: pd.Series, band: float = 0.01) -> pd.Series:
    """PDF 3단계 이중 임계값(Hysteresis) 필터.

    - 종가 > ma * (1 + band) → 1 (주식 보유)
    - 종가 < ma * (1 - band) → 0 (현금)
    - 그 사이 버퍼 존 → 직전 상태 유지 (휩소 차단)
    - ma가 NaN인 초기 구간과 시작 상태는 0 (현금)

    반환: close와 같은 index의 0/1 float Series.
    """
    upper, lower = ma * (1 + band), ma * (1 - band)
    state, out = 0.0, []
    for c, up, lo in zip(close.to_numpy(), upper.to_numpy(), lower.to_numpy()):
        if up != up:  # NaN: 이동평균이 아직 없으면 현금
            state = 0.0
        elif c > up:
            state = 1.0
        elif c < lo:
            state = 0.0
        out.append(state)  # 버퍼 존에서는 직전 상태 유지
    return pd.Series(out, index=close.index, dtype=float)
