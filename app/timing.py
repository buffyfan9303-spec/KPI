import pandas as pd


def hysteresis_position(close: pd.Series, ma: pd.Series, band: float = 0.01) -> pd.Series:
    """PDF 3단계 이중 임계값(Hysteresis) 필터.

    - 종가 > ma * (1 + band) → 1 (주식 보유)
    - 종가 < ma * (1 - band) → 0 (현금)
    - 그 사이 버퍼 존 → 직전 상태 유지 (휩소 차단)
    - ma가 NaN인 초기 구간과 시작 상태는 0 (현금)

    반환: close와 같은 index의 0/1 float Series.
    """
    # TODO(사용자): 여기를 구현해 주세요 (5~10줄).
    raise NotImplementedError
