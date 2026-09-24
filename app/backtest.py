import pandas as pd

SELL_TAX = 0.0020  # 코스피/코스닥 매도 증권거래세 (PDF 10쪽)
# ponytail: 고정 슬리피지. 소형주·대규모 자금이면 ADV 비율 기반 모델로 교체 (PDF: 0.5~2.5%)
SLIPPAGE = 0.005


def run(close: pd.Series, position: pd.Series,
        slippage: float = SLIPPAGE, sell_tax: float = SELL_TAX) -> pd.Series:
    """position(0~1)으로 운용한 자산곡선. 오늘 신호는 다음 날부터 적용(미래 참조 방지)."""
    ret = close.pct_change().fillna(0)
    pos = position.shift(1).fillna(0)
    dpos = pos.diff().fillna(pos)
    cost = dpos.abs() * slippage + (-dpos).clip(lower=0) * sell_tax
    return (1 + pos * ret - cost).cumprod()


def metrics(equity: pd.Series) -> dict:
    idx = pd.DatetimeIndex(equity.index)
    years = (idx[-1] - idx[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    mdd = (equity / equity.cummax() - 1).min()
    return {"cagr": round(float(cagr), 4), "mdd": round(float(mdd), 4),
            "total": round(float(equity.iloc[-1] - 1), 4)}
