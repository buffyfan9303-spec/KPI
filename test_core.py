import pandas as pd

from app import backtest
from app.timing import hysteresis_position


def test_hysteresis_holds_state_inside_band():
    ma = pd.Series([100.0] * 6)
    close = pd.Series([100.5, 101.5, 100.2, 99.5, 98.5, 99.8])
    #  버퍼안(초기 0) / 상향돌파 1 / 버퍼 유지 1 / 버퍼 유지 1 / 하향돌파 0 / 버퍼 유지 0
    assert hysteresis_position(close, ma, 0.01).tolist() == [0, 1, 1, 1, 0, 0]


def test_hysteresis_nan_ma_is_cash():
    ma = pd.Series([float("nan"), 100.0])
    assert hysteresis_position(pd.Series([200.0, 200.0]), ma, 0.01).tolist() == [0, 1]


def test_costs_on_round_trip():
    idx = pd.date_range("2026-01-01", periods=4)
    close = pd.Series([100.0, 100.0, 100.0, 100.0], idx)
    pos = pd.Series([1.0, 1.0, 0.0, 0.0], idx)
    eq = backtest.run(close, pos, slippage=0.01, sell_tax=0.002)
    # 매수 슬리피지 1% → 매도 슬리피지 1% + 거래세 0.2%
    assert abs(eq.iloc[-1] - 0.99 * 0.988) < 1e-9
