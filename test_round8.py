"""8차 신호의 미래 참조 방지·규칙 검증. 외부 데이터 없이 합성 시계열로만 돈다."""
import numpy as np
import pandas as pd

from research import round8 as r8


def _px(seed=0, n=600):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n)), index=idx)


def _mutate_after(s: pd.Series, d) -> pd.Series:
    s = s.copy()
    s.loc[s.index > d] = s.loc[s.index > d].to_numpy()[::-1] * 3.0 + 7.0
    return s


def test_trend_signal_uses_only_past_and_lags_one_day():
    px = _px()
    d = px.index[400]
    a, b = r8.trend_signal(px), r8.trend_signal(_mutate_after(px, d))
    pd.testing.assert_series_equal(a.loc[:d], b.loc[:d])
    raw = px > px.rolling(200).mean()
    assert bool(a.iloc[300]) == bool(raw.iloc[299])  # 오늘 신호는 어제 종가 기준


def test_vol_target_uses_only_past():
    px = _px(1)
    rf = pd.Series(0.02, index=px.index)
    cash = pd.Series(0.0, index=px.index)
    d = px.index[450]
    a = r8.vol_target_letf(px, r8.synthetic_letf(px, rf, 3), cash, 3, 0.25, on=r8.trend_signal(px))
    m = _mutate_after(px, d)
    b = r8.vol_target_letf(m, r8.synthetic_letf(m, rf, 3), cash, 3, 0.25, on=r8.trend_signal(m))
    # 판단(주말)→다음 날 반영이라 d 다음 날까지의 비중·수익은 같아야 한다
    pd.testing.assert_series_equal(a.loc[:d], b.loc[:d])


def test_synthetic_letf_formula():
    px = pd.Series([100.0, 101.0, 99.0], index=pd.bdate_range("2020-01-01", periods=3))
    rf = pd.Series(0.05, index=px.index)
    r = r8.synthetic_letf(px, rf, 3)
    assert abs(r.iloc[1] - (3 * 0.01 - 2 * 0.055 / 252 - 0.0095 / 252)) < 1e-12


def test_tom_mask_days():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    on = r8.tom_mask(idx, before=1, after=3)
    days = on[on].index
    # 1월 마지막 거래일(1/31)과 2월 첫 3거래일(2/3~2/5)
    assert pd.Timestamp("2020-01-31") in days and pd.Timestamp("2020-02-05") in days
    assert pd.Timestamp("2020-02-06") not in days and pd.Timestamp("2020-01-30") not in days
    assert int(on.loc["2020-02"].sum()) == 3 + 1  # 2월: 월초 3일 + 2/28


def test_switch_curve_charges_cost_only_on_switch():
    idx = pd.bdate_range("2020-01-01", periods=5)
    risk = pd.Series([0.0, 0.1, 0.1, 0.1, 0.1], index=idx)
    cash = pd.Series(0.0, index=idx)
    on = pd.Series([False, True, True, False, False], index=idx)
    c = r8.switch_curve(risk, cash, on, 0.01)
    assert abs(c.iloc[1] - (1 + 0.1 - 0.01)) < 1e-12
    assert abs(c.iloc[2] / c.iloc[1] - 1.1) < 1e-12
    assert abs(c.iloc[3] / c.iloc[2] - 0.99) < 1e-12
    assert abs(c.iloc[4] / c.iloc[3] - 1.0) < 1e-12


def test_after_tax_deducts_22pct_over_allowance():
    idx = pd.bdate_range("2021-01-01", "2021-12-31")
    curve = pd.Series(np.linspace(1.0, 1.10, len(idx)), index=idx)  # 1년 +10% → 1억 기준 1,000만 이익
    s = r8.after_tax(curve)
    expected = 1.10 - 0.22 * (1e7 - 2.5e6) / 1e8
    assert abs(float(s.iloc[-1]) - expected) < 1e-9
    down = pd.Series(np.linspace(1.0, 0.9, len(idx)), index=idx)
    assert abs(float(r8.after_tax(down).iloc[-1]) - 0.9) < 1e-9  # 손실이면 세금 없음
