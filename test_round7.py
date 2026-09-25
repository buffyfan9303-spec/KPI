"""7차 신호·이벤트 엔진의 미래 참조 방지 테스트. 리밸런싱일 d 이후 자료를 바꿔도 d의 선정·신호가 같아야 한다."""
import numpy as np
import pandas as pd

from research import round7 as r7


def _panel(seed: int = 0, n_days: int = 320, n_codes: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    codes = [f"{i:05d}0" for i in range(n_codes)]
    Rraw = pd.DataFrame(rng.normal(0, 0.03, (n_days, n_codes)), index=idx, columns=codes)
    Rraw.iloc[50, 0] = -0.20  # 급락일 하나(avoid_matrix·crash_signal이 비어 있지 않게)
    R = Rraw.clip(-r7.LIMIT, r7.LIMIT)
    P = (1 + R).cumprod()
    vol = pd.DataFrame(rng.integers(1000, 5000, (n_days, n_codes)).astype(float), index=idx, columns=codes)
    close = P * 1000
    return dict(Rraw=Rraw, R=R, P=P, vol=vol, vol20=vol.shift(1).rolling(20, min_periods=10).mean(),
                amt20=pd.DataFrame(1e9, index=idx, columns=codes), mcap=pd.DataFrame(np.tile(rng.uniform(1e9, 1e12, n_codes), (n_days, 1)), index=idx, columns=codes),
                open=close.shift(1).fillna(1000.0), close=close, market=pd.Series("KOSPI", index=codes),
                dead_next=pd.Series(dtype="datetime64[ns]"), delisted=set())


def _mutate_future(pn: dict, d: pd.Timestamp) -> dict:
    """d 이후 행을 전부 뒤섞은 복사본."""
    out = {}
    for k, v in pn.items():
        if isinstance(v, pd.DataFrame):
            v = v.copy()
            v.loc[v.index > d] = v.loc[v.index > d].to_numpy()[::-1] * 3.0 + 0.5
        out[k] = v
    return out


def test_scores_use_only_past_data():
    pn = _panel()
    d = pn["P"].index[280]
    alt = _mutate_future(pn, d)
    for style in r7.STYLES:
        codes = r7.tier_codes(pn, d, "ALL", "mid")
        a, b = r7.score(pn, d, codes, style), r7.score(alt, d, codes, style)
        pd.testing.assert_series_equal(a, b, check_names=False)
        assert len(a) > 0, style


def test_event_signals_use_only_past_data():
    pn = _panel(seed=1)
    d = pn["P"].index[200]
    alt = _mutate_future(pn, d)
    for f in (lambda p: r7.surge_signal(p, lo=0.03, vol_mult=1.0), lambda p: r7.crash_signal(p, drop=-0.03, vol_mult=1.0),
              lambda p: r7.breakout_signal(p, window=20, vol_mult=1.0), r7.avoid_matrix):
        a, b = f(pn).loc[:d], f(alt).loc[:d]
        pd.testing.assert_frame_equal(a, b)
        assert a.to_numpy().sum() > 0


def test_event_curve_enters_next_open_and_delays_exit_on_limit_down(monkeypatch):
    """신호 t → t+1 시가 매수(같은 날 시가→종가) → t+1 종가 매도(hold=1). t+1이 하한가면 t+2 종가로 청산이 밀린다."""
    monkeypatch.setattr(r7, "START", "2020-01-01")
    monkeypatch.setattr(r7, "END", "2020-12-31")
    monkeypatch.setattr(r7, "sell_tax", lambda d: 0.0)
    idx = pd.bdate_range("2020-03-02", periods=6)
    c = ["000010", "000020"]
    close = pd.DataFrame({"000010": [100, 100, 110, 120, 120, 120], "000020": [100, 100, 70, 80, 80, 80]}, index=idx, dtype=float)
    opn = pd.DataFrame({"000010": [100, 100, 100, 110, 120, 120], "000020": [100, 100, 100, 70, 80, 80]}, index=idx, dtype=float)
    Rraw = close.pct_change()
    pn = dict(Rraw=Rraw, R=Rraw.clip(-0.3, 0.3), open=opn, close=close, vol=pd.DataFrame(1.0, index=idx, columns=c),
              dead_next=pd.Series(dtype="datetime64[ns]"), delisted=set())
    sig = pd.DataFrame(False, index=idx, columns=c)
    sig.loc[idx[1], :] = True  # 두 종목 모두 t=idx[1] 신호
    curve, info = r7.event_curve(pn, sig, hold=1, slip=0.0, max_pos=1)
    r = curve.pct_change().fillna(curve.iloc[0] - 1)
    # t+1: 000010 시가100→종가110 (+10%), 000020 시가100→종가70 (-30%, 하한가 → 못 팖). 두 포지션 평균 = -10%
    assert abs(r.loc[idx[2]] - (-0.10)) < 1e-6
    # t+2: 000010은 t+1 종가에 팔렸고, 000020만 남아 70→80 (+14.3%)로 청산
    assert abs(r.loc[idx[3]] - (80 / 70 - 1)) < 1e-6
    assert r.loc[idx[4]] == 0.0
    assert info["n_events"] == 2
