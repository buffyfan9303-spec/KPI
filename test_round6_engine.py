import pandas as pd


def test_rank_value_and_select_basic(monkeypatch):
    from research import engine
    codes = ["000010", "000020", "000030", "000040", "000050", "999990"]
    snap = pd.DataFrame({"Name": list("abcdef"), "Market": ["KOSPI"] * 6, "Volume": [100] * 6,
                        "Marcap": [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0]}, index=codes)
    acc = pd.DataFrame({"net": [5.0, 1.0, -3.0, 10.0, 10.0, 10.0], "equity": [20.0, 20.0, 20.0, 10.0, 100.0, 10.0],
                        "sales": [100.0] * 6, "op": [10.0] * 6, "assets": [50.0] * 6}, index=codes)
    monkeypatch.setattr(engine, "_accounts_at", lambda d, lag=0: acc)
    amt20 = pd.Series(1e9, index=codes)
    cfg = engine.Config(small_q=0.8, top_n=2, min_amount=0)
    d = pd.Timestamp("2020-04-20")
    # 시총 하위 80% = 1~5, 적자(3) 제외 → PER 2,20,4,5 / PBR .5,1,4,.5 → 순위합 최소: 1, 그다음 5
    assert engine.select(d, snap, amt20, cfg) == ["000010", "000050"]
    assert len(engine.rank_value(d, snap, amt20, cfg)) == 4  # 1,2,4,5 (3은 적자 제외, 999990은 시총 상위 제외)


def test_hold_buffer_keeps_drifted_weight_and_fills_from_top(monkeypatch):
    from research import engine
    score = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    monkeypatch.setattr(engine, "rank_value", lambda d, snap, amt20, cfg: score)
    cfg = engine.Config(top_n=2, hold_buffer=1)  # buffer_n = 1*2 = 2
    current = pd.Series({"B": 0.7, "C": 0.3})  # B 순위2(<=2) 유지, C 순위3(>2) 이탈
    target, extra_q = engine._hold_buffer_target(pd.Timestamp("2020-01-01"), None, None, cfg, current, {},
                                                 pd.DataFrame(), 0)
    assert target["B"] == 0.7          # 기존 보유는 드리프트 비중 그대로(재조정 없음)
    assert "C" not in target.index     # 순위 밖 보유는 이탈
    assert abs(target["A"] - 0.3) < 1e-9  # 빈자리 1개를 상위 순위(A)가 남는 비중으로 채움
    assert extra_q == {}


def test_chandelier_only_stops_losers_never_winners():
    from research import engine
    d = pd.Timestamp("2020-01-01")
    w = pd.Series({"WIN": 0.5, "LOSS": 0.5})
    val = pd.Series({"WIN": 1.2, "LOSS": 0.8})  # WIN은 이익, LOSS는 손실
    close = pd.DataFrame({"WIN": [90.0], "LOSS": [90.0]}, index=[d])
    stop = pd.DataFrame({"WIN": [100.0], "LOSS": [100.0]}, index=[d])  # 둘 다 종가<이탈선
    assert engine._chandelier_hits(w, val, d, close, stop, bear=False) == ["LOSS"]


def test_chandelier_regime_bear_stops_winners_too():
    from research import engine
    d = pd.Timestamp("2020-01-01")
    w = pd.Series({"WIN": 0.5, "LOSS": 0.5})
    val = pd.Series({"WIN": 1.2, "LOSS": 0.8})
    close = pd.DataFrame({"WIN": [90.0], "LOSS": [90.0]}, index=[d])
    stop = pd.DataFrame({"WIN": [100.0], "LOSS": [100.0]}, index=[d])
    assert set(engine._chandelier_hits(w, val, d, close, stop, bear=True)) == {"WIN", "LOSS"}


def test_fscore_five_points_all_improving(monkeypatch):
    from research import fscore
    cur = pd.DataFrame({"net": [10.0], "equity": [60.0], "sales": [100.0], "op": [20.0], "assets": [100.0],
                        "filed": [pd.Timestamp("2020-03-01")]}, index=["000010"])
    prev = pd.DataFrame({"net": [5.0], "equity": [50.0], "sales": [100.0], "op": [10.0], "assets": [110.0],
                         "filed": [pd.Timestamp("2019-03-01")]}, index=["000010"])
    calls = {2019: cur, 2018: prev}
    monkeypatch.setattr(fscore, "_accounts_avail", lambda d, fy: calls.get(fy, prev.iloc[0:0]))
    d = pd.Timestamp("2020-05-01")  # fy=2019
    s = fscore.fscore(d, ["000010"])
    assert s["000010"] == 5  # ROA>0, 전부 개선


def test_accel_ok_before_2018_04_always_true():
    from research import fscore
    assert bool(fscore.accel_ok(pd.Timestamp("2018-01-01"), ["000010"])["000010"])


def test_daytrade_loop_marks_to_market_and_stops():
    from research.daytrade import _loop
    o = pd.Series([100.0, 101.0, 102.0, 103.0])
    close = pd.Series([100.0, 105.0, 110.0, 90.0])
    buy_today = pd.Series([False, True, False, False])
    sell_today = pd.Series([False, False, True, False])
    eq = _loop(o, close, buy_today, sell_today, cost=0.0)
    # 진입 101, 보유중 종가 반영(105/101-1), 청산 다음날 시가(102/105-1)
    assert abs(eq.iloc[1] - (105 / 101)) < 1e-9
    assert abs(eq.iloc[2] - (105 / 101) * (102 / 105)) < 1e-9
    assert eq.iloc[3] == eq.iloc[2]  # 청산 후 보유 없음 → 그대로


def test_daytrade_atr_stop_exits_next_open():
    from research.daytrade import _loop
    o = pd.Series([100.0, 101.0, 102.0, 103.0])
    close = pd.Series([100.0, 101.0, 90.0, 103.0])
    buy_today = pd.Series([False, True, False, False])
    sell_today = pd.Series([False, False, False, False])
    atr = pd.Series([float("nan"), 5.0, 5.0, 5.0])
    eq = _loop(o, close, buy_today, sell_today, cost=0.0, atr=atr, atr_k=1.0)
    # 진입가 101, 손절선 96. 2일차 종가 90<=96 → 3일차 시가(103)에 청산
    assert abs(eq.iloc[3] - eq.iloc[2] * (103 / 90)) < 1e-9


def test_volatility_breakout_hits_target():
    from research.daytrade import volatility_breakout
    ohlc = pd.DataFrame({"Open": [100.0, 103.0, 112.0], "High": [105.0, 110.0, 112.0],
                        "Low": [95.0, 100.0, 108.0], "Close": [102.0, 109.0, 111.0]})
    eq = volatility_breakout(ohlc, k=0.5, cost=0.0005)
    # 목표가 103+0.5*(105-95)=108, 고가110>=108 → 익일 시가112에 청산
    expect = (112 - 108) / 108 - 2 * 0.0005
    assert abs(eq.iloc[1] - (1 + expect)) < 1e-9
