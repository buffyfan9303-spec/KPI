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


def test_value_screen_small_cap_low_per_pbr():
    from app.selection import value_screen
    listing = pd.DataFrame({"Name": list("abcdef"), "Marcap": [10, 20, 30, 40, 50, 1000.0]},
                           index=list("123456"))
    acc = pd.DataFrame({"net": [5, 1, -3, 10, 10, 10.0], "equity": [20, 20, 20, 10, 100, 10.0]},
                       index=list("123456"))
    out = value_screen(listing, acc, small_q=0.8, top=2)
    # 시총 하위 80% = 1~5, 적자(3) 제외 → PER 2,20,4,5 / PBR 0.5,1,4,0.5 → 순위합 최소: 1, 그다음 5
    assert out.index.tolist() == ["1", "5"]


def test_news_gate_rejects_confident_negative_only():
    from app.news_gate import gate
    assert gate([{"title": "OO건설 대표 횡령 혐의 기소", "label": "negative"}]) == "REJECT"
    assert gate([{"title": "삼성전자 다시 들썩", "label": "negative"}]) == "EXECUTE"  # 단어 없는 부정은 통과
    assert gate([{"title": "거래정지 해제, 급등", "label": "positive"}]) == "EXECUTE"  # 긍정이면 통과
    assert gate([]) == "EXECUTE"


def test_research_tax_and_metrics():
    from research.engine import metrics, sell_tax
    assert sell_tax(pd.Timestamp("2018-05-01")) == 0.0030
    assert sell_tax(pd.Timestamp("2025-06-01")) == 0.0015
    assert sell_tax(pd.Timestamp("2026-09-01")) == 0.0020
    idx = pd.date_range("2020-01-01", "2022-01-01", freq="D")
    s = pd.Series(2.0 ** ((idx - idx[0]).days / 730.5), index=idx) * 5  # 2년에 2배, 시작값 5
    assert abs(metrics(s)["cagr"] - (2 ** 0.5 - 1)) < 1e-3  # 시작값과 무관


def test_dsr_rejects_best_of_noise():
    import numpy as np
    from research.validate import deflated_sharpe
    rng = np.random.default_rng(1)
    rs = [pd.Series(rng.normal(0, 0.01, 2500)) for _ in range(50)]
    srs = [float(r.mean() / r.std()) for r in rs]
    assert deflated_sharpe(rs[int(np.argmax(srs))], 50, float(np.var(srs, ddof=1)))["dsr"] < 0.95
