"""sim2020 신호의 미래 참조 방지: t 이후 자료를 지워도 t까지의 신호(133690 추세·S 회전·DAA 월말 배분)가 같아야 한다.
외부 데이터 없이 합성 시계열로만 돈다."""
import numpy as np
import pandas as pd

from research import sim2020

DAA_COLS = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD", "SHY", "IEF", "BND"]


def test_signals_unchanged_when_future_removed():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2017-01-02", periods=900)

    def px(n):
        return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, (len(idx), n)), axis=0), index=idx)

    us = px(len(DAA_COLS))
    us.columns = DAA_COLS
    three = px(3)
    nq, s_curve, kospi = three[0], three[1], three[2]
    t = idx[idx <= "2019-11-30"][-1]  # 실제 월말(잘라도 월말 집합이 안 바뀌게)
    full = sim2020.signals(nq, s_curve, kospi, us, idx)
    cut = sim2020.signals(nq.loc[:t], s_curve.loc[:t], kospi.loc[:t], us.loc[:t], idx[idx <= t])
    pd.testing.assert_frame_equal(full.loc[:t], cut)
    # 검사가 공허하지 않은지: 세 신호 모두 t 이전에 실제로 값이 있고 변한다
    assert full["daa"].loc[:t].str.len().gt(0).sum() >= 12
    assert full["nq_on"].loc[:t].any() and not full["nq_on"].loc[:t].all()
    assert full["s_rot"].loc[:t].nunique() == 2
