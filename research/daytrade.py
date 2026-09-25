"""ETF 데이트레이드 3종 (T5/T6/T7, quant-researcher 명세 2026-09-25).

각 함수는 일별 자산곡선(pd.Series, 1.0 시작)을 반환한다. 편도 비용 0.05%(cost), 세금 없음(ETF).
신호는 종가로 판정하고 체결은 다음날 시가(변동성 돌파는 당일 목표가 도달 시 즉시 체결)로 한다.
ohlc는 Open/High/Low/Close 열을 가진 FinanceDataReader 결과.
"""
import pandas as pd

COST = 0.0005


def volatility_breakout(ohlc: pd.DataFrame, k: float = 0.5, cost: float = COST) -> pd.Series:
    """T5: 목표가 = 오늘 시가 + k*(전일 고가-전일 저가). 장중 고가가 그 값 이상이면 그 값에 매수, 다음날 시가에 청산(하루 최대 1회)."""
    o, h = ohlc["Open"], ohlc["High"]
    target = o + k * (ohlc["High"] - ohlc["Low"]).shift(1)
    hit = (h >= target) & target.notna()
    exit_px = o.shift(-1)
    ret = pd.Series(0.0, index=ohlc.index)
    ok = hit & exit_px.notna()
    ret[ok] = (exit_px[ok] - target[ok]) / target[ok] - 2 * cost
    return pd.Series((1 + ret).cumprod(), index=ohlc.index)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return pd.Series(100 - 100 / (1 + up / dn))


def rsi2_mr(ohlc: pd.DataFrame, cost: float = COST) -> pd.Series:
    """T6: 종가>200일선 & RSI(2)<10 → 다음날 시가 매수. 종가>5일선 되면 다음날 시가 청산."""
    close, o = pd.Series(ohlc["Close"]), pd.Series(ohlc["Open"])
    buy_today = pd.Series(((close > close.rolling(200).mean()) & (_rsi(close, 2) < 10)).shift(1).fillna(False))
    sell_today = pd.Series((close > close.rolling(5).mean()).shift(1).fillna(False))
    return _loop(o, close, buy_today, sell_today, cost)


def donchian(ohlc: pd.DataFrame, entry_n: int = 55, exit_n: int = 20, atr_n: int = 20, atr_k: float = 2.0,
            cost: float = COST) -> pd.Series:
    """T7: 종가>직전 55일 고점 → 다음날 시가 매수. 종가<직전 20일 저점 또는 진입가-2*ATR20 하회 → 다음날 시가 청산."""
    o, h, low, close = pd.Series(ohlc["Open"]), pd.Series(ohlc["High"]), pd.Series(ohlc["Low"]), pd.Series(ohlc["Close"])
    high_prior = h.shift(1).rolling(entry_n).max()
    ll20 = low.shift(1).rolling(exit_n).min()
    prev_close = close.shift(1)
    tr = pd.concat([h - low, (h - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr.rolling(atr_n).mean())
    buy_today = pd.Series((close > high_prior).shift(1).fillna(False))
    sell_today = pd.Series((close < ll20).shift(1).fillna(False))
    return _loop(o, close, buy_today, sell_today, cost, atr=atr, atr_k=atr_k)


def _loop(o: pd.Series, close: pd.Series, buy_today: pd.Series, sell_today: pd.Series, cost: float,
         atr: pd.Series | None = None, atr_k: float = 0.0) -> pd.Series:
    idx = o.index
    eq, out, pos, entry_price, stop = 1.0, [], False, 0.0, float("-inf")
    for i in range(len(idx)):
        ret = 0.0
        if not pos:
            if bool(buy_today.iloc[i]) and pd.notna(o.iloc[i]) and pd.notna(close.iloc[i]):
                entry_price = float(o.iloc[i])
                pos = True
                a = float(atr.iloc[i]) if atr is not None and pd.notna(atr.iloc[i]) else 0.0
                stop = entry_price - atr_k * a if atr is not None else float("-inf")
                ret = close.iloc[i] / entry_price - 1 - cost
        else:
            stop_hit = i > 0 and float(close.iloc[i - 1]) <= stop
            if bool(sell_today.iloc[i]) or stop_hit:
                ret = o.iloc[i] / close.iloc[i - 1] - 1 - cost
                pos = False
            else:
                ret = close.iloc[i] / close.iloc[i - 1] - 1 if i > 0 else 0.0
        eq *= 1 + ret
        out.append(eq)
    return pd.Series(out, index=idx)
