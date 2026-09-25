"""머신러닝 횡단면 종목 선택 (그래디언트 부스팅, Noh 2023·JKSS 2023의 한국 결과를 따름).

- 분기마다(1·4·7·10월 15일 이후 첫 거래일) 팩터 특징 X와 '다음 리밸런싱까지 수익률 순위' y를 만든다.
- 워크포워드: 시점 t에는 라벨이 이미 확정된(다음 리밸런싱이 t 이전인) 과거 표본만으로 학습 → t에 예측 → 상위 N 선택.
- 하이퍼파라미터는 sklearn 기본값 고정(튜닝 안 함).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research import data, factors
from research.engine import EXCLUDE, Config, _accounts_at, _rebalance_dates

FEATS = ["per", "pbr", "psr", "opa", "roe", "ey", "ag", "eg", "mom", "h52", "vol", "rev", "maxr", "turn", "logcap"]


def panel(cfg: Config) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    px = data.prices()
    R = data.returns_wide(px)
    P = (1 + R.fillna(0.0)).cumprod().where(R.notna().cummax())
    amt20 = px.pivot_table(index="Date", columns="Code", values="Amount", aggfunc="last").rolling(20, min_periods=10).mean()
    idx = pd.DatetimeIndex(R.index[(R.index >= pd.Timestamp(cfg.start)) & (R.index <= pd.Timestamp(cfg.end))])
    dates = _rebalance_dates(idx, cfg)
    by_date = {k: g.set_index("Code") for k, g in px[px["Date"].isin(dates)].groupby("Date")}
    # 엔진과 같은 규칙: 거래정지 상태로 사라진 종목은 0원(-100%). P는 이후 평평해 라벨이 0% 근처로 부풀려진다.
    last = px.groupby("Code").tail(1).set_index("Code")
    frozen_exit = last[(last["Volume"] == 0) & (last["Date"] < px["Date"].max())]["Date"]
    rows = []
    for k, d in enumerate(dates):
        snap = by_date[d]
        u = snap[snap.index.str.endswith("0") & ~snap["Name"].str.contains(EXCLUDE, na=False, regex=True)
                 & (snap["Volume"] > 0) & (pd.DataFrame(amt20).loc[d].reindex(snap.index).fillna(0) >= cfg.min_amount)]
        acc, prev = _accounts_at(d), _accounts_at(d, lag=1)
        prev = pd.DataFrame(prev[prev["fy"].reindex(prev.index) < acc["fy"].reindex(prev.index).fillna(9999)])
        f = factors.features(d, pd.DataFrame(u), P, R, acc, prev)
        f["logcap"] = np.log(f["marcap"])
        if k + 1 < len(dates):  # 다음 리밸런싱까지 수익률(정지는 평평, 정리매매 후 폐지는 마지막 값, 정지 상태 폐지는 -100%)
            nxt = dates[k + 1]
            fwd = pd.Series(P.loc[nxt] / P.loc[d] - 1).reindex(f.index)
            fwd[pd.Series(frozen_exit).reindex(f.index) < nxt] = -1.0  # 폐지일 < nxt 이므로 nxt 종가엔 이미 알려진 사실
            f["y"] = fwd.rank(pct=True)
        else:
            f["y"] = np.nan
        f["date"] = d
        rows.append(f)
    return pd.concat(rows), dates


def walk_forward_picks(cfg: Config, top_n: int = 30, min_train: int = 6, seed: int = 0,
                       df: pd.DataFrame | None = None, dates: list | None = None) -> dict[str, list[str]]:
    if df is None or dates is None:
        df, dates = panel(cfg)
    picks = {}
    for k, d in enumerate(dates):
        if k < min_train:
            continue
        # dates[k-1]의 라벨(다음 리밸런싱=오늘까지 수익)은 오늘 확정되므로 dates[:k]까지 학습 가능. 미래 라벨은 없음.
        train = df[df["date"].isin(dates[:k])]
        train = pd.DataFrame(train).dropna(subset=["y"])
        assert train["date"].max() < d  # 라벨 끝점(다음 리밸런싱)이 d 이하 → d 이후 가격 미사용
        model = HistGradientBoostingRegressor(random_state=seed)  # 기본값: 표본>1만이면 무작위 10% 조기종료 검증
        model.fit(train[FEATS], train["y"])
        test = df[df["date"] == d]
        pred = pd.Series(model.predict(test[FEATS]), index=test.index)
        picks[str(d.date())] = [str(c) for c in pred.nlargest(top_n).index]
    return picks
