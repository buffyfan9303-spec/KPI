"""재무 5점 F-score-lite(T8)와 영업이익 가속 필터(T9) (quant-researcher 명세, 라운드6, 2026-09-25).

둘 다 사업보고서(연간) 전년 대비 비교. avail 규칙은 research.engine._accounts_at과 동일
(filed와 이듬해 4/15 중 이른 쪽부터 사용 가능).
"""
import pandas as pd

from research import data

ACCEL_START = pd.Timestamp("2018-04-01")


def _accounts_avail(d: pd.Timestamp, fy: int) -> pd.DataFrame:
    """d 시점에 공개돼 있던 fy 사업연도 사업보고서(종목별)."""
    if fy < 2015:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in ("net", "equity", "sales", "op", "assets")})
    a = data.annual_accounts(fy).copy()
    avail = a["filed"].where(a["filed"] < pd.Timestamp(fy + 1, 4, 15), pd.Timestamp(fy + 1, 4, 15))
    return pd.DataFrame(a[avail <= d])


def latest_fy(d: pd.Timestamp) -> int:
    """d 시점에 이용 가능한 가장 최근 사업연도(4/15 이전엔 재작년치가 최신)."""
    return d.year - 1 if d >= pd.Timestamp(d.year, 4, 15) else d.year - 2


def fscore(d: pd.Timestamp, codes) -> pd.Series:
    """5점 F-score-lite: ROA>0, ΔROA>0, Δ(영업이익/매출)>0, Δ(매출/자산)>0, Δ(자본/자산)>0."""
    fy = latest_fy(d)
    cur, prev = _accounts_avail(d, fy), _accounts_avail(d, fy - 1)
    df = cur.join(prev, how="inner", lsuffix="_c", rsuffix="_p")
    if df.empty:
        return pd.Series(0, index=pd.Index(codes), dtype=int)
    roa_c, roa_p = df["net_c"] / df["assets_c"], df["net_p"] / df["assets_p"]
    ops_c, ops_p = df["op_c"] / df["sales_c"], df["op_p"] / df["sales_p"]
    sa_c, sa_p = df["sales_c"] / df["assets_c"], df["sales_p"] / df["assets_p"]
    ea_c, ea_p = df["equity_c"] / df["assets_c"], df["equity_p"] / df["assets_p"]
    score = ((roa_c > 0).astype(int) + (roa_c > roa_p).astype(int) + (ops_c > ops_p).astype(int)
             + (sa_c > sa_p).astype(int) + (ea_c > ea_p).astype(int))
    return score.reindex(codes).fillna(0).astype(int)


def accel_ok(d: pd.Timestamp, codes) -> pd.Series:
    """T9: (ΔOP_FY − ΔOP_FY-1)/자산 > 0 (lag=2 사업보고서). 2018-04 이전엔 필터 없음(전부 통과)."""
    if d < ACCEL_START:
        return pd.Series(True, index=pd.Index(codes))
    fy = latest_fy(d)
    a0, a1, a2 = _accounts_avail(d, fy), _accounts_avail(d, fy - 1), _accounts_avail(d, fy - 2)
    if a0.empty or a1.empty or a2.empty:
        return pd.Series(False, index=pd.Index(codes))
    op1 = pd.DataFrame(a1[["op"]]).rename(columns={"op": "op_1"})
    op2 = pd.DataFrame(a2[["op"]]).rename(columns={"op": "op_2"})
    df = pd.DataFrame(a0[["op", "assets"]]).join(op1, how="inner").join(op2, how="inner")
    accel = ((df["op"] - df["op_1"]) - (df["op_1"] - df["op_2"])) / df["assets"]
    return (accel > 0).reindex(codes).fillna(False)
