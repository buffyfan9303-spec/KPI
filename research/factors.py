"""한국 주식 횡단면 팩터: 리밸런싱일 d 시점에 알 수 있는 값만으로 종목별 특징을 만든다.

가격 특징은 d 종가까지, 재무는 engine._accounts_at(d)(공시 이후) 기준.
spec 예: [("per", 1), ("opa", -1)] → PER 낮을수록·영업이익/자산 높을수록 좋음. 순위 합이 작은 종목부터 편입.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from research import data

# 팩터 이름 → (특징 조합, 설명). 파라미터는 문헌 표준값 그대로(튜닝 안 함)
FACTORS: dict[str, tuple[list[tuple[str, int]], str]] = {
    "value_pbr":      ([("pbr", 1)], "저PBR (Fama-French HML)"),
    "value_per":      ([("per", 1)], "저PER"),
    "value_psr":      ([("psr", 1)], "저PSR"),
    "value_combo":    ([("per", 1), ("pbr", 1), ("psr", 1)], "가치 3지표 합"),
    "quality_opa":    ([("opa", -1)], "영업이익/자산 높음 (Novy-Marx 수익성 대용)"),
    "quality_roe":    ([("roe", -1)], "ROE 높음"),
    "magic_formula":  ([("ey", -1), ("opa", -1)], "그린블라트 마법공식(이익수익률+자본수익률)"),
    "value_quality":  ([("per", 1), ("pbr", 1), ("opa", -1)], "가치+수익성"),
    "momentum_12_1":  ([("mom", -1)], "12-1개월 모멘텀 (Jegadeesh-Titman)"),
    "high_52w":       ([("h52", -1)], "52주 신고가 근접 (George-Hwang)"),
    "low_vol":        ([("vol", 1)], "저변동성 (Ang 등)"),
    "reversal_1m":    ([("rev", 1)], "1개월 단기 반전"),
    "low_max":        ([("maxr", 1)], "MAX 낮음(복권주 회피, Bali 등)"),
    "asset_growth":   ([("ag", 1)], "자산 증가율 낮음 (Cooper-Gulen-Schill)"),
    "earn_growth":    ([("eg", -1)], "순이익 증가율 높음"),
    "illiquidity":    ([("turn", 1)], "회전율 낮음(비유동성 프리미엄)"),
    "value_momentum": ([("per", 1), ("pbr", 1), ("mom", -1)], "가치+모멘텀 (Asness 등)"),
    "lowvol_value":   ([("vol", 1), ("pbr", 1)], "저변동+가치"),
    "qvm":            ([("per", 1), ("pbr", 1), ("opa", -1), ("mom", -1)], "가치+퀄리티+모멘텀"),
    "pead_op":        ([("sur_op", -1)], "영업이익 전년동기 대비 변화/자산 (Foster)"),
    "pead_sue":       ([("sue_op", -1)], "SUE (Bernard-Thomas)"),
}

_QORDER = ["Q1", "H1", "Q3", "FY"]  # 연내 순서(분기 보고서 종류)


def _period_key(year: int, q: str) -> int:
    """연도·보고서 종류 → 정수 순번(1 증가 = 다음 보고서, 4 증가 = 1년 전 같은 종류)."""
    return year * 4 + _QORDER.index(q)


@lru_cache(maxsize=128)
def _quarterly(year: int, q: str) -> pd.DataFrame:
    """분기 보고서 캐시(백테스트 리밸런싱마다 같은 연도·분기 재조회 방지)."""
    return data.quarterly_accounts(year, q)


def _delta(op: pd.Series, period: int) -> float:
    """Δ_k = op[k] − op[k−4] (전년동기 대비, 같은 보고서 종류)."""
    a, b = op.get(period), op.get(period - 4)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def _pead_row(g: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    """종목 하나의 point-in-time 보고서(filed < d)들로 sur_op·sue_op·filed_age 계산."""
    g = g.drop_duplicates(subset="pkey").set_index("pkey").sort_index()
    t = max(int(x) for x in g.index)
    age = float((d - g.at[t, "filed"]).days)
    op = pd.Series(g["op"])
    assets_t = g.at[t, "assets"]
    sur = np.nan
    if pd.notna(assets_t) and float(assets_t) > 0:
        prev = _delta(op, t)  # op[t]-op[t-4] 자체가 분자(전년동기 대비 변화)
        if pd.notna(prev):
            sur = prev / float(assets_t)
    delta_t = _delta(op, t)
    deltas = [_delta(op, k) for k in range(t - 1, t - 9, -1)]
    valid = [x for x in deltas if pd.notna(x)]
    sue = np.nan
    if pd.notna(delta_t) and len(valid) >= 6:
        sd = float(np.std(valid, ddof=0))
        if sd > 0:
            sue = delta_t / sd
    if age > 120:  # 120일 넘게 지난 보고서는 신선도 부족으로 신호 폐기
        sur, sue = np.nan, np.nan
    return pd.Series({"sur_op": sur, "sue_op": sue, "filed_age": age})


def pead_features(d: pd.Timestamp, codes: pd.Index) -> pd.DataFrame:
    """d 시점에 알 수 있었던(filed < d) 최신 분기·연간 보고서로 어닝 서프라이즈 특징을 만든다."""
    codes = pd.Index(codes).unique()
    out = pd.DataFrame(index=codes, columns=pd.Index(["sur_op", "sue_op", "filed_age"]), dtype=float)
    periods = [(y, q) for y in range(max(2015, d.year - 4), d.year + 1) for q in _QORDER
               if q == "FY" or y >= 2016]
    rows = []
    for y, q in periods:
        df = _quarterly(y, q).reindex(codes)[["op", "assets", "filed"]]
        df.index.name = "Code"
        sub = df.reset_index()
        sub["pkey"] = _period_key(y, q)
        rows.append(sub)
    if not rows:
        return out
    long = pd.concat(rows, ignore_index=True).dropna(subset=["filed"])
    long = long[long["filed"] < d]
    if long.empty:
        return out
    res = long.groupby("Code", group_keys=False)[["op", "assets", "filed", "pkey"]].apply(lambda g: _pead_row(g, d))
    return out.drop(columns=list(res.columns)).join(res).reindex(codes)


def features(d: pd.Timestamp, snap: pd.DataFrame, P: pd.DataFrame, R: pd.DataFrame,
             acc: pd.DataFrame, acc_prev: pd.DataFrame) -> pd.DataFrame:
    """snap: d일 종목 스냅샷(index=Code). P: 수정가격지수(상장 전 NaN). R: 일간 수익률."""
    i = int(P.index.get_indexer([d])[0])
    f = pd.DataFrame(index=snap.index)
    f["marcap"] = snap["Marcap"]
    a = acc.reindex(f.index)
    f["per"] = snap["Marcap"] / a["net"]
    f["pbr"] = snap["Marcap"] / a["equity"]
    f["psr"] = snap["Marcap"] / a["sales"]
    f["opa"] = a["op"] / a["assets"]
    f["roe"] = a["net"] / a["equity"]
    f["ey"] = a["op"] / snap["Marcap"]
    ap = acc_prev.reindex(f.index)
    f["ag"] = a["assets"] / ap["assets"] - 1
    f["eg"] = (a["net"] - ap["net"]) / ap["net"].abs()
    if i >= 252:
        f["mom"] = (P.iloc[i - 21] / P.iloc[i - 252] - 1).reindex(f.index)
        f["h52"] = (P.iloc[i] / P.iloc[i - 251: i + 1].max()).reindex(f.index)
        win = R.iloc[i - 251: i + 1]
        f["vol"] = win.std().reindex(f.index)
    if i >= 21:
        f["rev"] = (P.iloc[i] / P.iloc[i - 21] - 1).reindex(f.index)
        f["maxr"] = R.iloc[i - 20: i + 1].max().reindex(f.index)
    f["turn"] = snap["Amount"] / snap["Marcap"]
    f = f.join(pead_features(d, f.index))
    return f.replace([np.inf, -np.inf], np.nan)


def pick(f: pd.DataFrame, spec: list[tuple[str, int]], n: int) -> list[str]:
    cols = [c for c, _ in spec]
    g = pd.DataFrame(f.dropna(subset=[c for c in cols if c in f.columns]))
    for c in ("per", "pbr", "psr"):  # 적자·자본잠식은 가치 비율 의미 없음
        if c in cols:
            g = pd.DataFrame(g[g[c] > 0])
    if g.empty or any(c not in g.columns for c in cols):
        return []
    if n <= 0:  # 0이면 상위 20%(5분위) 전체 — 학계 표준 팩터 검정용
        n = max(len(g) // 5, 1)
    score = sum(pd.Series(g[c]).rank(ascending=(sign == 1)) for c, sign in spec)
    return [str(c) for c in pd.Series(score).nsmallest(n).index]
