"""가상 실측(paper-trading 방식 전진 시뮬레이션) 2020-01-02 ~ 최신 자료일, 초기자본 1억 원.

왜 따로 돌리나: 기존 곡선(2017-10~)은 일부가 후보 선택에 쓰였다. 여기서는 2020-01-02에 1억 원으로 새로 시작해
매수 비용·월말 재조정·세금을 원화로 직접 집계한다(기존 곡선을 기준일로 나누는 '리베이스'는 교차검증에만 쓴다).

규칙(전부 round8.py / build_final.py에 이미 있던 그대로, 파라미터 재조정 없음)
- S    : 소형 하위10% + 가치(PER·PBR) + 수익성 상위50%, 분기(1·4·7·10월 15일), 슬리피지 1% + 매도세,
         252일 성과가 KODEX200보다 낮으면 월말에 KODEX200으로 전환(전환 비용 2.5%).
- DAA  : Keller DAA-G12, 월말, 신호 = 미국 ETF 전일 종가, 매매 = 국내 상장 ETF(편도 0.1%).
- PP   : 영구 포트폴리오(SPY·TLT·GLD·BIL 25%씩 → 국내 ETF), 월말.
- LETF : TIGER 나스닥100(133690) 종가 > 200일선이면 보유, 아니면 KODEX 단기채(153130). 매일 판단→다음 날, 편도 0.05%,
         매년 말 그해 이익에 15.4% 과세(전부 실현 가정).
- L3   : QQQ 합성 3배 + 200일선(달러) → 원화 환산 → 매년 말 (이익−250만)×22% 과세.
- 조합 : 월말에 목표 비중으로 재조정(슬리브별 편도 비용 S 1.2%, DAA·PP 0.1%, LETF·L3 0.25%).
- C4   : 조합 곡선에 변동성 목표 20%(60일, 주 1회, 비용 1%). H: CPPI(바닥 80%, 승수 4, 주 1회).

시작 방식: 2020-01-02 종가에 각 슬리브의 '그날 모델 보유 종목'을 목표 비중대로 매수(편도 비용 지불).
슬리브 신호·보유 종목은 2016-04-15부터 규칙대로 굴린 상태(워밍업)이며, t 시점 신호는 t까지 자료만 쓴다(test_sim2020.py).
세금은 슬리브 단위 근사(그해 이익 전부 실현, 손실 이월 없음; L3 공제는 1억 원 기준 — round8과 동일, 실제 30% 슬리브면 공제가 상대적으로 커서 보수적).
정수 주식 수·호가 단위는 반영하지 않았다(1억 원·20종목이면 영향 미미).

실행: python -m research.sim2020 → research/sim2020/{summary,yearly,stress,trades,crosscheck}.csv, equity.parquet, events_B.csv, *.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research import alloc  # noqa: E402
from research.combo import blended_cost, cppi  # noqa: E402
from research.engine import Config, rotate_signal, run, vol_target  # noqa: E402
from research.round8 import (KR_ETF_COST, US_COST, US_FX_COST, after_tax, kr_etf, synthetic_letf, switch_curve,  # noqa: E402
                             to_krw, trend_signal, us)

HERE = Path(__file__).parent
OUT = HERE / "sim2020"
START, END, CAPITAL = "2020-01-02", "2026-09-23", 1e8
WARM = "2016-04-15"  # 슬리브 규칙 워밍업 시작(자본 계산은 START부터)
COSTS = {"S": 0.012, "DAA": 0.001, "PP": 0.001, "LETF": US_COST, "L3": US_COST, "KOSPI": KR_ETF_COST, "NQ": KR_ETF_COST}
ENTRY_EXTRA = {"L3": US_FX_COST}  # 첫날 환전 비용
SCENARIOS = {
    "A 최고 S40+DAA30+QQQx3추세30": {"S": .4, "DAA": .3, "L3": .3},
    "B 추천 S40+DAA30+나스닥추세30": {"S": .4, "DAA": .3, "LETF": .3},
    "C B+변동성목표20": "vol:B 추천 S40+DAA30+나스닥추세30",
    "D 기존최선 S40+DAA30+PP30": {"S": .4, "DAA": .3, "PP": .3},
    "E S30+DAA30+PP20+나스닥추세20": {"S": .3, "DAA": .3, "PP": .2, "LETF": .2},
    "G 나스닥추세 단독(세후)": {"LETF": 1.0},
    "H D+CPPI": "cppi:D 기존최선 S40+DAA30+PP30",
    "KOSPI200 보유(069500)": {"KOSPI": 1.0},
    "나스닥100 보유(133690, 세전)": {"NQ": 1.0},
}
STRESS = [("2020 코로나(2/3~3/31)", "2020-02-03", "2020-03-31"), ("2022 약세장", "2022-01-03", "2022-12-29"),
          ("2024-08 급락(8/1~8/5)", "2024-08-01", "2024-08-05"), ("2024-08 전체", "2024-08-01", "2024-08-30"),
          ("최근 12개월", (pd.Timestamp(END) - pd.DateOffset(years=1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), END)]


def _month_ends(ix: pd.DatetimeIndex) -> set:
    return set(pd.Series(ix, index=ix).groupby(ix.to_period("M")).max().to_numpy())


# ---------- 신호(순수 함수: t까지 자료만) ----------
def signals(nq_close: pd.Series, s_curve: pd.Series, kospi_close: pd.Series, us_px: pd.DataFrame,
            kr_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """B 시나리오 신호. nq_on: t일 보유 여부(t-1 종가>200일선), s_rot: 1=S 보유·0=KODEX200(전월 말 판단),
    daa: 월말 배분(JSON, 미국 전일 종가 기준). 모두 t까지의 자료만 쓴다."""
    ix = pd.DatetimeIndex(kr_idx)
    nq_on = trend_signal(nq_close).reindex(ix).fillna(False).astype(bool)
    rb = kospi_close.pct_change().reindex(s_curve.index).fillna(0.0)
    s_rot = rotate_signal(s_curve, rb).reindex(ix).ffill()
    sig = us_px.shift(1).reindex(ix, method="ffill").ffill()  # alloc.backtest와 동일(결측 전일값)
    daa = pd.Series("", index=ix, dtype=object)
    for d in sorted(_month_ends(ix)):
        i = int(sig.index.get_loc(d))
        if i >= 252:
            daa[d] = json.dumps(alloc.daa_g12(sig, i), sort_keys=True)
    return pd.DataFrame({"nq_on": nq_on, "s_rot": s_rot, "daa": daa})


# ---------- 슬리브(정규화 곡선의 일간 수익·비용·세금 비율) ----------
def _tax_frac(pre: pd.Series, post: pd.Series) -> pd.Series:
    """after_tax 전후 곡선에서 그날 차감된 세금 비율(연말에만 0이 아님)."""
    return (1 - (1 + post.pct_change()) / (1 + pre.pct_change())).fillna(0.0).clip(lower=0.0)


def sleeve_s() -> dict:
    b = run(Config(name="S", rebalance="quarterly", small_q=0.10, quality_q=0.5, start=WARM, end=END), log=False)
    curve = b["curve"]
    rb = kr_etf()["069500"].pct_change().reindex(curve.index).fillna(0.0)
    sig = rotate_signal(curve, rb, 252)
    rs = curve.pct_change().fillna(0.0)
    sw = sig.diff().abs().fillna(0.0)
    ret = pd.Series(np.where(sig == 1, rs, rb), index=curve.index) - sw * 0.025
    cost = b["cost"].where(sig == 1, 0.0) + sw * 0.025
    return {"ret": ret, "cost": cost, "tax": pd.Series(0.0, index=curve.index), "sig": sig,
            "picks": b["picks"], "turnover": b["turnover"], "curve": curve}


def sleeve_alloc(key: str) -> dict:
    name = {"DAA": "DAA_G12", "PP": "영구포트폴리오"}[key]
    usp = alloc.us_prices()
    need = sorted({a for v in alloc.NEEDS.values() for a in v} | {"BIL"})
    kr = alloc.kr_prices(need)
    sig = usp.shift(1).reindex(kr.index, method="ffill")  # 한국장 마감 시점엔 미국 전일 종가까지만
    cols = alloc.NEEDS[name] + (["BIL"] if "BIL" not in alloc.NEEDS[name] else [])
    alloc.COST = 0.001
    d = alloc.backtest(alloc.STRATEGIES[name], sig, pd.DataFrame(kr[cols]), WARM, END, detail=True)
    return {"ret": d["curve"].pct_change().fillna(0.0), "cost": d["cost"], "tax": d["cost"] * 0.0, "alloc": d["alloc"]}


def sleeve_letf() -> dict:
    e = kr_etf().loc[WARM:END]
    on = trend_signal(kr_etf()["133690"]).reindex(e.index).fillna(False).astype(bool)
    pre = switch_curve(e["133690"].pct_change().fillna(0.0), e["153130"].pct_change().fillna(0.0), on, KR_ETF_COST)
    post = after_tax(pre, rate=0.154, allowance=0.0)
    sw = on.astype(int).diff().abs().fillna(0.0)
    return {"ret": post.pct_change().fillna(0.0), "cost": sw * KR_ETF_COST, "tax": _tax_frac(pre, post), "on": on}


def sleeve_l3(kr_idx: pd.DatetimeIndex) -> dict:
    d = us()
    p, rf = d["p"], d["rf"]
    syn = synthetic_letf(p["QQQ"], rf, 3)
    on = trend_signal(p["QQQ"])
    pre = to_krw(switch_curve(syn, p["BIL"].pct_change().fillna(0.0), on, US_COST))
    post = after_tax(pre)  # 22%, 250만 공제(1억 기준)
    cost_us = on.astype(int).diff().abs().fillna(0.0) * US_COST
    tax_us = _tax_frac(pre, post)
    curve = post.reindex(kr_idx).ffill()

    def to_kr(x: pd.Series) -> pd.Series:  # 미국 거래일 비용을 다음 한국 거래일에 귀속
        return x.cumsum().reindex(kr_idx, method="ffill").fillna(0.0).diff().fillna(0.0)

    return {"ret": curve.pct_change().fillna(0.0), "cost": to_kr(cost_us), "tax": to_kr(tax_us), "on": on}


def sleeve_hold(code: str) -> dict:
    c = kr_etf()[code].loc[WARM:END]
    r = c.pct_change().fillna(0.0)
    return {"ret": r, "cost": r * 0.0, "tax": r * 0.0}


# ---------- 원화 회계 ----------
def portfolio(sl: dict[str, dict], weights: dict[str, float], start: str = START, capital: float = CAPITAL) -> dict:
    """start 종가에 목표 비중대로 매수(편도 비용), 이후 매일 표류, 월말 목표 비중으로 재조정(combo.combine 규칙).
    반환: eq(원화), cost_krw·tax_krw(일별 원화), w(일별 슬리브 비중, 수익 반영 후)."""
    df = pd.DataFrame({k: sl[k]["ret"] for k in weights}).dropna().loc[start:END]
    ix = pd.DatetimeIndex(df.index)
    cf = pd.DataFrame({k: sl[k]["cost"].reindex(ix).fillna(0.0) for k in weights})
    tf = pd.DataFrame({k: sl[k]["tax"].reindex(ix).fillna(0.0) for k in weights})
    month_end = _month_ends(ix)
    tgt = pd.Series(weights, dtype=float)
    cost_rate = pd.Series({k: COSTS[k] for k in weights}, dtype=float)
    entry = cost_rate + pd.Series({k: ENTRY_EXTRA.get(k, 0.0) for k in weights})
    if "S" in weights and float(sl["S"]["sig"].reindex(ix).iloc[0]) == 0.0:
        entry["S"] = 0.001  # 그날 S가 KODEX200 전환 상태면 ETF 비용
    w, eq = tgt.copy(), capital
    c0 = float((tgt * entry).sum()) * eq
    eq -= c0
    eqs, costs, taxes, ws = [eq], [c0], [0.0], [w.copy()]
    for d in ix[1:]:
        r = df.loc[d]
        c_krw = float((eq * w * cf.loc[d]).sum())
        t_krw = float((eq * w * tf.loc[d]).sum())
        g = float((w * (1 + r)).sum())
        w, eq = w * (1 + r) / g, eq * g
        if d in month_end:
            rc = float(((w - tgt).abs() * cost_rate).sum())
            c_krw += eq * rc
            eq *= 1 - rc
            w = tgt.copy()
        eqs.append(eq), costs.append(c_krw), taxes.append(t_krw), ws.append(w.copy())
    return {"eq": pd.Series(eqs, index=ix), "cost_krw": pd.Series(costs, index=ix), "tax_krw": pd.Series(taxes, index=ix),
            "w": pd.DataFrame(ws, index=ix)}


def with_warmup(fresh: pd.Series, full: pd.Series) -> pd.Series:
    """변동성 목표의 60일 워밍업용: START 이전은 전체 기간 곡선(START 값에 맞춰 스케일), 이후는 실측 곡선."""
    head = full.loc[: fresh.index[0]].iloc[:-1]
    head = head / float(full.loc[fresh.index[0]]) * float(fresh.iloc[0])
    return pd.concat([head, fresh])


def overlay(base: dict, kind: str, safe: pd.Series | None = None, full: pd.Series | None = None, cost: float = 0.01) -> dict:
    """조합 곡선 위 덧씌우기. kind='vol': engine.vol_target(20%, 60일, 주 1회, 비용 1%), 'cppi': combo.cppi(80%, 4배, 주 1회)."""
    eq = base["eq"]
    cost = 0.01 if kind == "vol" else cost  # vol_target은 비용 1% 고정(engine 기본값)
    if kind == "vol":
        v = vol_target(with_warmup(eq, full), "sim2020", 0.20, log=False)
        curve, w = v["curve"].loc[eq.index[0]:], v["weight"].loc[eq.index[0]:]
        cash = pd.Series(0.0, index=eq.index)
    else:
        curve, w = cppi(eq / float(eq.iloc[0]), safe, cost=cost, detail=True)
        cash = safe.reindex(eq.index).ffill().pct_change().fillna(0.0)
    out = curve / float(curve.iloc[0]) * float(eq.iloc[0])
    dw_cost = (w.diff().abs().fillna(0.0) * cost * out.shift(1).fillna(out.iloc[0]))
    return {"eq": out, "cost_krw": base["cost_krw"] * w + dw_cost, "tax_krw": base["tax_krw"] * w,
            "w": base["w"].mul(w, axis=0), "expo": w, "cash_ret": cash}


# ---------- 평가 ----------
def summarize(eq: pd.Series, capital: float = CAPITAL) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    rec = after[after >= float(eq.loc[peak])]
    m = eq.resample("ME").last()
    mr = (m / m.shift(1).fillna(capital) - 1)
    return {"final_krw": round(float(eq.iloc[-1])), "cagr": round((float(eq.iloc[-1]) / capital) ** (1 / years) - 1, 4),
            "mdd": round(float(dd.min()), 4), "mdd_peak": str(peak.date()), "mdd_trough": str(trough.date()),
            "mdd_recovery": str(rec.index[0].date()) if len(rec) else "미회복",
            "worst_month": round(float(mr.min()), 4), "worst_month_date": str(mr.idxmin().date()),
            "pos_months": round(float((mr > 0).mean()), 3), "n_months": int(len(mr))}


def yearly(eq: pd.Series, capital: float = CAPITAL) -> dict:
    y = eq.resample("YE").last()
    return {str(k.year): round(float(v), 4) for k, v in (y / y.shift(1).fillna(capital) - 1).items()}


def stress(eq: pd.Series) -> dict:
    out = {}
    for label, a, b in STRESS:
        before = eq.loc[:pd.Timestamp(a) - pd.Timedelta(days=1)]
        base = float(before.iloc[-1]) if len(before) else float(eq.iloc[0])
        out[label] = round(float(eq.loc[:b].iloc[-1]) / base - 1, 4)
    return out


# ---------- 매매 건수 ----------
def trade_counts(sl: dict, sig: pd.DataFrame) -> pd.DataFrame:
    """연도별 슬리브 이벤트 수. S_names: 분기 리밸런싱에서 바뀐 종목 수(매도+매수), S_rotate: 전환 횟수,
    DAA_changes: 배분이 바뀐 월말 수, PP_rebal: 월말 재조정 수, NQ/L3_switches: 추세 전환 수."""
    rows = {}
    picks = {pd.Timestamp(k): set(v) for k, v in sl["S"]["picks"].items()}
    prev = None
    for d in sorted(picks):
        if d >= pd.Timestamp(START):
            y = d.year
            rows.setdefault(y, {}).setdefault("S_names", 0)
            rows[y]["S_names"] += len(picks[d] - prev) + len(prev - picks[d]) if prev is not None else len(picks[d])
            rows[y]["S_rebal"] = rows[y].get("S_rebal", 0) + 1
        prev = picks[d]
    for col, s in (("S_rotate", sig["s_rot"]), ("NQ_switches", sig["nq_on"].astype(float))):
        sw = s.loc[START:].diff().abs().fillna(0.0)
        for y, v in sw.groupby(sw.index.year).sum().items():
            rows.setdefault(int(y), {})[col] = int(v)
    l3 = sl["L3"]["on"].astype(float).loc[START:END]
    for y, v in l3.diff().abs().fillna(0.0).groupby(l3.index.year).sum().items():
        rows.setdefault(int(y), {})["L3_switches"] = int(v)
    for key, col in (("DAA", "DAA_changes"), ("PP", "PP_rebal")):
        prev = None
        for d, a in sorted(sl[key]["alloc"].items()):
            if d >= pd.Timestamp(START) and (key == "PP" or a != prev):
                rows.setdefault(d.year, {})[col] = rows.setdefault(d.year, {}).get(col, 0) + 1
            prev = a
    return pd.DataFrame(rows).T.sort_index().fillna(0).astype(int)


def trades_per_year(tc: pd.DataFrame, weights: dict) -> float:
    cols = {"S": ["S_names", "S_rotate"], "DAA": ["DAA_changes"], "PP": ["PP_rebal"], "LETF": ["NQ_switches"], "L3": ["L3_switches"]}
    use = [c for k in weights for c in cols.get(k, []) if c in tc.columns]
    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    return round(float(tc[use].sum().sum()) / years, 1) if use else 0.0


# ---------- B 이벤트 로그 ----------
def events_b(sl: dict, sig: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    rows = []
    e = kr_etf()["133690"]
    on = sig["nq_on"].loc[START:]
    for d in on.index[on.astype(int).diff().fillna(0.0) != 0]:
        t = on.index[on.index.get_loc(d) - 1]  # 신호일(전날 종가 판단·매매)
        rows.append({"date": t.date(), "sleeve": "LETF", "event": "in" if on[d] else "out", "detail": f"133690 종가 {e[t]:.0f}"})
    for d, a in sorted(sl["DAA"]["alloc"].items()):
        if d >= pd.Timestamp(START):
            rows.append({"date": d.date(), "sleeve": "DAA", "event": "alloc",
                         "detail": "; ".join(f"{k}({alloc.KR_MAP[k]}) {v:.0%}" for k, v in a.items())})
    s_rot = sig["s_rot"].loc[START:]
    for d in s_rot.index[s_rot.diff().fillna(0.0) != 0]:
        rows.append({"date": d.date(), "sleeve": "S", "event": "rotate→S" if s_rot[d] == 1 else "rotate→KODEX200", "detail": ""})
    prev: set = set()
    for k, v in sorted(sl["S"]["picks"].items()):
        d = pd.Timestamp(k)
        if d >= pd.Timestamp(START):
            turn = float(sl["S"]["turnover"].get(d, 0.0))
            rows.append({"date": d.date(), "sleeve": "S", "event": "rebalance",
                         "detail": f"{len(v)}종목, 회전 {turn:.0%}, 교체 {len(set(v) - prev)}: "
                                   + ", ".join(f"{c} {names.get(c, '')}" for c in v)})
        prev = set(v)
    return pd.DataFrame(rows).sort_values(["date", "sleeve"]).reset_index(drop=True)


# ---------- 그림 ----------
def charts(eqs: pd.DataFrame, yr: pd.DataFrame, res_b: dict, sl: dict, sig: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    kw = dict(figsize=(16, 9), dpi=100)
    fig, ax = plt.subplots(**kw)
    for c in eqs.columns:
        ax.plot(eqs.index, eqs[c] / 1e8, lw=2.2 if c.startswith("B") else 1.2, label=c)
    ax.set_yscale("log"), ax.set_title("가상 실측 2020-01-02~ 자산곡선(억 원, 로그)"), ax.legend(fontsize=9), ax.grid(alpha=.3)
    ax.set_yticks([0.7, 1, 1.5, 2, 3, 4, 5, 6, 7]), ax.set_yticklabels([f"{v:g}억" for v in (0.7, 1, 1.5, 2, 3, 4, 5, 6, 7)])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.savefig(OUT / "1_equity.png"), plt.close(fig)
    fig, ax = plt.subplots(**kw)
    for c in eqs.columns:
        ax.plot(eqs.index, (eqs[c] / eqs[c].cummax() - 1) * 100, lw=2.2 if c.startswith("B") else 1.0, label=c)
    ax.set_title("낙폭(%)"), ax.legend(fontsize=9), ax.grid(alpha=.3)
    fig.savefig(OUT / "2_drawdown.png"), plt.close(fig)
    fig, ax = plt.subplots(**kw)
    (yr * 100).T.plot.bar(ax=ax, width=.85)
    ax.set_title("연도별 수익률(%)"), ax.legend(fontsize=8, ncol=3), ax.grid(alpha=.3, axis="y"), ax.axhline(0, color="k", lw=.8)
    fig.savefig(OUT / "3_yearly.png"), plt.close(fig)
    # B 배분: S(주식/KODEX200), DAA(위험/채권), 133690/단기채
    w = res_b["w"]
    s_rot, nq = sig["s_rot"].reindex(w.index).ffill(), sig["nq_on"].reindex(w.index).ffill().astype(float)
    daa_safe = pd.Series({d: sum(v for k, v in a.items() if k in ("SHY", "IEF", "LQD")) for d, a in sl["DAA"]["alloc"].items()})
    daa_safe = daa_safe.reindex(w.index, method="ffill").fillna(0.0)
    parts = pd.DataFrame({"S 소형가치주": w["S"] * s_rot, "S→KODEX200 전환": w["S"] * (1 - s_rot),
                          "DAA 위험자산": w["DAA"] * (1 - daa_safe), "DAA 채권(SHY/IEF/LQD)": w["DAA"] * daa_safe,
                          "나스닥100(133690)": w["LETF"] * nq, "단기채(추세 이탈)": w["LETF"] * (1 - nq)})
    fig, ax = plt.subplots(**kw)
    ax.stackplot(parts.index, (parts.T * 100).to_numpy(), labels=parts.columns, alpha=.85)
    ax.set_ylim(0, 100), ax.set_title("시나리오 B 배분 추이(%)"), ax.legend(loc="lower left", fontsize=9), ax.grid(alpha=.3)
    fig.savefig(OUT / "4_B_allocation.png"), plt.close(fig)
    px = kr_etf()["133690"].loc["2019-07-01":END]
    sma = kr_etf()["133690"].rolling(200).mean().loc[px.index]
    on = sig["nq_on"].loc[START:]
    fig, ax = plt.subplots(**kw)
    ax.plot(px.index, px, label="133690 종가", lw=1.2), ax.plot(sma.index, sma, label="200일선", lw=1.2)
    sw = on.astype(int).diff().fillna(0.0)
    ins, outs = sw.index[sw > 0], sw.index[sw < 0]
    ax.scatter(ins, px.reindex(ins), marker="^", color="g", s=90, zorder=5, label=f"매수({len(ins)})")
    ax.scatter(outs, px.reindex(outs), marker="v", color="r", s=90, zorder=5, label=f"매도→단기채({len(outs)})")
    ax.set_title("시나리오 B: 나스닥100(133690) 추세 전환"), ax.legend(), ax.grid(alpha=.3)
    fig.savefig(OUT / "5_B_trend_switches.png"), plt.close(fig)


# ---------- 교차검증 ----------
def crosscheck(eqs: pd.DataFrame) -> pd.DataFrame:
    r8 = pd.read_parquet(HERE / "round8_curves.parquet")
    fin = pd.read_parquet(HERE / "final_curves.parquet")
    old = {"A 최고 S40+DAA30+QQQx3추세30": r8[[c for c in r8.columns if c.startswith("C2b")][0]],
           "B 추천 S40+DAA30+나스닥추세30": r8[[c for c in r8.columns if c.startswith("C2 ")][0]],
           "C B+변동성목표20": r8["C4 C2 + vol20"],
           "D 기존최선 S40+DAA30+PP30": fin["S40+DAA30+PP30"],
           "E S30+DAA30+PP20+나스닥추세20": r8[[c for c in r8.columns if c.startswith("C3 ")][0]],
           "G 나스닥추세 단독(세후)": r8[[c for c in r8.columns if c.startswith("T1 나스닥100(133690) +SMA200 (세후")][0]],
           "H D+CPPI": fin["S40+DAA30+PP30 +CPPI"]}
    rows = []
    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    for k, s in old.items():
        s = s.dropna().loc[START:END]
        s = s / float(s.iloc[0])
        cagr_old = float(s.iloc[-1]) ** (1 / years) - 1
        cagr_new = (float(eqs[k].iloc[-1]) / CAPITAL) ** (1 / years) - 1
        rows.append({"scenario": k, "rebased_cagr": round(cagr_old, 4), "fresh_cagr": round(cagr_new, 4),
                     "diff_pp": round((cagr_new - cagr_old) * 100, 2), "rebased_final_x": round(float(s.iloc[-1]), 3),
                     "fresh_final_x": round(float(eqs[k].iloc[-1]) / CAPITAL, 3)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sl = {"S": sleeve_s()}
    kr_idx = pd.DatetimeIndex(sl["S"]["ret"].index)
    sl.update(DAA=sleeve_alloc("DAA"), PP=sleeve_alloc("PP"), LETF=sleeve_letf(), L3=sleeve_l3(kr_idx),
              KOSPI=sleeve_hold("069500"), NQ=sleeve_hold("133690"))
    e = kr_etf()
    sig = signals(e["133690"], sl["S"]["curve"], e["069500"], alloc.us_prices(), kr_idx)
    assert (sig["s_rot"] == sl["S"]["sig"].reindex(kr_idx)).all()
    for d, a in sl["DAA"]["alloc"].items():  # 신호 함수와 백테스트 배분 일치
        if d in sig.index and sig.at[d, "daa"]:
            assert json.loads(sig.at[d, "daa"]) == a, d
    safe = e["153130"]
    res: dict[str, dict] = {}
    for name, spec in SCENARIOS.items():
        if isinstance(spec, dict):
            res[name] = portfolio(sl, spec)
            res[name]["weights"] = spec
        else:
            kind, base = spec.split(":", 1)
            full = portfolio(sl, res[base]["weights"], start=WARM)["eq"] if kind == "vol" else None
            res[name] = overlay(res[base], kind, safe=safe, full=full, cost=blended_cost(res[base]["weights"], COSTS))
            res[name]["weights"] = res[base]["weights"]
        print(name, f"최종 {res[name]['eq'].iloc[-1]:,.0f}원", flush=True)
    eqs = pd.DataFrame({k: v["eq"] for k, v in res.items()})
    eqs.to_parquet(OUT / "equity.parquet")
    tc = trade_counts(sl, sig)
    tc.to_csv(OUT / "trades.csv", encoding="utf-8")
    summ = pd.DataFrame([{"scenario": k, **summarize(v["eq"]), "costs_krw": round(float(v["cost_krw"].sum())),
                          "taxes_krw": round(float(v["tax_krw"].sum())), "trades_per_year": trades_per_year(tc, v["weights"]),
                          "avg_exposure": round(float(v["expo"].mean()), 3) if "expo" in v else 1.0} for k, v in res.items()])
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8")
    yr = pd.DataFrame({k: yearly(v["eq"]) for k, v in res.items()}).T
    yr.to_csv(OUT / "yearly.csv", encoding="utf-8")
    st = pd.DataFrame({k: stress(v["eq"]) for k, v in res.items()}).T
    st.to_csv(OUT / "stress.csv", encoding="utf-8")
    cc = crosscheck(eqs)
    cc.to_csv(OUT / "crosscheck.csv", index=False, encoding="utf-8")
    from research import data
    names = data.prices().groupby("Code")["Name"].last()
    events_b(sl, sig, names).to_csv(OUT / "events_B.csv", index=False, encoding="utf-8-sig")
    charts(eqs, yr, res["B 추천 S40+DAA30+나스닥추세30"], sl, sig)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False)), print(yr.to_string()), print(st.to_string()), print(cc.to_string(index=False)), print(tc.to_string())


if __name__ == "__main__":
    main()
