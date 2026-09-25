import pandas as pd
import pytest

from research import data
from research.factors import pead_features


def _acc(rows: dict) -> pd.DataFrame:
    """rows: stock_code -> (assets, op, filed_str)"""
    df = pd.DataFrame({k: {"assets": v[0], "op": v[1], "equity": 1.0, "net": 1.0, "sales": 1.0,
                            "filed": pd.Timestamp(v[2])} for k, v in rows.items()}).T
    df.index.name = "stock_code"
    df["filed"] = pd.to_datetime(df["filed"])
    return df


@pytest.fixture
def fake_reports(monkeypatch):
    # 종목 A: 2019 Q1, 2020 Q1(전년동기 대비 op 변화) — 손으로 계산 가능한 값
    tables = {
        (2019, "Q1"): _acc({"A": (1000.0, 100.0, "2019-05-10")}),
        (2020, "Q1"): _acc({"A": (1000.0, 150.0, "2020-05-10"),
                             "B": (1000.0, 100.0, "2020-05-10")}),  # B: 전년 보고서 없음
    }

    def fake_quarterly(year, q):
        return tables.get((year, q), pd.DataFrame(columns=["assets", "op", "equity", "net", "sales", "filed"]))

    monkeypatch.setattr(data, "quarterly_accounts", fake_quarterly)
    return tables


def test_sur_op_hand_computed(fake_reports):
    # d 이후 시점: 두 보고서 모두 filed < d → sur_op = (150-100)/1000 = 0.05
    f = pead_features(pd.Timestamp("2020-06-01"), pd.Index(["A", "B"]))
    assert f.loc["A", "sur_op"] == pytest.approx(0.05)
    assert pd.isna(f.loc["B", "sur_op"])  # 전년동기 보고서 없음


def test_strict_filed_before_d(fake_reports):
    # d가 2020 Q1 접수일(5/10) 당일이면 그 보고서는 아직 못 쓴다(strict <) → 2019 Q1만 남고 전년비교 불가
    f = pead_features(pd.Timestamp("2020-05-10"), pd.Index(["A"]))
    assert pd.isna(f.loc["A", "sur_op"])


def test_staleness_over_120_days(fake_reports):
    # d가 접수일로부터 120일 넘게 지나면 신호 폐기
    f = pead_features(pd.Timestamp("2020-09-10"), pd.Index(["A"]))  # filed 2020-05-10, 123일 경과
    assert pd.isna(f.loc["A", "sur_op"])
    assert f.loc["A", "filed_age"] > 120

    f2 = pead_features(pd.Timestamp("2020-09-06"), pd.Index(["A"]))  # 119일 경과, 신호 유효
    assert f2.loc["A", "sur_op"] == pytest.approx(0.05)
