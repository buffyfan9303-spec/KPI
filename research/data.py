"""백테스트용 시점 정합(point-in-time) 데이터.

- 시세·시가총액: FinanceData/marcap (상장폐지 종목 포함, 1995~). ~/alpha-data/marcap/*.parquet
- 재무: DART fnlttMultiAcnt 사업보고서(2015 사업연도~). 접수일(filed) 이후에만 쓸 수 있다.
- 수익률: KRX 등락률(ChangesRatio). 액면분할 등은 기준가 조정으로 반영, 배당은 빠져 있다(보수적).
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app import dart

DATA = Path.home() / "alpha-data"
MARCAP = DATA / "marcap"
COLS = ["Date", "Code", "Name", "Market", "Open", "High", "Low", "Close", "ChangesRatio", "Volume", "Amount", "Marcap"]


@lru_cache(maxsize=1)
def prices(start_year: int = 2015) -> pd.DataFrame:
    """코스피·코스닥 일별 패널(코넥스 제외). 인덱스 없음, Date·Code 열."""
    files = sorted(f for f in MARCAP.glob("marcap-*.parquet") if int(f.stem[-4:]) >= start_year)
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])]
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.DataFrame(df).sort_values(by=["Date", "Code"]).reset_index(drop=True)


def returns_wide(px: pd.DataFrame) -> pd.DataFrame:
    """날짜×종목 일간 수익률(소수). 거래가 없는 날(정지·상장 전후)은 NaN."""
    r = px.pivot_table(index="Date", columns="Code", values="ChangesRatio", aggfunc="last") / 100.0
    return r.sort_index()


def annual_accounts(year: int) -> pd.DataFrame:
    """year 사업연도 사업보고서의 순이익·자본총계·접수일. 파일로 캐시(DART 호출 절약)."""
    f = DATA / "dart" / f"accounts2-{year}.parquet"  # 2: 매출·영업이익·자산 포함
    if f.exists():
        return pd.read_parquet(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    codes = sorted(set(prices()["Code"]))  # 상장폐지 종목 포함
    acc = dart.annual_accounts(codes, year)
    acc["filed"] = pd.to_datetime(acc["filed"], format="%Y%m%d", errors="coerce")
    acc.to_parquet(f)
    return acc


QUARTERS = {"Q1": "11013", "H1": "11012", "Q3": "11014", "FY": "11011"}


def quarterly_accounts(year: int, q: str) -> pd.DataFrame:
    """분기·반기 보고서(당기 금액). 같은 종류끼리 전년 대비 비교하면 3개월/누적 여부와 무관하게 일관된다."""
    if q == "FY":
        return annual_accounts(year)
    f = DATA / "dart" / f"q-{year}-{q}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    acc = dart.annual_accounts(sorted(set(prices()["Code"])), year, QUARTERS[q])
    acc["filed"] = pd.to_datetime(acc["filed"], format="%Y%m%d", errors="coerce")
    acc.to_parquet(f)
    return acc
