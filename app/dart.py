"""DART OpenAPI 조회. 키는 config(.env)에서만 읽는다."""
from datetime import date, timedelta
from functools import lru_cache

import opendartreader
import pandas as pd
import requests

from app import config

API = "https://opendart.fss.or.kr/api/"
ACCOUNTS = {"당기순이익(손실)": "net", "당기순이익": "net", "자본총계": "equity",
            "매출액": "sales", "영업이익": "op", "자산총계": "assets"}


def _get(path: str, **params) -> list[dict]:
    r = requests.get(API + path, params={"crtfc_key": config.DART_API_KEY, **params}, timeout=30).json()
    if r["status"] == "013":  # 조회된 데이터 없음
        return []
    if r["status"] != "000":
        raise RuntimeError(f"DART {r['status']}: {r['message']}")
    return r["list"]


@lru_cache(maxsize=1)
def corp_codes() -> pd.Series:
    """종목코드 → DART 고유번호."""
    cc = pd.DataFrame(opendartreader.OpenDartReader(config.DART_API_KEY).corp_codes)
    cc = pd.DataFrame(cc[cc["stock_code"].astype(str).str.strip() != ""])
    return pd.Series(cc["corp_code"].tolist(), index=cc["stock_code"].tolist())


def annual_accounts(stock_codes: list[str], year: int, reprt_code: str = "11011") -> pd.DataFrame:
    """사업보고서 순이익·자본총계. 연결(CFS) 우선, 없으면 별도(OFS). filed = 접수일(이날부터 알 수 있던 값)."""
    cmap = corp_codes()
    codes = [str(cmap[c]) for c in stock_codes if c in cmap.index]
    rows = []
    for i in range(0, len(codes), 100):  # 한 번에 100개까지
        rows += _get("fnlttMultiAcnt.json", corp_code=",".join(codes[i:i + 100]),
                     bsns_year=str(year), reprt_code=reprt_code)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"net": [], "equity": [], "filed": []})
    df = pd.DataFrame(df[df["account_nm"].isin(list(ACCOUNTS))])
    df["key"] = [ACCOUNTS[a] for a in df["account_nm"]]
    df["amt"] = pd.to_numeric(df["thstrm_amount"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.sort_values(by="fs_div").drop_duplicates(["stock_code", "key"])  # "CFS" < "OFS"
    out = df.pivot(index="stock_code", columns="key", values="amt")
    out["filed"] = df.groupby("stock_code")["rcept_no"].first().str[:8]
    return out


def recent_filings(stock_code: str, days: int = 7) -> list[dict]:
    cmap = corp_codes()
    if stock_code not in cmap.index:
        return []
    start = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
    return [{"date": x["rcept_dt"], "title": x["report_nm"].strip(), "filer": x["flr_nm"],
             "link": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + x["rcept_no"]}
            for x in _get("list.json", corp_code=cmap[stock_code], bgn_de=start, page_count=20)]
