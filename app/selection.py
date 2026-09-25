"""1단계 A: 소형 가치주 후보 (PDF 5쪽). 시가총액 하위 20% 중 PER·PBR이 가장 낮은 종목."""
import pandas as pd


def value_screen(listing: pd.DataFrame, accounts: pd.DataFrame,
                 small_q: float = 0.2, top: int = 20) -> pd.DataFrame:
    """listing: index=종목코드, Marcap·Name 필요. accounts: index=종목코드, net·equity.

    - 시총 하위 small_q는 적자 기업 포함 전체 상장 보통주 기준으로 자른다.
    - PER·PBR이 양수인 종목만 남기고, 두 순위의 합이 작은 순으로 top개.
    """
    # ponytail: 현재 시가총액 ÷ 최근 사업보고서. 과거 시점 백테스트는 접수일(filed) 기준 시총으로 다시 계산해야 함
    small = listing[listing["Marcap"] <= listing["Marcap"].quantile(small_q)]
    df = small.join(accounts, how="inner")
    df = df.assign(per=df["Marcap"] / df["net"], pbr=df["Marcap"] / df["equity"])
    df = df[(df["per"] > 0) & (df["pbr"] > 0)]
    df = df.assign(score=pd.Series(df["per"]).rank() + pd.Series(df["pbr"]).rank())
    return pd.DataFrame(df.nsmallest(top, "score"))
