"""KIS Open API 최소 연결. 공식 저장소 examples_llm/kis_auth.py 기준(2026-08-26 커밋)."""
import requests

DOMAIN = {"mock": "https://openapivts.koreainvestment.com:29443",
          "real": "https://openapi.koreainvestment.com:9443"}


def verify(app_key: str, app_secret: str, mode: str = "mock") -> tuple[bool, str]:
    """토큰 발급으로 키가 맞는지만 확인한다. 토큰은 저장하지 않는다.
    KIS는 토큰 발급을 1분에 1번으로 제한하므로 연달아 누르면 실패할 수 있다."""
    try:
        r = requests.post(DOMAIN[mode] + "/oauth2/tokenP", timeout=10,
                          json={"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret})
    except requests.RequestException as e:
        return False, f"KIS 서버에 연결하지 못했어요 ({type(e).__name__})"
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.ok and body.get("access_token"):
        return True, "모의투자 키가 확인됐어요"
    return False, str(body.get("error_description") or body.get("msg1") or f"HTTP {r.status_code}")
