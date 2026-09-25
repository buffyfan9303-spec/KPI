import hmac
import html
import re
import threading
import time
from datetime import date, timedelta
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from pydantic import BaseModel, Field
from fastapi import Path as Path_
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app import accounts, backtest, config, dart, kis, news_gate, selection
from app.timing import hysteresis_position

STATIC = Path(__file__).parent / "static"
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
       "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; font-src 'self' https://cdn.jsdelivr.net data:; "
       "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


@app.middleware("http")
async def _guard(request: Request, call_next):
    """다른 사이트에서 보낸 POST·PUT·DELETE 차단(로그인 CSRF 포함) + 보안 헤더."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            return JSONResponse({"detail": "다른 사이트에서 온 요청은 막았어요"}, status_code=403)
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET,
                   https_only=config.COOKIE_SECURE, same_site="strict", max_age=12 * 3600)

accounts.init()

# ponytail: 실패 횟수를 메모리에 둔다(서버 재시작 시 초기화). 사용자가 많아지면 DB·Redis로
_fails: dict[str, tuple[int, float]] = {}
_fails_lock = threading.Lock()
MAX_FAILS, IP_MAX_FAILS, LOCK_SEC = 5, 20, 300


def _hit(key: str, limit: int, now: float) -> bool:
    """실패 1회 기록. 한도에 닿으면 LOCK_SEC 동안 잠근다. 잠겨 있으면 True."""
    n, until = _fails.get(key, (0, 0.0))
    if now < until:
        return True
    n += 1
    _fails[key] = (0, now + LOCK_SEC) if n >= limit else (n, 0.0)
    return n >= limit


def _locked(key: str, now: float) -> bool:
    return now < _fails.get(key, (0, 0.0))[1]


def _session_user(request: Request):
    uid = request.session.get("uid")
    u = accounts.get_user(uid) if isinstance(uid, int) else None
    if u is None or not u["active"] or request.session.get("tag") != accounts.session_tag(u):
        return None
    return u


@app.get("/")
def index(request: Request):
    return FileResponse(STATIC / ("index.html" if _session_user(request) else "login.html"))


@app.post("/login")
def login(request: Request, username: str = Form(..., max_length=40), password: str = Form(..., max_length=200)):
    name, now = username.strip().lower(), time.time()
    ip = request.client.host if request.client else "?"
    user_key, ip_key = f"u:{name}@{ip}", f"ip:{ip}"
    with _fails_lock:
        if len(_fails) > 5000:  # 만료된 기록 정리
            for k in [k for k, (n, until) in _fails.items() if n == 0 and until < now]:
                del _fails[k]
        if _locked(user_key, now) or _locked(ip_key, now):
            return RedirectResponse("/?e=lock", status_code=303)
        if not accounts.USERNAME.match(name):  # 형식이 틀린 아이디는 해시 계산 없이 실패 처리
            _hit(ip_key, IP_MAX_FAILS, now)
            return RedirectResponse("/?e=1", status_code=303)
        # 먼저 실패로 기록해 두고 성공하면 지운다 → 동시에 여러 번 보내도 한도를 못 넘는다
        _hit(user_key, MAX_FAILS, now)
        _hit(ip_key, IP_MAX_FAILS, now)
    u = accounts.authenticate(name, password)
    if u is None:
        return RedirectResponse("/?e=1", status_code=303)
    with _fails_lock:
        _fails.pop(user_key, None)
    request.session.clear()
    request.session.update(uid=u["id"], tag=accounts.session_tag(u))
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
def logout(request: Request):
    u = _session_user(request)
    if u is not None:
        accounts.end_sessions(u["id"])
    request.session.clear()
    return RedirectResponse("/", status_code=303)


def _me(request: Request):
    """로그인만 확인. 비밀번호 변경·동의 API용."""
    u = _session_user(request)
    if u is None:
        raise HTTPException(401)
    return u


def _need_auth(request: Request):
    """데이터 API용: 첫 비밀번호 변경과 이용 동의까지 끝난 사용자만."""
    u = _me(request)
    if u["must_change"]:
        raise HTTPException(403, "change_password")
    if u["consent_at"] is None:
        raise HTTPException(403, "consent")
    return u


def _admin(request: Request):
    u = _need_auth(request)
    if not u["is_admin"]:
        raise HTTPException(403, "관리자만 쓸 수 있어요")
    return u


def _invalid(fn, *a):
    try:
        return fn(*a)
    except accounts.Invalid as e:
        raise HTTPException(400, str(e))


# ponytail: 프로세스 메모리 캐시. 서버 여러 대로 늘리면 Redis로 이동
_cache: dict = {}


CACHE_MAX = 2000


def _cached(key, ttl: int, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    if len(_cache) >= CACHE_MAX:  # 가장 오래된 절반을 버린다
        for k in sorted(_cache, key=lambda k: _cache[k][0])[:CACHE_MAX // 2]:
            del _cache[k]
    _cache[key] = (time.time(), val)
    return val


def _listing() -> pd.DataFrame:
    return _cached("listing", 60, lambda: fdr.StockListing("KRX").set_index("Code"))


def _names() -> pd.DataFrame:
    """검색·이름표용. 주식 목록엔 ETF가 없어서 ETF 목록을 합친다."""
    def load():
        stocks = _listing()[["Name", "Market"]]
        etf = fdr.StockListing("ETF/KR").set_index("Symbol")[["Name"]].assign(Market="ETF")
        return pd.DataFrame(pd.concat([stocks, etf[~etf.index.isin(stocks.index)]]))
    return _cached("names", 3600, load)


def _spark(sym: str, days: int) -> dict:
    start = (date.today() - timedelta(days=int(days * 1.6) + 10)).isoformat()
    close = _cached(("px", sym, days), 300, lambda: fdr.DataReader(sym, start)["Close"].dropna().tail(days))
    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
    return {"dates": pd.DatetimeIndex(close.index).strftime("%Y-%m-%d").tolist(),
            "closes": [round(float(v), 2) for v in close],
            "last": last, "chg": last - prev, "pct": round((last / prev - 1) * 100, 2)}


INDICES = {"KS11": "코스피", "KQ11": "코스닥", "VIX": "VIX 변동성", "USD/KRW": "원/달러"}
CODE = r"^[0-9A-Z]{6}$"  # 영문이 섞인 신규 종목코드(예: 0161M0)도 있음


@app.get("/api/market", dependencies=[Depends(_need_auth)])
def api_market():
    df = _listing()
    df = pd.DataFrame(df[(df["Volume"] > 0) & (df["Marcap"] >= 5e10)])  # 시총 500억 미만 잡주 제외

    def rows(part):
        return [{"code": c, "name": r.Name, "market": r.Market, "close": int(r.Close),
                 "chg": int(r.Changes), "pct": float(r.ChagesRatio)} for c, r in part.iterrows()]
    by = df.sort_values("ChagesRatio")
    return {"gainers": rows(by.tail(6).iloc[::-1]), "losers": rows(by.head(6)),
            "indices": [{"sym": s, "name": n, **_spark(s, 60)} for s, n in INDICES.items()]}


@app.get("/api/watchlist")
def api_watchlist(u=Depends(_need_auth)):
    names = _names()["Name"]
    return [{"code": c, "name": names.get(c, c), **_spark(c, 30)} for c in accounts.get_watch(u["id"])]


@app.post("/api/watchlist/{code}")
def watch_add(code: str = Path_(pattern=CODE), u=Depends(_need_auth)):
    _invalid(accounts.add_watch, u["id"], code)
    return {"ok": True}


@app.delete("/api/watchlist/{code}")
def watch_del(code: str = Path_(pattern=CODE), u=Depends(_need_auth)):
    accounts.remove_watch(u["id"], code)
    return {"ok": True}


@app.get("/api/search", dependencies=[Depends(_need_auth)])
def api_search(q: str = Query(..., min_length=1, max_length=20)):
    df = _names()
    hit = df[df.index.str.startswith(q) | df["Name"].str.contains(q, case=False, regex=False)]
    return [{"code": c, "name": r.Name, "market": r.Market} for c, r in hit.head(8).iterrows()]


@app.get("/api/chart", dependencies=[Depends(_need_auth)])
def api_chart(symbol: str = Query(..., pattern=CODE), days: int = Query(120, ge=20, le=750)):
    days = next(b for b in (22, 66, 120, 250, 750) if b >= days)  # 캐시 키가 무한히 늘지 않게 구간으로 맞춤
    names = _names()["Name"]
    return {"code": symbol, "name": names.get(symbol, symbol), **_spark(symbol, days)}


def _google_news(query: str, n: int) -> list[dict]:
    q = urllib.parse.urlencode({"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"})
    xml = urllib.request.urlopen("https://news.google.com/rss/search?" + q, timeout=8).read()
    out = []
    for it in list(ET.fromstring(xml).iter("item"))[:n]:
        link = it.findtext("link") or ""
        out.append({"title": it.findtext("title") or "", "source": it.findtext("source") or "",
                    "date": it.findtext("pubDate"), "link": link if link.startswith("https://") else ""})
    return out


def _naver_news(query: str, n: int) -> list[dict]:
    # 2026-07-31부터 개발자센터 신규 발급이 끝나 NAVER API HUB(네이버 클라우드) 키를 쓴다. 주소·헤더가 다르다.
    r = requests.get("https://naverapihub.apigw.ntruss.com/search/v1/news", timeout=8,
                     params={"query": query, "display": n, "sort": "date"},
                     headers={"X-NCP-APIGW-API-KEY-ID": config.NAVER_CLIENT_ID,
                              "X-NCP-APIGW-API-KEY": config.NAVER_CLIENT_SECRET})
    r.raise_for_status()
    strip = lambda t: html.unescape(re.sub(r"<[^>]+>", "", t))  # noqa: E731
    return [{"title": strip(i["title"]), "source": "네이버 뉴스", "date": i["pubDate"],
             "link": next((u for u in (i["originallink"], i["link"]) if u.startswith(("https://", "http://"))), "")}
            for i in r.json()["items"]]


def _news(query: str, n: int = 8) -> dict:
    """네이버 우선, 실패하면 Google RSS. 실패 사유는 warning으로 화면에 알린다."""
    warning = None
    if config.NAVER_CLIENT_ID:
        try:
            return {"provider": "네이버", "warning": None, "items": _naver_news(query, n)}
        except requests.RequestException as e:
            code = getattr(e.response, "status_code", None)
            warning = "네이버 인증 실패(키 확인 필요)" if code == 401 else f"네이버 오류: {type(e).__name__}"
    return {"provider": "Google", "warning": warning, "items": _google_news(query, n)}


def _with_sentiment(items: list[dict]) -> list[dict]:
    scored = news_gate.classify([it["title"] for it in items])
    return [{**it, **sc, "critical": news_gate.critical({**it, **sc})} for it, sc in zip(items, scored)]


@app.get("/api/news", dependencies=[Depends(_need_auth)])
def api_news():
    def load():
        d = _news("코스피 OR 코스닥 증시")
        return {**d, "items": _with_sentiment(d["items"])}
    try:
        return _cached("news", 300, load)
    except (OSError, ET.ParseError):
        raise HTTPException(502, "뉴스를 불러오지 못했어요")


def _need_dart():
    if not config.DART_API_KEY:
        raise HTTPException(503, "DART_API_KEY가 설정되지 않았어요")


@app.get("/api/gate", dependencies=[Depends(_need_auth), Depends(_need_dart)])
def api_gate(u=Depends(_need_auth)):
    """관심종목별 4단계 뉴스 게이트: 최근 7일 DART 공시 + 종목 뉴스 제목을 FinBERT로 판정."""
    names = _names()["Name"]

    def one(code: str) -> dict:
        name = str(names.get(code, code))
        filings = [{**f, "kind": "공시"} for f in dart.recent_filings(code)]
        news = [{**x, "kind": "뉴스"} for x in _news(f'"{name}"', 5)["items"]]
        items = _with_sentiment(filings + news)
        return {"code": code, "name": name, "verdict": news_gate.gate(items), "items": items}
    return [_cached(("gate", c), 600, lambda c=c: one(c)) for c in accounts.get_watch(u["id"])]


@app.get("/api/value", dependencies=[Depends(_need_auth), Depends(_need_dart)])
def api_value():
    """1단계 A 후보. 최근 사업보고서 기준. 첫 계산은 DART 호출 수십 번이라 1분 가까이 걸린다."""
    today = date.today()
    year = today.year - (1 if today.month >= 4 else 2)  # 사업보고서 제출 기한: 3월 말

    def calc():
        lst = _listing()
        lst = pd.DataFrame(lst[lst.index.str.endswith("0") & lst["Market"].isin(["KOSPI", "KOSDAQ"])
                               & ~lst["Name"].str.contains("스팩")])  # 보통주만, 스팩 제외
        acc = dart.annual_accounts(lst.index.tolist(), year)
        top = selection.value_screen(lst, acc)
        return {"year": year, "universe": len(lst), "with_accounts": len(acc),
                "rows": [{"code": c, "name": r.Name, "market": r.Market, "marcap": float(r.Marcap),
                          "per": round(float(r.per), 2), "pbr": round(float(r.pbr), 2), "filed": r.filed}
                         for c, r in top.iterrows()]}
    return _cached("value", 86400, calc)


@app.get("/api/backtest", dependencies=[Depends(_need_auth)])
def api_backtest(symbol: str = Query("069500", pattern=CODE),
                 ma: int = Query(60, ge=5, le=250),
                 band: float = Query(0.01, ge=0, le=0.1),
                 start: str = Query("2015-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")):
    close = pd.Series(fdr.DataReader(symbol, start)["Close"], dtype=float)
    if len(close) < ma + 2:
        raise HTTPException(400, "데이터가 부족합니다")
    pos = hysteresis_position(close, pd.Series(close.rolling(ma).mean()), band)
    strat = backtest.run(close, pos)
    hold = backtest.run(close, close * 0 + 1)
    return {
        "dates": pd.DatetimeIndex(close.index).strftime("%Y-%m-%d").tolist(),
        "strategy": strat.round(4).tolist(),
        "hold": hold.round(4).tolist(),
        "trades": int(pos.diff().abs().sum()),
        "metrics": {"strategy": backtest.metrics(strat), "hold": backtest.metrics(hold)},
    }


# ── 내 계정 ──
class PasswordIn(BaseModel):
    old: str = Field(max_length=accounts.MAX_PW)
    new: str = Field(max_length=accounts.MAX_PW)


class KeysIn(BaseModel):
    app_key: str = Field(max_length=100)
    app_secret: str = Field(max_length=600)
    account: str = Field(max_length=20)
    mode: str = Field("mock", max_length=10)


class SettingsIn(BaseModel):
    auto_trade: bool
    max_daily_loss_pct: float
    max_order_krw: int


@app.get("/api/me")
def api_me(u=Depends(_me)):
    return {"username": u["username"], "is_admin": bool(u["is_admin"]), "must_change": bool(u["must_change"]),
            "consented": u["consent_at"] is not None}


@app.post("/api/me/password")
def api_password(body: PasswordIn, request: Request, u=Depends(_me)):
    _invalid(accounts.change_password, u["id"], body.old, body.new)
    fresh = accounts.get_user(u["id"])
    if fresh is not None:
        request.session["tag"] = accounts.session_tag(fresh)  # 이 기기는 유지, 다른 기기는 끊김
    return {"ok": True}


@app.post("/api/me/consent")
def api_consent(u=Depends(_me)):
    if u["must_change"]:
        raise HTTPException(403, "change_password")
    accounts.consent(u["id"])
    return {"ok": True}


@app.get("/api/me/settings")
def api_settings(u=Depends(_need_auth)):
    return {**accounts.get_settings(u["id"]), "kis": accounts.masked_keys(u["id"])}


@app.put("/api/me/settings")
def api_settings_put(body: SettingsIn, u=Depends(_need_auth)):
    _invalid(accounts.update_settings, u["id"], body.auto_trade, body.max_daily_loss_pct, body.max_order_krw)
    return accounts.get_settings(u["id"])


@app.put("/api/me/kis")
def api_kis_put(body: KeysIn, u=Depends(_need_auth)):
    _invalid(accounts.save_keys, u["id"], body.app_key, body.app_secret, body.account, body.mode)
    return accounts.masked_keys(u["id"])


@app.delete("/api/me/kis")
def api_kis_del(u=Depends(_need_auth)):
    accounts.delete_keys(u["id"])
    return {"ok": True}


@app.post("/api/me/kis/verify")
def api_kis_verify(u=Depends(_need_auth)):
    k = accounts.load_keys(u["id"])
    if k is None:
        raise HTTPException(400, "등록된 키가 없어요")
    ok, msg = kis.verify(k["app_key"], k["app_secret"], k["mode"])
    return {"ok": ok, "message": msg}


# ── 관리자: 지인 계정 발급. 다른 사람의 키·자동매매는 건드릴 수 없다 ──
class NewUserIn(BaseModel):
    username: str = Field(max_length=40)
    is_admin: bool = False


class ActiveIn(BaseModel):
    active: bool


@app.get("/api/admin/users", dependencies=[Depends(_admin)])
def admin_users():
    return accounts.list_users()


@app.post("/api/admin/users", dependencies=[Depends(_admin)])
def admin_create(body: NewUserIn):
    return {"username": body.username.strip().lower(),
            "temp_password": _invalid(accounts.create_user, body.username, body.is_admin)}


def _target(uid: int, me) -> None:
    """관리자가 손댈 수 있는 계정인지. 자기 자신과 다른 관리자는 안 된다(비밀번호는 내 계정에서 직접 변경)."""
    t = accounts.get_user(uid)
    if t is None:
        raise HTTPException(404)
    if uid == me["id"] or t["is_admin"]:
        raise HTTPException(400, "관리자 계정은 여기서 바꿀 수 없어요")


@app.post("/api/admin/users/{uid}/reset")
def admin_reset(uid: int, me=Depends(_admin)):
    _target(uid, me)
    return {"temp_password": accounts.reset_password(uid)}


@app.post("/api/admin/users/{uid}/active")
def admin_active(uid: int, body: ActiveIn, me=Depends(_admin)):
    _target(uid, me)
    accounts.set_active(uid, body.active)
    return {"ok": True}
