import hmac
import time
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app import backtest, config
from app.timing import hysteresis_position

STATIC = Path(__file__).parent / "static"
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET,
                   https_only=config.COOKIE_SECURE, same_site="strict", max_age=12 * 3600)

# ponytail: 전역 잠금 1개. 사용자 1명 전제, 다중 사용자면 IP별로 분리
_fails = {"n": 0, "until": 0.0}
MAX_FAILS, LOCK_SEC = 5, 300


def _authed(request: Request) -> bool:
    return request.session.get("ok") is True


@app.get("/")
def index(request: Request):
    return FileResponse(STATIC / ("index.html" if _authed(request) else "login.html"))


@app.post("/login")
def login(request: Request, password: str = Form(...)):
    now = time.time()
    if now < _fails["until"]:
        return RedirectResponse("/?e=lock", status_code=303)
    if hmac.compare_digest(password.encode(), config.DASH_PASSWORD.encode()):
        _fails["n"] = 0
        request.session.clear()
        request.session["ok"] = True
        return RedirectResponse("/", status_code=303)
    _fails["n"] += 1
    if _fails["n"] >= MAX_FAILS:
        _fails.update(n=0, until=now + LOCK_SEC)
    return RedirectResponse("/?e=1", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@app.get("/api/backtest")
def api_backtest(request: Request,
                 symbol: str = Query("069500", pattern=r"^\d{6}$"),
                 ma: int = Query(60, ge=5, le=250),
                 band: float = Query(0.01, ge=0, le=0.1),
                 start: str = Query("2015-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")):
    if not _authed(request):
        raise HTTPException(401)
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
