"""지인용 다중 사용자: 계정, KIS 앱키(암호화), 개인 설정, 관심종목.

- DB는 OneDrive 밖(~/KIS/config/alpha.db). 앱키·시크릿·계좌번호는 Fernet(AES)으로 암호화해 저장한다.
- 복호화된 키는 주문 엔진(load_keys)만 쓴다. 화면·관리자에게는 가린 값만 준다.
- 실전(real) 모드는 DB 제약으로 막아 둔다. 실전 전환은 사용자 결정 후 별도 작업.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager

from cryptography.fernet import Fernet, InvalidToken

from app import config

DB_FILE = config.ENV_FILE.parent / "alpha.db"
LEGACY_WATCH = config.ENV_FILE.parent / "watchlist.json"
DEFAULT_WATCH = ["005930", "000660", "069500", "035420"]
MAX_WATCH = 12
USERNAME = re.compile(r"^[a-z0-9_]{3,20}$")
MAX_PW = 200
TEMP_TTL = 72 * 3600  # 임시 비밀번호 유효 시간
MIN_PW = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw TEXT NOT NULL,
  is_admin INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
  must_change INTEGER NOT NULL DEFAULT 1, consent_at REAL, created_at REAL NOT NULL,
  session_ver INTEGER NOT NULL DEFAULT 0, temp_at REAL);
CREATE TABLE IF NOT EXISTS settings(
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  auto_trade INTEGER NOT NULL DEFAULT 0,
  max_daily_loss_pct REAL NOT NULL DEFAULT 2.0,
  max_order_krw INTEGER NOT NULL DEFAULT 1000000);
CREATE TABLE IF NOT EXISTS kis_keys(
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  mode TEXT NOT NULL CHECK(mode = 'mock'),
  app_key BLOB NOT NULL, app_secret BLOB NOT NULL, account BLOB NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS watchlist(
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  code TEXT NOT NULL, pos INTEGER NOT NULL, PRIMARY KEY(user_id, code));
"""


class Invalid(ValueError):
    """사용자에게 그대로 보여줘도 되는 입력 오류."""


@contextmanager
def _db():
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    try:
        with c:
            yield c
    finally:
        c.close()


def _fernet() -> Fernet:
    return Fernet(config.DATA_KEY.encode())


# ── 비밀번호 ──
def hash_pw(pw: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_pw(pw: str, stored: str) -> bool:
    _, salt, h = stored.split("$")
    got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1).hex()
    return hmac.compare_digest(got, h)


_DUMMY = hash_pw(secrets.token_hex(8))  # 없는 아이디도 같은 시간이 걸리게


def _check_new_pw(pw: str):
    if len(pw) < MIN_PW:
        raise Invalid(f"비밀번호는 {MIN_PW}자 이상이어야 해요")


# ── 계정 ──
def init():
    """스키마 생성. 사용자가 없으면 기존 대시보드 비밀번호로 관리자(admin)를 만들고 옛 관심종목을 옮긴다."""
    _fernet()  # DATA_KEY 형식이 틀리면 여기서 바로 실패
    with _db() as c:
        c.executescript(SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(users)")}  # 예전 DB에 새 칸 추가
        if "session_ver" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN session_ver INTEGER NOT NULL DEFAULT 0")
        if "temp_at" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN temp_at REAL")
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
            return
        uid = _insert_user(c, "admin", hash_pw(config.DASH_PASSWORD), is_admin=True, must_change=False)
        codes = json.loads(LEGACY_WATCH.read_text()) if LEGACY_WATCH.exists() else DEFAULT_WATCH
        _set_watch(c, uid, codes)


def _insert_user(c, username: str, pw_hash: str, is_admin: bool, must_change: bool) -> int:
    now = time.time()
    cur = c.execute("INSERT INTO users(username, pw, is_admin, must_change, created_at, temp_at) VALUES (?,?,?,?,?,?)",
                    (username, pw_hash, int(is_admin), int(must_change), now, now if must_change else None))
    uid = int(cur.lastrowid or 0)
    c.execute("INSERT INTO settings(user_id) VALUES (?)", (uid,))
    return uid


def authenticate(username: str, pw: str) -> sqlite3.Row | None:
    with _db() as c:
        u = c.execute("SELECT * FROM users WHERE username = ?", (username.lower(),)).fetchone()
    if u is None:
        check_pw(pw, _DUMMY)
        return None
    if not check_pw(pw, u["pw"]) or not u["active"]:
        return None
    if u["must_change"] and u["temp_at"] and time.time() - u["temp_at"] > TEMP_TTL:
        return None  # 만료된 임시 비밀번호 → 관리자가 다시 초기화
    return u


def get_user(uid: int) -> sqlite3.Row | None:
    with _db() as c:
        return c.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


def session_tag(u: sqlite3.Row) -> str:
    """비밀번호가 바뀌거나 로그아웃하면 달라지는 값. 세션에 넣어 두면 그때 모든 기기 로그인이 끊긴다."""
    return hashlib.sha256(f'{u["pw"]}:{u["session_ver"]}'.encode()).hexdigest()[:16]


def end_sessions(uid: int):
    """로그아웃: 이 사용자의 모든 기기 세션을 끊는다(쿠키를 복사해 둔 경우 포함)."""
    with _db() as c:
        c.execute("UPDATE users SET session_ver = session_ver + 1 WHERE id = ?", (uid,))


def create_user(username: str, is_admin: bool = False) -> str:
    """임시 비밀번호를 한 번만 돌려준다. 첫 로그인 때 바꾸게 한다."""
    username = username.strip().lower()
    if not USERNAME.match(username):
        raise Invalid("아이디는 영문 소문자·숫자·_ 3~20자예요")
    temp = secrets.token_urlsafe(9)
    with _db() as c:
        if c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise Invalid("이미 있는 아이디예요")
        uid = _insert_user(c, username, hash_pw(temp), is_admin=is_admin, must_change=True)
        _set_watch(c, uid, DEFAULT_WATCH)
    return temp


def change_password(uid: int, old: str, new: str):
    u = get_user(uid)
    if u is None or not check_pw(old, u["pw"]):
        raise Invalid("현재 비밀번호가 틀렸어요")
    _check_new_pw(new)
    if old == new:
        raise Invalid("새 비밀번호가 지금과 같아요")
    with _db() as c:
        c.execute("UPDATE users SET pw = ?, must_change = 0, temp_at = NULL WHERE id = ?", (hash_pw(new), uid))


def reset_password(uid: int) -> str:
    """임시 비밀번호 발급. 관리자가 초기화 후 그 계정으로 들어가 자동매매를 켜지 못하도록
    동의·KIS 키·자동매매를 함께 지운다. 본인이 다시 동의하고 키를 넣어야 한다."""
    temp = secrets.token_urlsafe(9)
    with _db() as c:
        c.execute("UPDATE users SET pw = ?, must_change = 1, consent_at = NULL, temp_at = ? WHERE id = ?",
                  (hash_pw(temp), time.time(), uid))
        c.execute("DELETE FROM kis_keys WHERE user_id = ?", (uid,))
        c.execute("UPDATE settings SET auto_trade = 0 WHERE user_id = ?", (uid,))
    return temp


def set_active(uid: int, active: bool):
    with _db() as c:
        c.execute("UPDATE users SET active = ? WHERE id = ?", (int(active), uid))
        if not active:
            c.execute("UPDATE settings SET auto_trade = 0 WHERE user_id = ?", (uid,))


def consent(uid: int):
    with _db() as c:
        c.execute("UPDATE users SET consent_at = ? WHERE id = ?", (time.time(), uid))


def list_users() -> list[dict]:
    """관리자용. 키 값은 절대 넣지 않고 등록 여부만."""
    with _db() as c:
        rows = c.execute("""SELECT u.id, u.username, u.is_admin, u.active, u.must_change, u.consent_at, u.temp_at,
                                   s.auto_trade, k.user_id IS NOT NULL AS has_keys
                            FROM users u JOIN settings s ON s.user_id = u.id
                            LEFT JOIN kis_keys k ON k.user_id = u.id ORDER BY u.id""").fetchall()
    return [{**dict(r), "is_admin": bool(r["is_admin"]), "active": bool(r["active"]),
             "must_change": bool(r["must_change"]), "consented": r["consent_at"] is not None,
             "temp_expired": bool(r["must_change"] and r["temp_at"] and time.time() - r["temp_at"] > TEMP_TTL),
             "auto_trade": bool(r["auto_trade"]), "has_keys": bool(r["has_keys"])} for r in rows]


# ── KIS 앱키 ──
APP_KEY = re.compile(r"^[A-Za-z0-9]{16,64}$")
APP_SECRET = re.compile(r"^[A-Za-z0-9+/=]{64,512}$")
ACCOUNT = re.compile(r"^\d{8}-?\d{2}$")


def save_keys(uid: int, app_key: str, app_secret: str, account: str, mode: str = "mock"):
    app_key, app_secret, account = app_key.strip(), app_secret.strip(), account.strip()
    if mode != "mock":
        raise Invalid("지금은 모의투자 키만 등록할 수 있어요")
    if not APP_KEY.match(app_key):
        raise Invalid("앱키 형식이 아니에요")
    if not APP_SECRET.match(app_secret):
        raise Invalid("앱시크릿 형식이 아니에요")
    if not ACCOUNT.match(account):
        raise Invalid("계좌번호는 8자리-2자리 형식이에요 (예: 12345678-01)")
    f = _fernet()
    with _db() as c:
        c.execute("""INSERT INTO kis_keys(user_id, mode, app_key, app_secret, account, updated_at) VALUES (?,?,?,?,?,?)
                     ON CONFLICT(user_id) DO UPDATE SET mode=excluded.mode, app_key=excluded.app_key,
                     app_secret=excluded.app_secret, account=excluded.account, updated_at=excluded.updated_at""",
                  (uid, mode, f.encrypt(app_key.encode()), f.encrypt(app_secret.encode()),
                   f.encrypt(account.replace("-", "").encode()), time.time()))


def load_keys(uid: int) -> dict | None:
    """복호화한 키. 주문 엔진·키 확인에서만 쓰고 응답으로 보내지 않는다."""
    with _db() as c:
        r = c.execute("SELECT * FROM kis_keys WHERE user_id = ?", (uid,)).fetchone()
    if r is None:
        return None
    f = _fernet()
    return {"mode": r["mode"], "app_key": f.decrypt(r["app_key"]).decode(),
            "app_secret": f.decrypt(r["app_secret"]).decode(), "account": f.decrypt(r["account"]).decode(),
            "updated_at": r["updated_at"]}


def masked_keys(uid: int) -> dict | None:
    try:
        k = load_keys(uid)
    except InvalidToken:  # DATA_KEY가 바뀌어 옛 키를 못 읽는 경우
        return {"mode": "mock", "app_key": "복호화 실패", "account": "다시 등록해 주세요", "updated_at": 0}
    if k is None:
        return None
    return {"mode": k["mode"], "app_key": k["app_key"][:2] + "…" + k["app_key"][-4:],
            "account": "******" + k["account"][-4:-2] + "-" + k["account"][-2:], "updated_at": k["updated_at"]}


def delete_keys(uid: int):
    with _db() as c:
        c.execute("DELETE FROM kis_keys WHERE user_id = ?", (uid,))
        c.execute("UPDATE settings SET auto_trade = 0 WHERE user_id = ?", (uid,))


# ── 개인 설정 ──
def get_settings(uid: int) -> dict:
    with _db() as c:
        r = c.execute("SELECT * FROM settings WHERE user_id = ?", (uid,)).fetchone()
    return {"auto_trade": bool(r["auto_trade"]), "max_daily_loss_pct": r["max_daily_loss_pct"],
            "max_order_krw": r["max_order_krw"]}


def update_settings(uid: int, auto_trade: bool, max_daily_loss_pct: float, max_order_krw: int):
    if not 0.5 <= max_daily_loss_pct <= 10:
        raise Invalid("하루 최대 손실은 0.5~10% 사이로 정해요")
    if not 10_000 <= max_order_krw <= 100_000_000:
        raise Invalid("1회 최대 주문 금액은 1만~1억 원 사이로 정해요")
    u = get_user(uid)
    if auto_trade and (u is None or u["consent_at"] is None or load_keys(uid) is None):
        raise Invalid("자동매매를 켜려면 동의와 KIS 모의투자 키 등록이 먼저 필요해요")
    with _db() as c:
        c.execute("UPDATE settings SET auto_trade = ?, max_daily_loss_pct = ?, max_order_krw = ? WHERE user_id = ?",
                  (int(auto_trade), max_daily_loss_pct, max_order_krw, uid))


# ── 관심종목 ──
def _set_watch(c, uid: int, codes: list[str]):
    c.execute("DELETE FROM watchlist WHERE user_id = ?", (uid,))
    c.executemany("INSERT INTO watchlist(user_id, code, pos) VALUES (?,?,?)",
                  [(uid, code, i) for i, code in enumerate(codes[:MAX_WATCH])])


def get_watch(uid: int) -> list[str]:
    with _db() as c:
        return [r["code"] for r in c.execute("SELECT code FROM watchlist WHERE user_id = ? ORDER BY pos", (uid,))]


def add_watch(uid: int, code: str):
    codes = get_watch(uid)
    if code in codes:
        return
    if len(codes) >= MAX_WATCH:
        raise Invalid(f"관심종목은 {MAX_WATCH}개까지예요")
    with _db() as c:
        _set_watch(c, uid, codes + [code])


def remove_watch(uid: int, code: str):
    with _db() as c:
        c.execute("DELETE FROM watchlist WHERE user_id = ? AND code = ?", (uid, code))
