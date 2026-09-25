"""다중 사용자 보안 흐름. 임시 DB에서 실행하고 네트워크는 쓰지 않는다."""
import importlib

import pytest
from fastapi.testclient import TestClient

from app import accounts, config

KEY = "PS" + "a1B2c3D4e5" * 3          # 32자 가짜 앱키
SECRET = "Zx9Y" * 45                  # 180자 가짜 시크릿


@pytest.fixture()
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_FILE", tmp_path / "t.db")
    monkeypatch.setattr(accounts, "LEGACY_WATCH", tmp_path / "none.json")
    monkeypatch.setattr(config, "DASH_PASSWORD", "admin-pass-123")
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    import app.web
    return importlib.reload(app.web)   # 임시 DB로 accounts.init() 다시 실행


def login(web, user, pw):
    c = TestClient(web.app)
    r = c.post("/login", data={"username": user, "password": pw}, follow_redirects=False)
    assert r.status_code == 303 and "e=" not in r.headers["location"], r.headers["location"]
    return c


def onboard(web, admin, name):
    temp = admin.post("/api/admin/users", json={"username": name}).json()["temp_password"]
    c = login(web, name, temp)
    assert c.get("/api/me/settings").json()["detail"] == "change_password"
    assert c.post("/api/me/password", json={"old": temp, "new": name + "-new-pass"}).status_code == 200
    assert c.get("/api/me/settings").json()["detail"] == "consent"
    assert c.post("/api/me/consent").status_code == 200
    return c


def test_password_hash_roundtrip():
    h = accounts.hash_pw("correct horse")
    assert accounts.check_pw("correct horse", h) and not accounts.check_pw("wrong", h)
    assert "correct horse" not in h


def test_full_flow_isolation_and_masking(web):
    admin = login(web, "admin", "admin-pass-123")
    assert admin.post("/api/me/consent").status_code == 200
    alice, bob = onboard(web, admin, "alice"), onboard(web, admin, "bob")

    # 모의 키만 허용, 형식 검사
    bad = alice.put("/api/me/kis", json={"app_key": KEY, "app_secret": SECRET, "account": "12345678-01", "mode": "real"})
    assert bad.status_code == 400
    ok = alice.put("/api/me/kis", json={"app_key": KEY, "app_secret": SECRET, "account": "12345678-01"})
    assert ok.status_code == 200
    shown = alice.get("/api/me/settings").text
    assert KEY not in shown and SECRET not in shown and "12345678" not in shown  # 화면엔 가린 값만

    # DB 안에도 평문이 없다
    raw = accounts.DB_FILE.read_bytes()
    assert KEY.encode() not in raw and SECRET.encode() not in raw
    assert accounts.load_keys(accounts.authenticate("alice", "alice-new-pass")["id"])["app_key"] == KEY

    # bob은 alice 키·관심종목이 안 보이고, 키 없이 자동매매를 켤 수 없다
    assert bob.get("/api/me/settings").json()["kis"] is None
    alice_id = accounts.authenticate("alice", "alice-new-pass")["id"]
    accounts.add_watch(alice_id, "035720")
    assert "035720" not in accounts.get_watch(accounts.authenticate("bob", "bob-new-pass")["id"])
    r = bob.put("/api/me/settings", json={"auto_trade": True, "max_daily_loss_pct": 2, "max_order_krw": 1_000_000})
    assert r.status_code == 400

    # 관리자는 등록 여부만 보고 키는 못 본다. 일반 사용자는 관리자 API 금지
    users = admin.get("/api/admin/users").text
    assert '"has_keys":true' in users and KEY not in users
    assert alice.get("/api/admin/users").status_code == 403

    # 비밀번호 초기화 → 기존 세션 끊김, 계정 끄기 → 로그인 불가
    admin.post(f"/api/admin/users/{alice_id}/reset")
    assert alice.get("/api/me").status_code == 401
    admin.post(f"/api/admin/users/{alice_id}/active", json={"active": False})
    assert accounts.authenticate("alice", "alice-new-pass") is None


def test_login_lockout(web):
    c = TestClient(web.app)
    for _ in range(5):
        c.post("/login", data={"username": "admin", "password": "nope"}, follow_redirects=False)
    r = c.post("/login", data={"username": "admin", "password": "admin-pass-123"}, follow_redirects=False)
    assert r.headers["location"] == "/?e=lock"


def test_admin_reset_cannot_hijack_trading(web):
    """관리자가 비밀번호를 초기화해도 동의·키·자동매매가 지워져 그 계정으로 매매를 켤 수 없다."""
    admin = login(web, "admin", "admin-pass-123")
    admin.post("/api/me/consent")
    alice = onboard(web, admin, "alice")
    alice.put("/api/me/kis", json={"app_key": KEY, "app_secret": SECRET, "account": "12345678-01"})
    assert alice.put("/api/me/settings", json={"auto_trade": True, "max_daily_loss_pct": 2,
                                               "max_order_krw": 1_000_000}).status_code == 200
    uid = accounts.authenticate("alice", "alice-new-pass")["id"]
    temp = admin.post(f"/api/admin/users/{uid}/reset").json()["temp_password"]
    assert accounts.load_keys(uid) is None and not accounts.get_settings(uid)["auto_trade"]
    hijack = login(web, "alice", temp)
    hijack.post("/api/me/password", json={"old": temp, "new": "admin-owns-you-now"})
    assert hijack.get("/api/me/settings").json()["detail"] == "consent"  # 다시 동의부터
    # 관리자끼리·자기 자신은 초기화/끄기 불가
    admin_id = accounts.authenticate("admin", "admin-pass-123")["id"]
    assert admin.post(f"/api/admin/users/{admin_id}/reset").status_code == 400


def test_cross_site_post_blocked(web):
    c = TestClient(web.app)
    r = c.post("/login", data={"username": "admin", "password": "admin-pass-123"},
               headers={"Origin": "https://evil.example"}, follow_redirects=False)
    assert r.status_code == 403
    assert "frame-ancestors 'none'" in c.get("/").headers["content-security-policy"]


def test_logout_ends_all_sessions_and_temp_expires(web, monkeypatch):
    admin = login(web, "admin", "admin-pass-123")
    admin.post("/api/me/consent")
    other_device = login(web, "admin", "admin-pass-123")
    admin.post("/logout")
    assert other_device.get("/api/me").status_code == 401  # 다른 기기(복사된 쿠키)도 끊김

    admin = login(web, "admin", "admin-pass-123")
    temp = admin.post("/api/admin/users", json={"username": "late"}).json()["temp_password"]
    import time as _t
    real = _t.time
    monkeypatch.setattr(accounts.time, "time", lambda: real() + accounts.TEMP_TTL + 60)
    assert accounts.authenticate("late", temp) is None  # 72시간 지나면 임시 비밀번호로 못 들어옴


def test_backup_roundtrip(tmp_path):
    """백업 파일은 복구 비밀번호가 맞을 때만 키·DB·.env를 돌려주고, 파일 안에 평문이 없다."""
    import json as _json
    import sqlite3 as _sq
    from cryptography.fernet import Fernet, InvalidToken
    from app import keybackup
    db = tmp_path / "a.db"
    with _sq.connect(db) as c:
        c.execute("CREATE TABLE t(x)"); c.execute("INSERT INTO t VALUES ('secret-row')")
    key = Fernet.generate_key().decode()
    text = keybackup.pack(key, "DART_API_KEY=abc123", keybackup._db_snapshot(db))
    blob = keybackup.seal(text, "correct-horse-battery")
    raw = _json.dumps(blob)
    assert key not in raw and "abc123" not in raw and "secret-row" not in raw
    data = _json.loads(keybackup.unseal(blob, "correct-horse-battery"))
    assert data["data_key"] == key and "abc123" in data["env"]
    (tmp_path / "r.db").write_bytes(__import__("base64").b64decode(data["db"]))
    with _sq.connect(tmp_path / "r.db") as c:
        assert c.execute("SELECT x FROM t").fetchone()[0] == "secret-row"
    with pytest.raises(InvalidToken):
        keybackup.unseal(blob, "wrong-passphrase-123")
