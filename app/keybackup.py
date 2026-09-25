"""PC 초기화 대비 백업·복구: DATA_KEY + 계정 DB(alpha.db) + .env 를 한 파일로 묶어 복구 비밀번호로 잠근다.

기본 저장 위치는 구글 드라이브(드라이브 데스크톱 앱의 "내 드라이브")라 PC가 초기화돼도 남는다.
파일만으로는 아무것도 꺼낼 수 없고 복구 비밀번호가 함께 있어야 한다. 복구 비밀번호는 PC 밖(휴대폰 메모 앱 등)에 적어 둔다.

  백업:  python -m app.keybackup export            (기본: 구글 드라이브 내 드라이브\\alpha-backup\\alpha-backup-날짜.json)
  복구:  python -m app.keybackup import <백업 파일>  [--force]
"""
import base64
import getpass
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import keyring
from cryptography.fernet import Fernet, InvalidToken

SERVICE, NAME = "alpha-trader", "DATA_KEY"
CONFIG_DIR = Path.home() / "KIS" / "config"


def google_drive() -> Path | None:
    """구글 드라이브 데스크톱 앱이 연결한 드라이브(G: 등)의 '내 드라이브' 폴더."""
    for letter in "GHIJKLMNOPQRSTUVWXYZDEF":
        for name in ("내 드라이브", "My Drive"):
            p = Path(f"{letter}:/") / name
            if p.is_dir():
                return p
    return None
MIN_PASS = 12


def _wrap(passphrase: str, salt: bytes) -> Fernet:
    k = hashlib.scrypt(passphrase.encode(), salt=salt, n=2**15, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=32)
    return Fernet(base64.urlsafe_b64encode(k))


def seal(text: str, passphrase: str) -> dict:
    salt = os.urandom(16)
    return {"v": 2, "salt": salt.hex(), "token": _wrap(passphrase, salt).encrypt(text.encode()).decode()}


def unseal(blob: dict, passphrase: str) -> str:
    return _wrap(passphrase, bytes.fromhex(blob["salt"])).decrypt(blob["token"].encode()).decode()


def _db_snapshot(db: Path) -> bytes:
    """서버가 켜져 있어도 깨지지 않게 SQLite 백업 기능으로 복사."""
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "snap.db"
        src_c, dst_c = sqlite3.connect(db), sqlite3.connect(dst)
        with dst_c:
            src_c.backup(dst_c)
        src_c.close()
        dst_c.close()
        return dst.read_bytes()


def pack(key: str, env_text: str, db_bytes: bytes | None) -> str:
    Fernet(key.encode())  # 올바른 키인지 확인
    return json.dumps({"data_key": key, "env": env_text,
                       "db": base64.b64encode(db_bytes).decode() if db_bytes else None})


def export(path: Path | None):
    key = keyring.get_password(SERVICE, NAME)
    if key is None:
        sys.exit("자격 증명 관리자에 DATA_KEY가 없어요. 서버를 한 번 실행한 뒤 다시 해 주세요.")
    env_file, db_file = CONFIG_DIR / ".env", CONFIG_DIR / "alpha.db"
    text = pack(key, env_file.read_text(encoding="utf-8") if env_file.exists() else "",
                _db_snapshot(db_file) if db_file.exists() else None)
    p1 = getpass.getpass(f"복구 비밀번호 ({MIN_PASS}자 이상, 입력해도 화면에 안 보여요): ")
    if len(p1) < MIN_PASS:
        sys.exit(f"{MIN_PASS}자 이상으로 정해 주세요.")
    if getpass.getpass("한 번 더: ") != p1:
        sys.exit("두 번 입력한 값이 달라요.")
    if path is None:
        drive = google_drive()
        if drive is None:
            sys.exit("구글 드라이브(내 드라이브)를 찾지 못했어요. 드라이브 앱이 켜져 있는지 확인하거나 저장할 파일 경로를 직접 적어 주세요.")
        (drive / "alpha-backup").mkdir(exist_ok=True)
        path = drive / "alpha-backup" / f"alpha-backup-{time.strftime('%Y%m%d-%H%M')}.json"
    path.write_text(json.dumps(seal(text, p1)), encoding="utf-8")
    assert unseal(json.loads(path.read_text(encoding="utf-8")), p1) == text  # 만든 파일로 실제 복구되는지 확인
    print(f"백업 완료: {path}\n복구 비밀번호는 PC 밖(휴대폰 메모 등)에 따로 적어 두세요. 파일과 비밀번호가 둘 다 있어야 복구돼요.")


def restore(path: Path):
    try:
        data = json.loads(unseal(json.loads(path.read_text(encoding="utf-8")), getpass.getpass("복구 비밀번호: ")))
    except InvalidToken:
        sys.exit("복구 비밀번호가 틀렸어요.")
    force = "--force" in sys.argv
    current = keyring.get_password(SERVICE, NAME)
    if current and current != data["data_key"] and not force:
        sys.exit("이 PC에 이미 다른 키가 있어요. 정말 덮어쓰려면 끝에 --force를 붙여 주세요.")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in ((".env", data["env"].encode("utf-8")),
                          ("alpha.db", base64.b64decode(data["db"]) if data["db"] else b"")):
        f = CONFIG_DIR / name
        if not content:
            continue
        if f.exists() and not force:
            sys.exit(f"{f} 가 이미 있어요. 덮어쓰려면 끝에 --force를 붙여 주세요.")
        f.write_bytes(content)
    keyring.set_password(SERVICE, NAME, data["data_key"])
    print("복구 완료: 암호화 키, 계정 DB, .env 를 되살렸어요. 서버를 켜면 지인 계정과 앱키가 그대로예요.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("export", "import") or (sys.argv[1] == "import" and len(sys.argv) < 3):
        sys.exit(__doc__)
    if sys.argv[1] == "export":
        export(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        restore(Path(sys.argv[2]))
