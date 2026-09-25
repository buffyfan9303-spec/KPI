import os
from pathlib import Path

from dotenv import load_dotenv

# 비밀값은 OneDrive 동기화 폴더 밖(KIS 샘플과 같은 위치)에 둔다.
ENV_FILE = Path.home() / "KIS" / "config" / ".env"
load_dotenv(ENV_FILE)

# 없으면 바로 실패시켜서 비밀번호 없는 대시보드가 뜨는 일을 막는다.
DASH_PASSWORD = os.environ["DASH_PASSWORD"]
SESSION_SECRET = os.environ["SESSION_SECRET"]
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "1") == "1"

# 선택 키: 없으면 해당 기능만 꺼진다.
DART_API_KEY = os.environ.get("DART_API_KEY", "")
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")


def _data_key() -> str:
    """KIS 앱키 암호화용 Fernet 키. 암호화된 DB(~/KIS/config)와 같은 폴더에 두지 않도록
    Windows 자격 증명 관리자에 보관한다. .env에 남아 있으면 한 번 옮겨 담는다."""
    import keyring
    key = keyring.get_password("alpha-trader", "DATA_KEY")
    env = os.environ.get("DATA_KEY")
    if key is None and env:
        keyring.set_password("alpha-trader", "DATA_KEY", env)
        key = env
    if key is None:
        raise RuntimeError("DATA_KEY가 없어요. Windows 자격 증명 관리자의 alpha-trader 항목을 확인하세요")
    return key


DATA_KEY = _data_key()
