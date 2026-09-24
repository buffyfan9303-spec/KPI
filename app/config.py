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
