# 대시보드 실행: 이 PC(127.0.0.1)에서만 열린다. 휴대폰 접속은 Tailscale serve가 HTTPS로 중계.
Set-Location $PSScriptRoot
& "C:\Users\buffy\.venvs\alpha-trader\Scripts\python.exe" -m uvicorn app.web:app --host 127.0.0.1 --port 8000
