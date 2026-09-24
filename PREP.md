# 제작 준비 상태 (2026-09-25)

원본: `Engineered_Alpha_Hybrid_Architecture.pdf` (14쪽, 이미지형 슬라이드 → 페이지 렌더링으로 전부 읽음)

## PDF 요약 → 구현 모듈

| 단계 | PDF 내용 | 구현 모듈(예정) |
|---|---|---|
| 1 Selection | A: 시총 하위 20% 중 PER·PBR 최하위 / B: 대형주 최근 5일 EPS 추정치 상향 | `selection.py` |
| 2 Allocation | CBVR 3중 필터(EMA ±1~4% 레벨·변동성 속도·추세 벡터) + DRL Super-Agent, 리스크 패리티 ±3.0% 봉투 | `allocation.py` |
| 3 Timing | Hysteresis ±1.0% 버퍼, VIX 급등 시 비중 1/10, 하락장 3거래일 매일 30% 현금화 | `timing.py` |
| 4 Execution | FinBERT 뉴스 EXECUTE/REJECT, KIS API 초당 15건 Token Bucket + Redis 큐, 현재가 ±0.2% 지정가·미체결 정정, WebSocket 체결통보 | `news_gate.py`, `broker_kis.py` |
| 공통 | 매도 거래세 0.20%, 슬리피지 0.5~2.5%, ADV 대비 Market Impact 한도, 손절 -3~-5% | `costs.py`, 백테스트 |

주의: PDF의 CAGR 30%·MDD 13%·샤프 1.2는 **시뮬레이션 주장값**이다. 실측이 아니므로 백테스트로 재검증한다.

## 설치·확인 상태

| 항목 | 상태 | 확인 방법 |
|---|---|---|
| Python venv `C:\Users\buffy\.venvs\alpha-trader` (3.14, OneDrive 밖) | 기능 실행 확인 | import 전부 성공 |
| torch 2.14 CPU / transformers 5.17 / stable-baselines3 2.9 / gymnasium | 기능 실행 확인 | import |
| pandas, FinanceDataReader, pykrx, yfinance | 기능 실행 확인 | 삼성전자 9월 17행 수신 |
| Redis 7 (Docker `alpha-redis`, 127.0.0.1:6379, 자동 재시작) | 기능 실행 확인 | `PING` → `PONG` |
| pyright 1.1.414 (npm 전역) + `pyright-lsp` 플러그인(활성) | 등록 확인 | `pyright --version` |
| context7 MCP (라이브러리 문서) | 연결 확인 | 세션 도구 목록 |
| KIS Code Assistant MCP (공식, API 검색·샘플코드, 키 불필요) | 등록 확인 · 승인 대기 | `.mcp.json`, `--help` 실행 성공. 새 세션에서 승인 |
| uv 0.12.18 (`C:\Users\buffy\AppData\Roaming\Python\Python314\Scripts\uv.exe`) | 기능 실행 확인 | `uv --version` |
| KIS Trading MCP (AI가 직접 주문) | 이번 범위 제외 | 실전 키 필수·런타임 코드 다운로드 실행 → 자동매매 코드엔 불필요 |
| KIS 앱키/시크릿·계좌 | **설정·인증 필요 (사용자)** | KIS Developers에서 발급 → `.env` |
| FinBERT 모델 가중치 | 이번 범위 제외 | 구현 시 한국어 모델 선택 후 다운로드 |
| pykrx KRX 로그인(`KRX_ID/KRX_PW`) | 선택 사항 | 없어도 FDR로 시세 수집 가능 |

## KIS 공식 저장소 확인 (koreainvestment/open-trading-api, b4e6249, 2026-08-26)

- 참고할 코드: `examples_user/kis_auth.py`(토큰·공통 호출), `examples_user/domestic_stock/domestic_stock_functions.py`(`order_cash`, `order_rvsecncl`, `inquire_balance`, `inquire_price`, `inquire_daily_itemchartprice`), `domestic_stock_functions_ws.py`(`ccnl_notice` 체결통보)
- 설정: `~/KIS/config/kis_devlp.yaml`에 실전·모의 앱키, 계좌 입력. 토큰은 같은 폴더에 파일로 캐시.
- 호출 한도: 실전 0.05초(초당 20건), 모의 0.5초(초당 2건). PDF의 초당 15건은 실전 기준이며 모의투자에서는 초당 2건 이하로 제한할 것.
- 샘플 버그: `changeTREnv`에서 `_smartSleep`에 `global` 선언이 없어 실전/모의 전환 시 간격이 바뀌지 않음(항상 0.1초). 직접 구현 시 재사용 금지.
- 클론 시 Windows 경로 길이 제한 → `git -c core.longpaths=true` 필요.
- `backtester/`는 QuantConnect Lean(Docker) + 웹 UI, `strategy_builder/`는 프리셋 10종 + 지표 80종. 이 시스템의 1~4단계 로직은 없으므로 참고만.

## 웹 대시보드 (모바일 접속)

| 파일 | 역할 |
|---|---|
| `app/web.py` | FastAPI. 비밀번호 로그인(5회 실패 시 5분 잠금), 세션 쿠키(Secure·SameSite=Strict·12시간), `/api/backtest` 입력 검증, API 문서 페이지 비활성 |
| `app/backtest.py` | 다음 날 체결, 슬리피지 0.5%, 매도세 0.20%, CAGR·MDD |
| `app/timing.py` | 3단계 Hysteresis — **사용자 구현 대기** (`test_core.py` 2개가 기준) |
| `app/static/*.html` | 로그인·대시보드, 모바일 우선, 다크 모드 |
| `C:\Users\buffy\KIS\config\.env` | `DASH_PASSWORD`, `SESSION_SECRET` (OneDrive 밖, 본인 계정만 접근) |

검증: 로그인·401·잠금·로그아웃 흐름 테스트 통과, 비용 테스트 통과, pyright 0 errors. Hysteresis 미구현이라 백테스트 화면은 아직 오류(500).

실행: `run_web.ps1` → 이 PC에서 http://127.0.0.1:8000 (로컬에서 쿠키가 안 붙으면 `.env`에 `COOKIE_SECURE=0`, Tailscale HTTPS로 쓸 땐 1)

휴대폰 연결 (사용자 작업, 계정 로그인 필요):
1. PC와 휴대폰에 Tailscale 설치 후 같은 계정으로 로그인 (https://tailscale.com/download)
2. 관리 콘솔에서 MagicDNS·HTTPS 인증서 켜기
3. PC에서 `tailscale serve --bg 8000` → 휴대폰에서 `https://<PC이름>.<tailnet>.ts.net`
4. `tailscale funnel`은 쓰지 말 것(인터넷 전체 공개됨)

## 다음 세션 작업 배정

| 작업 | 도구/스킬 | 모델 |
|---|---|---|
| 모듈 설계 | `feature-dev:feature-dev`, `Plan` 에이전트 | Opus |
| KIS 주문·레이트리밋·보안(키 관리) | 직접 구현 + `claude-security:scan` | Opus (금액·보안) |
| 데이터 수집·백테스트 | 직접 구현, `pytest` | Sonnet |
| 코드 리뷰 | `code-review`, `pr-review-toolkit:silent-failure-hunter` | Opus |

## 수용 기준
- 모의투자 계좌에서만 먼저 주문한다. 실계좌 전환은 사용자가 결정한다.
- 백테스트에 거래세·슬리피지를 반영해 CAGR과 MDD를 실측으로 보고한다.
- KIS 호출은 초당 15건을 넘지 않는다(테스트로 확인).
- 키는 `.env`에만 두고 git에 넣지 않는다.

## 실행
```bash
C:\Users\buffy\.venvs\alpha-trader\Scripts\activate
```

Claude Code에 붙여 넣을 문구:
> `PREP.md`를 읽고 1단계(데이터 수집 + 비용 포함 백테스트 골격)부터 구현해줘. 모의투자만 사용.
