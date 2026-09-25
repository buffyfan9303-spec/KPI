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
| 한국어 FinBERT `snunlp/KR-FinBert-SC` | 기능 실행 확인 | 뉴스 문장 긍정/부정 판정 성공 (라이선스 표기 없음 → 개인 연구용) |
| python-kis·opendartreader·aiolimiter·skfolio | import 확인 | 2026-09-25 설치 |
| pykrx KRX 로그인(`KRX_ID/KRX_PW`) | **PER/PBR에 필요** | 로그인 없으면 `get_market_fundamental` 빈 값(2026-09-25 확인). 시세는 FDR로 가능 |

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
| `app/timing.py` | 3단계 Hysteresis (±버퍼 밖으로 나갈 때만 전환, 버퍼 안은 직전 상태 유지) — 구현·테스트 통과 |
| `app/static/index.html` | 다크 대시보드: 관심종목(추세선·추가/삭제), 상승/하락 상위 6(시총 500억↑), 종목 차트(1개월~3년), 시장 지표(코스피·코스닥·VIX·원/달러, VIX 20%↑ 경고), 뉴스(Google RSS), 종목 검색(주식+ETF), 백테스트. 상승=빨강·하락=파랑. 1000px 미만은 햄버거 메뉴 |
| `app/static/login.html` | 로그인 |
| API | `/api/market` `/api/watchlist`(GET·POST·DELETE, `~/KIS/config/watchlist.json`, 최대 12) `/api/search` `/api/chart` `/api/news` — 모두 로그인 필요, 종목코드 `^[0-9A-Z]{6}$` |
| `C:\Users\buffy\KIS\config\.env` | `DASH_PASSWORD`, `SESSION_SECRET` (OneDrive 밖, 본인 계정만 접근) |

검증: 로그인·401·잠금·로그아웃 흐름 테스트 통과, 비용 테스트 통과, pyright 0 errors. 백테스트 실측(KODEX 200, 2015-01~2026-09, 60일선·±1%, 비용 포함): 전략 CAGR 9.5%·MDD -34.7%·매매 73회 / 단순 보유 CAGR 16.1%·MDD -40.8%. 3단계 단독으로는 보유보다 수익이 낮고 낙폭만 약간 줄었다.

실행: `run_web.ps1` → 이 PC에서 http://127.0.0.1:8000 (로컬에서 쿠키가 안 붙으면 `.env`에 `COOKIE_SECURE=0`, Tailscale HTTPS로 쓸 땐 1)

휴대폰 연결 (사용자 작업, 계정 로그인 필요):
1. PC와 휴대폰에 Tailscale 설치 후 같은 계정으로 로그인 (https://tailscale.com/download)
2. 관리 콘솔에서 MagicDNS·HTTPS 인증서 켜기
3. PC에서 `tailscale serve --bg 8000` → 휴대폰에서 `https://<PC이름>.<tailnet>.ts.net`
4. `tailscale funnel`은 쓰지 말 것(인터넷 전체 공개됨)

## 지인 계정 (다중 사용자, 2026-09-25)

| 파일 | 역할 |
|---|---|
| `app/accounts.py` | SQLite `~/KIS/config/alpha.db`(OneDrive 밖). 계정(scrypt 해시), KIS 앱키·시크릿·계좌(Fernet 암호화), 개인 설정, 관심종목 |
| `app/kis.py` | 모의투자 토큰 발급으로 키 확인만 (토큰 저장 안 함). 실제 호출은 키가 없어 미확인 |
| `test_accounts.py` | 비밀번호 변경 강제 → 동의 → 키 암호화·가림 → 사용자 간 분리 → 관리자 권한 → 세션 끊김 → 로그인 잠금 |

운영 방법
1. 처음 실행하면 `admin` 계정이 기존 `DASH_PASSWORD`로 만들어지고, 옛 관심종목이 옮겨진다.
2. 관리자 화면(내 계정 아래 "지인 계정 관리")에서 아이디를 만들면 임시 비밀번호가 **한 번만** 보인다. 본인에게 직접 전달한다.
3. 지인은 첫 로그인 때 비밀번호 변경 → 동의 5개 체크 → 본인 KIS **모의투자** 앱키 등록.
4. 관리자는 다른 사람의 키 값·자동매매 스위치를 볼 수도 바꿀 수도 없다(등록 여부만 보임). 실전 모드는 DB 제약으로 막혀 있다.

주의
- `.env`의 `DATA_KEY`를 잃거나 바꾸면 저장된 앱키를 다시 읽을 수 없다. `.env`를 따로 안전하게 백업한다(메신저·클라우드 공유 금지).
- 이용료·수익 배분을 받지 않는다(자본시장법상 "영업" 해당 소지). 실전 전 법률 상담 권장.
- 자동매매 스위치는 저장만 된다. 주문 엔진은 아직 없다.

보안 검토 (독립 에이전트, 2026-09-25) — 고친 것: 관리자 비밀번호 초기화로 남의 자동매매를 켜는 문제(High, 초기화 시 동의·키·자동매매 삭제), 로그인 잠금 동시요청 우회·관리자 잠금 공격((아이디,IP)+IP 한도, 스레드 잠금), 관리자끼리 초기화 금지, 입력 길이 제한, 다른 사이트발 POST 차단, 보안 헤더(CSP·X-Frame-Options), DATA_KEY 시작 시 검사.
2차 수정: `DATA_KEY`를 `.env`에서 Windows 자격 증명 관리자(`alpha-trader`/`DATA_KEY`)로 이동 → 암호화 DB와 분리, 로그아웃하면 모든 기기 세션 종료, 임시 비밀번호 72시간 만료, 메모리 캐시 2,000개 제한·차트 기간 구간화.
남은 것: 서버 관리자는 DATA_KEY로 누구의 키든 복호화할 수 있다(동의 화면에 명시).

## PC 초기화 대비 백업 (`app/keybackup.py`)
- 묶는 것: DATA_KEY(자격 증명 관리자) + 계정 DB(`alpha.db`) + `.env`. 복구 비밀번호(12자 이상, scrypt)로 잠가 **구글 드라이브 `G:\내 드라이브\alpha-backup\`** 에 저장한다(드라이브 데스크톱 앱 자동 감지).
- 파일 안에는 평문이 없다(테스트로 확인). 복구 비밀번호는 PC 밖(휴대폰 메모 등)에 적어 둔다. 비밀번호를 잊으면 복구 불가.
- 지인 계정을 만들거나 키가 바뀐 뒤에 다시 백업한다(자동 백업 없음: 복구 비밀번호를 저장해 둘 수 없어서).
- 백업: `C:\Users\buffy\.venvs\alpha-trader\Scripts\python.exe -m app.keybackup export`
- 새 PC 복구: 저장소·venv 설치 → 구글 드라이브 앱 로그인 → `python -m app.keybackup import "G:\내 드라이브\alpha-backup\<파일>"`

## 모바일 화면 (860px 이하)
토스증권·Robinhood식 패턴 조사 후 적용: 하단 탭 5개(홈·관심·발견·뉴스·내 계정), 관심종목 리스트 행(이름 | 추세선 | 가격·등락), 시장 지표 가로 스와이프, 홈 상단 자동매매 상태 줄(스위치는 내 계정에만), 뉴스 게이트 '매수 보류'를 관심종목 행에 표시, 터치 영역 44~48px. 데스크톱 화면은 그대로.

## 에이전트 팀
조사 근거·모델 배정·PDF 중 불가 항목: [docs/AGENT_TEAM.md](docs/AGENT_TEAM.md)

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
