# 자동매매 개발·검증 에이전트 팀 (2026-09-25)

정의 파일: `.claude/agents/*.md` (프로젝트 범위, 9개). 공통 규칙은 `PREP.md`의 수용 기준을 따른다.

## 1. 근거 자료 (온라인 조사, 2026-09-25)

| 자료 | 가져온 역할·원칙 |
|---|---|
| TradingAgents (arXiv 2412.20138, github.com/TauricResearch/TradingAgents) | 분석가 → 강세/약세 토론 → 트레이더 → 리스크팀 |
| ai-hedge-fund (github.com/virattt/ai-hedge-fund) | Risk Manager가 Portfolio Manager 전에 비중 제한 |
| FinRobot (arXiv 2405.14767) | 숫자 계산은 결정적 코드, LLM은 서술만 |
| RD-Agent(Q) (Microsoft, github.com/microsoft/RD-Agent) | 가설 → 구현 → 백테스트 피드백 루프 |
| AlphaAgents (BlackRock, arXiv 2508.11152) | 구조화된 토론으로 환각 감소 |
| QuantAgent (arXiv 2509.09995) | 가격 규칙 에이전트 + RiskAgent |
| FinCon (NeurIPS 2024) | 관리자-분석가 계층, 이중 리스크 통제 |
| HARLF (arXiv 2507.18560) | PDF 2단계 원본 논문. 공개 GitHub 없음(논문 Colab만) |
| Bailey & López de Prado, Deflated Sharpe (SSRN 2460551), PBO/CSCV | 시도 횟수 보정, 과최적화 확률 |
| 미 연준 SR 11-7 | 모델 개발자와 검증자 분리, 지속 모니터링 |
| LLM 미래참조 편향 (arXiv 2512.23847) | 백테스트 뉴스 판정에 생성형 LLM 사용 주의 |

조사한 LLM 트레이딩 프레임워크는 대부분 백테스트 결과만 보고했고, 실전 검증이나 한국 시장 검증은 찾지 못했다. 그래서 이 팀은 "매매 판단 에이전트"가 아니라 "프로그램을 만들고 검증하는 팀"으로 구성했다.

## 2. 팀 구성과 모델

| 에이전트 | 역할 | 모델 | 이유 | 도구 |
|---|---|---|---|---|
| quant-researcher | PDF → 검증 가능한 명세·반증 조건 | fable | 가장 어려운 판단 | 읽기·웹 |
| data-engineer | 시세·공시·뉴스 수집, 시점 정합성 | sonnet | 일반 구현 | 편집 |
| strategy-developer | 1~3단계 규칙 코드·백테스트 실행 | fable | 시뮬레이션 (2026-09-25 사용자 지시) | 편집 |
| ml-engineer | DRL Super-Agent, FinBERT 게이트, ML 검증 | fable | 분석·시뮬레이션 (사용자 지시) | 편집 |
| execution-engineer | KIS 주문·속도 제한·킬 스위치 | opus | 금액·보안 | 편집 |
| risk-manager | 리스크 규칙 독립 검토 | fable | 금액, 독립성 | 읽기+실행만 |
| backtest-validator | DSR·PBO·워크포워드 판정 | fable | 신뢰성 핵심, 독립성 | 읽기+실행만 |
| compliance-security-reviewer | 한국 규제·키·대시보드 보안 | fable | 심층조사·보안 (사용자 지시) | 읽기·웹 |
| ops-monitor | 모의투자 일일 대사 | haiku | 단순 집계 | 읽기+실행만 |

모델 원칙(2026-09-25 사용자 지시): 이 프로젝트의 심층조사·분석·시뮬레이션은 fable 5.1. 단순 수집(data-engineer, sonnet)·주문 연결 코드(execution-engineer, opus)·운영 점검(ops-monitor, haiku)·디자인은 제외.
모델 호출 확인: fable·haiku는 이 세션에서 서브에이전트로 실제 호출 성공. sonnet은 이번 조사 에이전트 2개로 실행 성공. opus는 현재 메인 세션 모델.
검증자와 리스크 매니저는 편집 도구를 주지 않았다(직무 분리). 판정만 돌려주고 수정은 개발 에이전트가 한다.

## 3. 작업 흐름

```
quant-researcher(명세) → data-engineer / strategy-developer / ml-engineer(구현, pytest)
   → backtest-validator(판정) ─불합격→ 명세·구현으로 되돌림
   → risk-manager + compliance-security-reviewer(검토)
   → execution-engineer(모의투자 연결) → ops-monitor(일일 대사)
```

같은 파일을 두 에이전트가 동시에 고치지 않는다. 실계좌 전환은 사용자만 결정한다.

## 4. 도구 상태 (이 PC, alpha-trader venv)

| 도구 | 상태 | 확인 |
|---|---|---|
| python-kis 2.1.6 (`pykis`) | 기능 실행 확인(import) | KIS 키 없음 → 실제 호출 미확인 |
| opendartreader 0.3.3 + DART_API_KEY | 기능 실행 확인 | 2026-09-25 삼성전자 공시 2,769건 조회 성공. 전체 시장 당일 조회는 빈 응답이라 종목별 조회로 사용 |
| aiolimiter 1.3.0, skfolio 1.3.1, scipy 1.18.1 | import 확인 | |
| snunlp/KR-FinBert-SC | 기능 실행 확인 | "적자 기조 탈출, 흑자 전환" → positive 0.91. 라이선스 표기 없음 → 개인 연구용 |
| pykrx `get_market_fundamental` | 설정·인증 필요 | KRX 로그인 없으면 PER/PBR 빈 값 (직접 확인) |
| NAVER API HUB (뉴스·웹문서·오타변환·검색어 트렌드) | 기능 실행 확인 | 네이버 클라우드 API HUB 키. 주소 `naverapihub.apigw.ntruss.com`, 헤더 `X-NCP-APIGW-API-KEY-ID/KEY`. 뉴스 `/search/v1/news`, 트렌드 `/search-trend/v1/search` 200 응답 확인. 대시보드 뉴스는 네이버 우선 |
| `app/selection.py` 1단계 A 후보 | 기능 실행 확인 | 2025 사업보고서, 대상 2,528개 중 2,503개 재무 확보, 25초 |
| `app/news_gate.py` 4단계 뉴스 게이트 | 기능 실행 확인 | FinBERT 단독은 호재를 악재로 오판(실측) → 치명적 사건 단어 + 비긍정일 때만 REJECT |
| KIS Code Assistant MCP | 기능 실행 확인 | 원인: uv가 설치한 Python 3.12 손상 → 재설치 후 initialize·tools/list 응답(도구 9개). 이 세션엔 재연결 안 됨, 새 세션부터 사용 |
| 프로젝트 에이전트 로딩 | 파일 작성 완료, 로딩 미확인 | 헤드리스 `claude -p`가 OAuth 만료로 실패. 새 세션 `/agents`에서 확인 |
| finrl, vectorbt, pypbo | 이번 범위 제외 | finrl 의존성 과다(sb3로 직접 구현), vectorbt 불필요, pypbo는 AGPL |

## 5. PDF 기능 중 못 하거나 조건이 붙는 것

| PDF 항목 | 판정 | 이유 / 필요한 것 |
|---|---|---|
| 1단계 B: 최근 5일 애널리스트 EPS 추정치 상향 | **KIS 실전 키로 앞으로는 가능, 과거 백테스트는 불가** | KIS `estimate-perform`(HHKST668300C0, 종목추정실적)을 매일 저장해 5일 변화 계산. `invest-opinion`(FHKST663300C0)은 증권사 의견·목표가 이력 조회 가능. 과거 추정치 이력은 유료 데이터 필요 |
| 1단계 B: "GA + LLM 탐색 수식" | 구현 가능, 내용 불명 | PDF에 수식이 없음. 연구자가 새로 정의해야 함 |
| 1단계 A: PER·PBR 최하위 | **해결 가능** | DART 재무제표(순이익·자본, 키 확보) ÷ 공시일 이후 시가총액(FDR)으로 직접 계산 → 과거 시점도 미래 참조 없이 가능. 보조: KRX 무료 회원 로그인(pykrx), KIS 현재가 per/pbr |
| 2단계 CBVR 3중 필터 | 구현 가능, 수치 정의 필요 | "가속도 임계값", "벡터 분석"의 수식이 PDF에 없음 |
| 2단계 HARLF DRL 26%·샤프 1.2 | 재구현만 가능 | 공개 코드 없음(논문 Colab). 성과 재현은 보장 못 함 |
| 3단계 VIX 급등 1/10 | 가능 | yfinance `^VIX`. "급등" 기준값은 정의 필요 |
| 3단계 한국 VKOSPI | **KIS 키로 가능** | KIS 업종코드 `0503` = VKOSPI(마스터 파일 확인). `inquire-index-daily-price`(FHPUP02120000). FDR·yfinance·네이버는 실패 확인 |
| 4단계 FNSPID 뉴스 | **한국 데이터 아님** | FNSPID는 미국 S&P500 뉴스. 한국은 Google 뉴스 RSS·DART 공시로 대체 |
| 4단계 실시간 한국 뉴스 피드 | 조건부 가능 | KIS `news-title`(FHKST01011800, 종합 시황/공시 제목), DART 공시 폴링(키 확보), 네이버 뉴스 검색 API(무료, 개발자 등록 필요) |
| 4단계 KIS 주문·WebSocket | 가능, 사용자 키 필요 | KIS 모의투자 앱키·계좌를 사용자가 발급 |
| 초당 15건 | 실전 기준 | 모의투자는 초당 2건 이하(공식 샘플 코드 기준) |
| CAGR 30%·MDD 13%·"절대 잃지 않는 매매" | **약속 불가** | 시뮬레이션 주장값. 실측으로만 판단. 손실 가능성은 항상 있음 |
| 한국 알고리즘 매매 규제 | 확인 필요 | 과다호가부담금 법안(2025-12 발의) 통과 여부, KRX 등록 의무를 원문으로 확인해야 함 |

## 6. 수용 기준
- 각 단계 코드는 pytest와 pyright를 통과하고, backtest-validator가 비용·누수·DSR·PBO를 점검한 뒤에만 다음 단계로 간다.
- 주문은 모의투자에서만, 속도 제한·킬 스위치 테스트를 통과한 뒤에만 연결한다.
- 보고서에는 "작성 / 로컬 실행 / 모의 운용 / 실전" 중 어디까지 확인했는지 적는다.
