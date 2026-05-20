# Crypto Futures Trading Bot

고승률 암호화폐 선물 자동매매 봇. 다중 기술적 분석 지표를 결합한 복합 시그널 전략과 철저한 리스크 관리 시스템을 갖추고 있습니다.

## 주요 기능

### 전략
- **Multi-Signal Strategy**: RSI + EMA 크로스오버 + MACD + Bollinger Bands + Volume + ADX를 가중치 스코어링으로 결합
- **Scalping Strategy**: 1m~5m 타임프레임용 단타 전략 (Stochastic RSI + BB + 캔들 패턴)
- 다이버전스 감지 (MACD/가격 발산)
- ADX 기반 트렌드/횡보장 자동 판별

### 리스크 관리
- 고정 비율 포지션 사이징 (기본 2%/거래)
- 손절매 (SL) / 익절 (TP) 자동 설정 (기본 R:R = 2:1)
- 트레일링 스탑 (수익 보호)
- 일일 최대 손실 한도 (기본 5%) → 자동 거래 중단
- 최대 낙폭 한도 (기본 10%) → 긴급 정지
- 연속 5패 시 쿨다운

### 백테스팅
- Walk-forward 백테스트 엔진
- 승률, ROI, 최대 낙폭, Sharpe Ratio, Profit Factor 등 상세 지표
- 다양한 타임프레임/전략 조합 테스트 가능

### 운영 모드
| 모드 | 설명 |
|------|------|
| `paper` | 모의 거래 (테스트넷 / 시뮬레이션) |
| `live` | 실거래 (실제 자금 사용) |
| `backtest` | 과거 데이터 기반 전략 검증 |

## 설치

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -e ".[dev]"
```

## 설정

```bash
# .env 파일 생성
cp .env.example .env
# .env 파일을 편집하여 API 키 입력
```

### 주요 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BOT_EXCHANGE_ID` | `binance` | 거래소 (`binance`, `bybit`) |
| `BOT_API_KEY` | - | 거래소 API 키 |
| `BOT_API_SECRET` | - | 거래소 API 시크릿 |
| `BOT_MODE` | `paper` | 운영 모드 |
| `BOT_SYMBOL` | `BTC/USDT:USDT` | 거래 페어 |
| `BOT_TIMEFRAME` | `15m` | 캔들 타임프레임 |
| `BOT_LEVERAGE` | `5` | 레버리지 배율 |
| `BOT_STOP_LOSS_PCT` | `0.015` | 손절 비율 (1.5%) |
| `BOT_TAKE_PROFIT_PCT` | `0.03` | 익절 비율 (3%) |
| `BOT_MIN_SIGNAL_SCORE` | `0.6` | 최소 시그널 스코어 |

## 사용법

### 백테스트 실행

```bash
# 기본 백테스트 (BTC/USDT, 15분봉, multi_signal 전략)
python main.py backtest

# 커스텀 설정
python main.py backtest --symbol BTC/USDT:USDT --timeframe 1h --strategy scalping --balance 50000

# 사용 가능한 옵션
python main.py backtest --help
```

### 모의 거래 (Paper Trading)

```bash
python main.py paper
```

### 실거래 (주의: 실제 자금 사용)

```bash
python main.py live
```

## 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --tb=short -v
```

## 린트 & 타입 체크

```bash
ruff check src/ tests/
mypy src/
```

## 프로젝트 구조

```
futures-trading-bot/
├── main.py                    # 진입점
├── src/
│   ├── core/
│   │   ├── config.py          # 설정 (pydantic-settings)
│   │   ├── models.py          # 도메인 모델
│   │   ├── exchange.py        # 거래소 어댑터 (ccxt)
│   │   ├── risk_manager.py    # 리스크 관리 엔진
│   │   ├── backtester.py      # 백테스팅 프레임워크
│   │   └── bot.py             # 메인 봇 오케스트레이터
│   ├── strategies/
│   │   ├── base.py            # 전략 인터페이스
│   │   ├── indicators.py      # 기술적 지표 계산
│   │   ├── multi_signal.py    # 복합 시그널 전략
│   │   └── scalping.py        # 스캘핑 전략
│   └── utils/
│       └── logger.py          # 로깅 설정
├── tests/                     # 테스트 코드
├── .env.example               # 환경변수 템플릿
└── pyproject.toml             # 프로젝트 설정
```

## 위험 경고

⚠️ **이 봇은 교육/연구 목적으로 제작되었습니다.** 실거래 사용 시 원금 손실 위험이 있습니다.
반드시 `paper` 모드와 백테스트로 충분히 검증한 후 소액으로 시작하세요.
