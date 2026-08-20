# 그럼 섹션별로 나눠서 작성하겠습니다. 먼저 기본 설정과 설정 클래스들입니다:
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta
import aiohttp
import yfinance as yf
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
import logging
from scipy import stats
import plotly.graph_objects as go
from textblob import TextBlob

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# API 키 검증
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")


@dataclass
class AnalysisConfig:
    """분석 파라미터 설정"""

    period_years: int = 5
    interval: str = "1wk"
    moving_averages: List[int] = None
    rsi_period: int = 14
    volume_ma_period: int = 20
    sentiment_lookback_days: int = 30  # 감정 분석 기간
    var_confidence_level: float = 0.95  # Value at Risk 신뢰수준
    cache_expiry_hours: int = 24  # 캐시 만료 시간

    def __post_init__(self):
        if self.moving_averages is None:
            self.moving_averages = [50, 200]


@dataclass
class RiskMetrics:
    """리스크 메트릭스 데이터 클래스"""

    value_at_risk: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    correlation_matrix: pd.DataFrame


@dataclass
class SentimentScore:
    """감정 분석 점수 데이터 클래스"""

    news_sentiment: float
    social_sentiment: float
    overall_score: float
    confidence: float


# 다음은 캐싱 시스템과 리스크 관리 클래스입니다:
class AnalysisCache:
    """캐싱 시스템"""

    def __init__(self, expiry_hours: int = 24):
        self.cache = {}
        self.expiry = timedelta(hours=expiry_hours)

    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.expiry:
                logger.info(f"캐시 히트: {key}")
                return data
        return None

    def set(self, key: str, data: Dict) -> None:
        self.cache[key] = (data, datetime.now())
        logger.info(f"캐시 저장: {key}")


class RiskManager:
    """리스크 관리 시스템"""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def calculate_var(self, returns: pd.Series) -> float:
        """Value at Risk 계산"""
        return abs(np.percentile(returns, (1 - self.config.var_confidence_level) * 100))

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """샤프 비율 계산"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """최대 낙폭 계산"""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return abs(drawdown.min())

    def analyze_risk(self, prices: pd.Series) -> RiskMetrics:
        """종합 리스크 분석"""
        returns = prices.pct_change().dropna()

        return RiskMetrics(
            value_at_risk=self.calculate_var(returns),
            sharpe_ratio=self.calculate_sharpe_ratio(returns),
            volatility=returns.std() * np.sqrt(252),
            max_drawdown=self.calculate_max_drawdown(prices),
            correlation_matrix=pd.DataFrame(),  # 추후 확장을 위한 빈 상관관계 매트릭스
        )


class MarketSentimentAnalyzer:
    """시장 감정 분석기"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cache = AnalysisCache(config.cache_expiry_hours)

    async def analyze_news_sentiment(self, ticker: str) -> float:
        """뉴스 기사 감정 분석"""
        # 여기에 실제 뉴스 API 연동 코드 추가 필요
        # 현재는 예시로 임의의 값 반환
        return 0.7

    async def analyze_social_sentiment(self, ticker: str) -> float:
        """소셜 미디어 감정 분석"""
        # 여기에 실제 소셜 미디어 API 연동 코드 추가 필요
        # 현재는 예시로 임의의 값 반환
        return 0.6

    async def get_sentiment(self, ticker: str) -> SentimentScore:
        """종합 감정 분석 점수 계산"""
        cache_key = f"sentiment_{ticker}"
        cached_result = self.cache.get(cache_key)

        if cached_result:
            return cached_result

        news_score = await self.analyze_news_sentiment(ticker)
        social_score = await self.analyze_social_sentiment(ticker)

        overall_score = news_score * 0.6 + social_score * 0.4  # 뉴스에 더 높은 가중치

        sentiment = SentimentScore(
            news_sentiment=news_score,
            social_sentiment=social_score,
            overall_score=overall_score,
            confidence=0.8,  # 신뢰도 점수
        )

        self.cache.set(cache_key, sentiment)
        return sentiment


# 다음은 기술적 분석 시스템입니다:
class TechnicalAnalyzer:
    """향상된 기술적 분석 시스템"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cache = AnalysisCache(config.cache_expiry_hours)

    def calculate_advanced_indicators(self, data: pd.DataFrame) -> Dict:
        """고급 기술적 지표 계산"""
        indicators = {}

        # RSI 계산
        try:
            indicators["rsi"] = ta.rsi(data["Close"], length=self.config.rsi_period)
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            indicators["rsi"] = pd.Series([float("nan")] * len(data), index=data.index)

        # MACD 계산
        try:
            macd = ta.macd(data["Close"])
            indicators["macd"] = macd["MACD_12_26_9"]
            indicators["macd_signal"] = macd["MACDs_12_26_9"]
        except Exception as e:
            logger.error(f"Failed to calculate MACD: {e}")
            indicators["macd"] = pd.Series([float("nan")] * len(data), index=data.index)
            indicators["macd_signal"] = pd.Series(
                [float("nan")] * len(data), index=data.index
            )

        # 볼린저 밴드 계산 (length=20로 지정)
        try:
            bollinger = ta.bbands(data["Close"], length=20)
            print("Bollinger Bands keys:", list(bollinger.keys()))
            if "BBU_20_2.0" in bollinger and "BBL_20_2.0" in bollinger:
                indicators["bb_upper"] = bollinger["BBU_20_2.0"]
                indicators["bb_lower"] = bollinger["BBL_20_2.0"]
            else:
                raise KeyError(
                    "Missing Bollinger Bands keys (BBU_20_2.0 or BBL_20_2.0)"
                )
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {e}")
            indicators["bb_upper"] = pd.Series(
                [float("nan")] * len(data), index=data.index
            )
            indicators["bb_lower"] = pd.Series(
                [float("nan")] * len(data), index=data.index
            )

        # 이동평균선 계산
        for ma in self.config.moving_averages:
            try:
                indicators[f"ma_{ma}"] = ta.sma(data["Close"], length=ma)
            except Exception as e:
                logger.error(f"Failed to calculate moving average for {ma}: {e}")
                indicators[f"ma_{ma}"] = pd.Series(
                    [float("nan")] * len(data), index=data.index
                )

        # 볼륨 지표 계산
        try:
            indicators["volume_sma"] = ta.sma(
                data["Volume"], length=self.config.volume_ma_period
            )
        except Exception as e:
            logger.error(f"Failed to calculate volume SMA: {e}")
            indicators["volume_sma"] = pd.Series(
                [float("nan")] * len(data), index=data.index
            )

        return indicators

    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict
    ) -> Dict[str, float]:
        """매매 신호 생성"""
        signals = {
            "rsi_signal": 0,  # -1 (매도), 0 (중립), 1 (매수)
            "macd_signal": 0,  # -1 (매도), 0 (중립), 1 (매수)
            "ma_signal": 0,  # -1 (하락추세), 0 (중립), 1 (상승추세)
            "volume_signal": 0,  # -1 (거래량 감소), 0 (중립), 1 (거래량 증가)
            "overall_signal": 0,  # 종합 신호
        }

        # RSI 신호 계산
        try:
            last_rsi = indicators["rsi"].iloc[-1]
            if last_rsi < 30:
                signals["rsi_signal"] = 1  # 과매도 -> 매수
            elif last_rsi > 70:
                signals["rsi_signal"] = -1  # 과매수 -> 매도
        except Exception as e:
            logger.error(f"Error calculating RSI signal: {e}")

        # MACD 신호 계산
        try:
            if indicators["macd"].iloc[-1] > indicators["macd_signal"].iloc[-1]:
                signals["macd_signal"] = 1
            else:
                signals["macd_signal"] = -1
        except Exception as e:
            logger.error(f"Error calculating MACD signal: {e}")

        # 이동평균선 신호 계산
        try:
            if "ma_50" in indicators and "ma_200" in indicators:
                ma_50 = indicators["ma_50"].iloc[-1]
                ma_200 = indicators["ma_200"].iloc[-1]
                current_price = data["Close"].iloc[-1]
                if current_price > ma_50 > ma_200:
                    signals["ma_signal"] = 1
                elif current_price < ma_50 < ma_200:
                    signals["ma_signal"] = -1
            else:
                raise KeyError("ma_50 or ma_200 not found in indicators")
        except Exception as e:
            logger.error(f"Error calculating moving average signal: {e}")

        # 거래량 신호 계산
        try:
            if data["Volume"].iloc[-1] > indicators["volume_sma"].iloc[-1] * 1.5:
                signals["volume_signal"] = 1
            elif data["Volume"].iloc[-1] < indicators["volume_sma"].iloc[-1] * 0.5:
                signals["volume_signal"] = -1
        except Exception as e:
            logger.error(f"Error calculating volume signal: {e}")

        # 종합 신호 계산 (가중치 적용)
        weights = {
            "rsi_signal": 0.3,
            "macd_signal": 0.3,
            "ma_signal": 0.25,
            "volume_signal": 0.15,
        }
        signals["overall_signal"] = sum(signals[k] * weights[k] for k in weights)

        return signals

    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict
    ) -> Dict[str, float]:
        """매매 신호 생성"""
        signals = {
            "rsi_signal": 0,  # -1 (매도), 0 (중립), 1 (매수)
            "macd_signal": 0,  # -1 (매도), 0 (중립), 1 (매수)
            "ma_signal": 0,  # -1 (하락추세), 0 (중립), 1 (상승추세)
            "volume_signal": 0,  # -1 (거래량 감소), 0 (중립), 1 (거래량 증가)
            "overall_signal": 0,  # 종합 신호
        }

        # RSI 신호 계산
        try:
            last_rsi = indicators["rsi"].iloc[-1]
            if last_rsi < 30:
                signals["rsi_signal"] = 1
            elif last_rsi > 70:
                signals["rsi_signal"] = -1
        except Exception as e:
            logger.error(f"Error calculating RSI signal: {e}")

        # MACD 신호 계산
        try:
            if indicators["macd"].iloc[-1] > indicators["macd_signal"].iloc[-1]:
                signals["macd_signal"] = 1
            else:
                signals["macd_signal"] = -1
        except Exception as e:
            logger.error(f"Error calculating MACD signal: {e}")

        # 이동평균선 신호 계산
        try:
            if "ma_50" in indicators and "ma_200" in indicators:
                ma_50 = indicators["ma_50"].iloc[-1]
                ma_200 = indicators["ma_200"].iloc[-1]
                current_price = data["Close"].iloc[-1]
                if current_price > ma_50 > ma_200:
                    signals["ma_signal"] = 1
                elif current_price < ma_50 < ma_200:
                    signals["ma_signal"] = -1
            else:
                raise KeyError("ma_50 or ma_200 not found in indicators")
        except Exception as e:
            logger.error(f"Error calculating moving average signal: {e}")

        # 거래량 신호 계산
        try:
            if data["Volume"].iloc[-1] > indicators["volume_sma"].iloc[-1] * 1.5:
                signals["volume_signal"] = 1
            elif data["Volume"].iloc[-1] < indicators["volume_sma"].iloc[-1] * 0.5:
                signals["volume_signal"] = -1
        except Exception as e:
            logger.error(f"Error calculating volume signal: {e}")

        # 종합 신호 계산 (가중치 적용)
        weights = {
            "rsi_signal": 0.3,
            "macd_signal": 0.3,
            "ma_signal": 0.25,
            "volume_signal": 0.15,
        }
        signals["overall_signal"] = sum(signals[k] * weights[k] for k in weights)

        return signals


class BacktestEngine:
    """백테스팅 엔진"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.technical_analyzer = TechnicalAnalyzer(config)

    def run_backtest(
        self, data: pd.DataFrame, initial_capital: float = 100000.0
    ) -> Dict:
        """백테스트 실행"""
        results = {
            "trades": [],
            "portfolio_value": [],
            "returns": [],
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

        # 기술적 지표 계산
        indicators = self.technical_analyzer.calculate_advanced_indicators(data)

        capital = initial_capital
        position = 0  # 0: 없음, 1: 롱, -1: 숏
        entry_price = 0

        for i in range(len(data)):
            date = data.index[i]
            current_price = data["Close"].iloc[i]

            # 매매 신호 생성
            signals = self.technical_analyzer.generate_signals(
                data.iloc[: i + 1], {k: v.iloc[: i + 1] for k, v in indicators.items()}
            )

            # 포지션 진입/청산 로직
            if position == 0:  # 포지션 없음
                if signals["overall_signal"] > 0.5:  # 강한 매수 신호
                    position = 1
                    entry_price = current_price
                    results["trades"].append(
                        {"date": date, "type": "buy", "price": current_price}
                    )
                elif signals["overall_signal"] < -0.5:  # 강한 매도 신호
                    position = -1
                    entry_price = current_price
                    results["trades"].append(
                        {"date": date, "type": "sell", "price": current_price}
                    )
            else:  # 포지션 있음
                # 청산 조건
                if (position == 1 and signals["overall_signal"] < -0.3) or (
                    position == -1 and signals["overall_signal"] > 0.3
                ):
                    pnl = (current_price - entry_price) * position
                    capital += pnl
                    position = 0
                    results["trades"].append(
                        {
                            "date": date,
                            "type": "close",
                            "price": current_price,
                            "pnl": pnl,
                        }
                    )

            # 포트폴리오 가치 추적
            portfolio_value = capital
            if position != 0:
                portfolio_value += (current_price - entry_price) * position
            results["portfolio_value"].append(portfolio_value)

        # 성과 지표 계산
        if results["trades"]:
            winning_trades = len([t for t in results["trades"] if t.get("pnl", 0) > 0])
            results["win_rate"] = winning_trades / len(results["trades"])

            total_profit = sum(
                [t["pnl"] for t in results["trades"] if t.get("pnl", 0) > 0]
            )
            total_loss = abs(
                sum([t["pnl"] for t in results["trades"] if t.get("pnl", 0) < 0])
            )
            results["profit_factor"] = (
                total_profit / total_loss if total_loss > 0 else float("inf")
            )

        return results


# 다음은 메인 분석 클래스와 Tasks 클래스를 작성하겠습니다.
class AdvancedAnalysis:
    """향상된 종합 분석 시스템"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.technical_analyzer = TechnicalAnalyzer(config)
        self.risk_manager = RiskManager(config)
        self.sentiment_analyzer = MarketSentimentAnalyzer(config)
        self.backtest_engine = BacktestEngine(config)
        self.cache = AnalysisCache(config.cache_expiry_hours)

    async def run_complete_analysis(self, ticker: str) -> Dict:
        """종합 분석 실행"""
        # 데이터 가져오기
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.period_years * 365)
        data = stock.history(
            start=start_date, end=end_date, interval=self.config.interval
        )

        # 각종 분석 실행
        technical_indicators = self.technical_analyzer.calculate_advanced_indicators(
            data
        )
        signals = self.technical_analyzer.generate_signals(data, technical_indicators)
        risk_metrics = self.risk_manager.analyze_risk(data["Close"])
        sentiment = await self.sentiment_analyzer.get_sentiment(ticker)
        backtest_results = self.backtest_engine.run_backtest(data)

        # 최종 투자 결정
        decision = self._make_final_decision(
            signals=signals,
            risk_metrics=risk_metrics,
            sentiment=sentiment,
            backtest_results=backtest_results,
        )

        return {
            "decision": decision,
            "technical_signals": signals,
            "risk_metrics": risk_metrics,
            "sentiment": sentiment,
            "backtest_results": backtest_results,
            "current_price": data["Close"].iloc[-1],
            "analysis_date": datetime.now(),
        }

    def _make_final_decision(
        self,
        signals: Dict,
        risk_metrics: RiskMetrics,
        sentiment: SentimentScore,
        backtest_results: Dict,
    ) -> str:
        """최종 투자 결정 (반드시 BUY/HOLD/SELL 중 하나를 반환)"""
        # 점수 기반 시스템 (각 요소에 가중치 부여)
        score = 0.0

        # 기술적 신호 (40% 가중치)
        score += signals["overall_signal"] * 0.4

        # 리스크 메트릭스 (25% 가중치)
        risk_score = 0.0
        if risk_metrics.sharpe_ratio > 1.0:
            risk_score += 0.5
        if risk_metrics.value_at_risk < 0.05:  # VaR가 5% 미만
            risk_score += 0.5
        score += risk_score * 0.25

        # 감정 분석 (20% 가중치)
        sentiment_score = (sentiment.overall_score - 0.5) * 2  # -1에서 1 사이로 정규화
        score += sentiment_score * 0.2

        # 백테스트 결과 (15% 가중치)
        backtest_score = 0.0
        if backtest_results["win_rate"] > 0.5:
            backtest_score += 0.5
        if backtest_results["profit_factor"] > 1.5:
            backtest_score += 0.5
        score += backtest_score * 0.15

        # 최종 결정
        if score > 0.3:
            return "BUY"
        elif score < -0.3:
            return "SELL"
        else:
            return "HOLD"


class Tasks:
    """향상된 작업 정의"""

    def __init__(self):
        pass

    def financial_analysis(self, agent):
        return Task(
            description=f"""Perform detailed financial analysis for ticker {{company}}.
            1. Calculate key financial ratios
            2. Analyze growth trends
            3. Evaluate profitability metrics
            4. Assess debt levels and coverage ratios
            5. Compare with industry averages
            YOU MUST PROVIDE SPECIFIC NUMBERS AND EXACT VALUES.""",
            agent=agent,
            expected_output="""A financial analysis report with EXACT numbers:
                1. Profitability Ratios
                2. Growth Metrics
                3. Debt Ratios
                4. Cash Flow Analysis
                5. Clear BUY/HOLD/SELL recommendation""",
        )

    def research(self, agent):
        return Task(
            description="""Research the company with ticker {company} over the past 5 years. 
            1. Analyze major news events and their impact
            2. Track management changes and strategic shifts
            3. Evaluate competitive position changes
            4. Assess market sentiment trends
            5. Review analyst coverage and ratings history
            YOU MUST PROVIDE SPECIFIC NUMBERS AND EXACT VALUES.""",
            agent=agent,
            expected_output="""A comprehensive research report including:
                1. Timeline of major events with specific dates and price impacts
                2. Management changes with exact dates and subsequent performance
                3. Market share data with precise percentages
                4. Sentiment trends with quantified metrics
                5. Analyst ratings distribution with exact counts""",
        )

    def technical_analysis(self, agent):
        return Task(
            description="""Perform detailed technical analysis for ticker {company}.
            1. Calculate EXACT support/resistance levels
            2. Measure trend strength with ADX
            3. Analyze volume patterns with specific ratios
            4. Calculate precise RSI, MACD values
            5. Identify exact price targets
            YOU MUST PROVIDE SPECIFIC NUMBERS AND EXACT VALUES.""",
            agent=agent,
            expected_output="""A technical analysis report with EXACT numbers:
                1. Current price: XX.XX
                2. Support levels: XX.XX, XX.XX
                3. Resistance levels: XX.XX, XX.XX
                4. RSI: XX.XX
                5. MACD: XX.XX
                6. Volume ratio: XX.XX
                7. Price targets: XX.XX (upside), XX.XX (downside)
                8. Clear BULLISH/BEARISH/NEUTRAL rating""",
        )

    def risk_analysis(self, agent):
        return Task(
            description="""Perform comprehensive risk analysis for ticker {company}.
            1. Calculate Value at Risk (VaR)
            2. Measure Beta and volatility
            3. Analyze drawdown patterns
            4. Assess liquidity risks
            5. Evaluate correlation with market
            PROVIDE EXACT NUMBERS AND CONFIDENCE INTERVALS.""",
            agent=agent,
            expected_output="""A risk analysis report with precise metrics:
                1. VaR (95% confidence): XX.XX%
                2. Beta: XX.XX
                3. Historical volatility: XX.XX%
                4. Maximum drawdown: XX.XX%
                5. Risk rating: HIGH/MEDIUM/LOW""",
        )

    def sentiment_analysis(self, agent):
        return Task(
            description="""Analyze market sentiment for {company}.
            1. News sentiment scoring
            2. Social media sentiment analysis
            3. Insider trading patterns
            4. Institutional investor positions
            5. Options market sentiment
            PROVIDE EXACT METRICS AND TRENDS.""",
            agent=agent,
            expected_output="""A sentiment analysis report with metrics:
                1. News sentiment score: XX.XX/100
                2. Social sentiment: XX.XX%
                3. Insider confidence index: XX.XX
                4. Institutional holdings change: XX.XX%
                5. Overall sentiment: POSITIVE/NEUTRAL/NEGATIVE""",
        )

    def investment_recommendation(self, agent, context_tasks):
        return Task(
            description="""Synthesize all analysis results and provide final investment recommendation.
            Consider:
            1. Financial metrics and valuation
            2. Technical indicators and price targets
            3. Risk metrics and market conditions
            4. Sentiment indicators and momentum
            5. Research findings and company outlook
            PROVIDE CLEAR ACTIONABLE RECOMMENDATION.""",
            agent=agent,
            context_tasks=context_tasks,  # 이전 분석 결과들을 참조하기 위해 필요
            expected_output="""A comprehensive investment recommendation:
                1. Final Rating: STRONG BUY/BUY/HOLD/SELL/STRONG SELL
                2. Target Price: XX.XX
                3. Stop Loss: XX.XX
                4. Position Size: XX.XX%
                5. Key Risks and Catalysts
                6. Investment Timeline: SHORT/MEDIUM/LONG TERM""",
        )


#   다음은 Agent 클래스와 메인 실행 코드를 작성하겠습니다.


class AdvancedAgents:
    """향상된 에이전트 정의"""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def financial_analyst(self) -> Agent:
        return Agent(
            role="Financial Analyst",
            goal="Provide precise financial analysis with exact numbers and clear recommendations",
            backstory="""Seasoned financial analyst with 15 years of experience in equity research.
                     Known for detailed analysis and accurate price predictions.""",
            tools=[],
            verbose=True,
        )

    def technical_analyst(self) -> Agent:
        return Agent(
            role="Technical Analysis Expert",
            goal="Deliver precise technical analysis with specific price levels and clear signals",
            backstory="""Expert technical analyst with 20 years of experience in market analysis.
                     Specialized in pattern recognition and trend analysis.""",
            tools=[],
            verbose=True,
        )

    def risk_analyst(self) -> Agent:
        return Agent(
            role="Risk Management Expert",
            goal="Analyze and quantify all potential risks with specific metrics",
            backstory="""Experienced risk analyst with background in quantitative analysis.
                     Expert in risk metrics and portfolio optimization.""",
            tools=[],
            verbose=True,
        )

    def sentiment_analyst(self) -> Agent:
        return Agent(
            role="Market Sentiment Analyst",
            goal="Analyze market sentiment with precise metrics and clear interpretation",
            backstory="""Expert in market psychology and sentiment analysis.
                     Specialized in social media and news sentiment quantification.""",
            tools=[],
            verbose=True,
        )

    def hedge_fund_manager(self) -> Agent:
        return Agent(
            role="Hedge Fund Manager",
            goal="Make final investment decisions with clear BUY/SELL/HOLD recommendations",
            backstory="""Veteran hedge fund manager with 25 years of experience.
                     Known for making decisive and accurate investment calls.""",
            tools=[],
            verbose=True,
        )


async def create_analysis_crew(
    ticker: str, config: AnalysisConfig
) -> Tuple[Dict, Path]:
    """
    향상된 분석 크루 생성 및 실행
    This function is now async and awaits all async operations.
    """
    try:
        # 에이전트와 태스크 초기화
        agents = AdvancedAgents(config)
        tasks = Tasks()
        analysis = AdvancedAnalysis(config)

        # 모든 에이전트 생성
        financial_analyst = agents.financial_analyst()
        technical_analyst = agents.technical_analyst()
        risk_analyst = agents.risk_analyst()
        sentiment_analyst = agents.sentiment_analyst()
        hedge_fund_manager = agents.hedge_fund_manager()

        # 태스크 정의
        financial_task = tasks.financial_analysis(financial_analyst)
        technical_task = tasks.technical_analysis(technical_analyst)
        risk_task = tasks.risk_analysis(risk_analyst)
        sentiment_task = tasks.sentiment_analysis(sentiment_analyst)

        # 최종 투자 추천 태스크
        recommend_task = tasks.investment_recommendation(
            hedge_fund_manager,
            context_tasks=[financial_task, technical_task, risk_task, sentiment_task],
        )

        # Crew 실행 (assuming crew.kickoff() is a synchronous call; if it's async, await it!)
        crew = Crew(
            agents=[
                financial_analyst,
                technical_analyst,
                risk_analyst,
                sentiment_analyst,
                hedge_fund_manager,
            ],
            tasks=[
                financial_task,
                technical_task,
                risk_task,
                sentiment_task,
                recommend_task,
            ],
            verbose=True,
            process=Process.sequential,
            manager_llm=ChatOpenAI(
                model_name="o3-mini",
                temperature=0.4,
                api_key=OPENAI_API_KEY,
            ),
        )

        # If crew.kickoff is synchronous, call it directly:
        crew_result = crew.kickoff(inputs={"company": ticker})

        # Await the asynchronous analysis
        analysis_result = await analysis.run_complete_analysis(ticker)

        # Combine results
        final_result = {
            "crew_analysis": crew_result,
            "quantitative_analysis": analysis_result,
            "final_decision": analysis_result["decision"],
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save and return the report
        report_file = save_analysis_to_markdown(final_result, ticker)
        return final_result, report_file

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


# 마지막으로 보고서 저장 함수도 개선된 버전으로 작성하겠습니다.
def save_analysis_to_markdown(result: Dict, ticker: str) -> Path:
    """향상된 분석 결과 저장 함수"""
    # 저장 디렉토리 생성
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = reports_dir / f"{ticker}_analysis_{timestamp}.md"

    # 결과에서 필요한 데이터 추출
    decision = result["final_decision"]
    quant_analysis = result["quantitative_analysis"]
    crew_analysis = result["crew_analysis"]

    # 마크다운 형식으로 내용 구성
    content = f"""# Investment Analysis Report for {ticker}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
🎯 **Final Recommendation: {decision}**

### Key Metrics
- Current Price: {quant_analysis['current_price']:,.2f} KRW
- Risk Level: {'High' if quant_analysis['risk_metrics'].value_at_risk > 0.05 else 'Moderate' if quant_analysis['risk_metrics'].value_at_risk > 0.03 else 'Low'}
- Market Sentiment: {quant_analysis['sentiment'].overall_score:.2f}/1.00
- Win Rate (Backtest): {quant_analysis['backtest_results']['win_rate']:.1%}

## Technical Analysis
- RSI: {quant_analysis['technical_signals'].get('rsi_signal', 'N/A')}
- MACD Signal: {quant_analysis['technical_signals'].get('macd_signal', 'N/A')}
- Volume Signal: {quant_analysis['technical_signals'].get('volume_signal', 'N/A')}
- Overall Technical Signal: {quant_analysis['technical_signals'].get('overall_signal', 'N/A')}

## Risk Analysis
- Value at Risk (95%): {quant_analysis['risk_metrics'].value_at_risk:.2%}
- Sharpe Ratio: {quant_analysis['risk_metrics'].sharpe_ratio:.2f}
- Maximum Drawdown: {quant_analysis['risk_metrics'].max_drawdown:.2%}
- Volatility (Annual): {quant_analysis['risk_metrics'].volatility:.2%}

## Market Sentiment
- News Sentiment: {quant_analysis['sentiment'].news_sentiment:.2f}/1.00
- Social Media Sentiment: {quant_analysis['sentiment'].social_sentiment:.2f}/1.00
- Overall Sentiment: {quant_analysis['sentiment'].overall_score:.2f}/1.00
- Confidence Level: {quant_analysis['sentiment'].confidence:.2f}/1.00

## Backtest Results
- Total Trades: {len(quant_analysis['backtest_results']['trades'])}
- Win Rate: {quant_analysis['backtest_results']['win_rate']:.1%}
- Profit Factor: {quant_analysis['backtest_results']['profit_factor']:.2f}

## Qualitative Analysis
{crew_analysis}

## Investment Thesis
1. Technical Factors: {_get_signal_description(quant_analysis['technical_signals']['overall_signal'])}
2. Risk Assessment: {_get_risk_description(quant_analysis['risk_metrics'])}
3. Market Sentiment: {_get_sentiment_description(quant_analysis['sentiment'])}

## Action Plan
- Entry Strategy: {_get_entry_strategy(decision, quant_analysis)}
- Exit Strategy: {_get_exit_strategy(decision, quant_analysis)}
- Position Sizing: {_get_position_sizing(quant_analysis['risk_metrics'])}

---
*This report was generated automatically by the Advanced Stock Analysis System*
"""

    # 파일 저장
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename


def _get_signal_description(signal: float) -> str:
    """기술적 신호 설명 생성"""
    if signal > 0.5:
        return "Strong bullish signals from technical indicators"
    elif signal > 0:
        return "Moderately bullish technical signals"
    elif signal > -0.5:
        return "Slightly bearish technical signals"
    else:
        return "Strong bearish signals from technical indicators"


def _get_risk_description(metrics: RiskMetrics) -> str:
    """리스크 평가 설명 생성"""
    if metrics.sharpe_ratio > 1.5:
        risk_level = "Favorable"
    elif metrics.sharpe_ratio > 1:
        risk_level = "Acceptable"
    else:
        risk_level = "Concerning"

    return f"Risk-reward profile is {risk_level} with Sharpe ratio of {metrics.sharpe_ratio:.2f}"


def _get_sentiment_description(sentiment: SentimentScore) -> str:
    """시장 심리 설명 생성"""
    if sentiment.overall_score > 0.7:
        return "Extremely positive market sentiment"
    elif sentiment.overall_score > 0.5:
        return "Moderately positive sentiment"
    elif sentiment.overall_score > 0.3:
        return "Slightly positive to neutral sentiment"
    else:
        return "Negative market sentiment"


def _get_entry_strategy(decision: str, analysis: Dict) -> str:
    """진입 전략 생성"""
    if decision == "BUY":
        return f"Consider entering at current price or wait for pullback to {analysis['technical_signals'].get('support_level', 'nearest support level')}"
    elif decision == "SELL":
        return "No new positions recommended"
    else:
        return "Wait for clearer signals before entering new positions"


def _get_exit_strategy(decision: str, analysis: Dict) -> str:
    """청산 전략 생성"""
    if decision == "BUY":
        return f"Set stop-loss at {analysis['technical_signals'].get('stop_loss', 'recent low')} and take profit at {analysis['technical_signals'].get('target', 'resistance level')}"
    elif decision == "SELL":
        return "Consider closing existing positions"
    else:
        return "Maintain existing positions with trailing stops"


def _get_position_sizing(risk_metrics: RiskMetrics) -> str:
    """포지션 사이징 추천"""
    if risk_metrics.sharpe_ratio > 1.5 and risk_metrics.value_at_risk < 0.03:
        return "Consider standard position size (5% of portfolio)"
    elif risk_metrics.sharpe_ratio > 1:
        return "Reduce position size to 3% of portfolio"
    else:
        return "Limit position to 2% of portfolio due to elevated risk"


if __name__ == "__main__":
    try:
        # 분석 설정
        config = AnalysisConfig(
            period_years=5,
            interval="1wk",
            moving_averages=[50, 100, 200],
            rsi_period=14,
            volume_ma_period=20,
            sentiment_lookback_days=30,
            var_confidence_level=0.95,
            cache_expiry_hours=24,
        )

        # Run the async crew creation
        result, report_file = asyncio.run(create_analysis_crew("005930.KS", config))

        # 결과 출력
        print(f"\n최종 투자 추천: {result['final_decision']}")
        print(f"분석 보고서 저장 위치: {report_file}")
        print("\n정량적 분석 결과:")
        print(f"기술적 신호: {result['quantitative_analysis']['technical_signals']}")
        print(f"리스크 메트릭스: {result['quantitative_analysis']['risk_metrics']}")
        print(f"시장 심리: {result['quantitative_analysis']['sentiment']}")

    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
