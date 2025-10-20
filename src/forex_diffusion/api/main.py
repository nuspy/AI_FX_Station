"""
ForexGPT API Service

FastAPI endpoints for:
- Sentiment analysis data
- Economic calendar events
- Model predictions
- System health

Run with: uvicorn forex_diffusion.api.main:app --reload --port 8000
"""
from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Import services
try:
    from ..services.sentiment import SentimentService
    from ..services.event_calendar import EventCalendarService
    from ..services.db_service import DBService
except ImportError:
    # Fallback for standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from forex_diffusion.services.sentiment import SentimentService
    from forex_diffusion.services.event_calendar import EventCalendarService
    from forex_diffusion.services.db_service import DBService


# ============================================================================
# Pydantic Models
# ============================================================================

class SentimentMetrics(BaseModel):
    """Latest sentiment metrics for a symbol"""
    symbol: str
    timestamp: str
    overall_score: float = Field(..., ge=-1.0, le=1.0)
    news_score: float = Field(..., ge=-1.0, le=1.0)
    social_score: float = Field(..., ge=-1.0, le=1.0)
    analyst_score: float = Field(..., ge=-1.0, le=1.0)
    volume_weighted_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources_count: int
    last_updated: str


class SentimentHistoryPoint(BaseModel):
    """Single point in sentiment history"""
    timestamp: str
    overall_score: float
    news_score: float
    social_score: float
    analyst_score: float


class SentimentSignal(BaseModel):
    """Trading signal derived from sentiment"""
    symbol: str
    timestamp: str
    signal: str = Field(..., regex="^(BUY|SELL|NEUTRAL)$")
    strength: float = Field(..., ge=0.0, le=1.0)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    trend: str = Field(..., regex="^(BULLISH|BEARISH|NEUTRAL)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class CalendarEvent(BaseModel):
    """Economic calendar event"""
    event_id: str
    title: str
    country: str
    timestamp: str
    impact: str = Field(..., regex="^(LOW|MEDIUM|HIGH)$")
    currency: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    affected_symbols: List[str]


class RiskScore(BaseModel):
    """Risk score for a symbol based on upcoming events"""
    symbol: str
    timestamp: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    upcoming_events_count: int
    high_impact_events_count: int
    next_event_hours: Optional[float] = None
    events: List[CalendarEvent]


class HealthStatus(BaseModel):
    """API health status"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    services: Dict[str, str]


class PredictionRequest(BaseModel):
    """Request for model prediction"""
    symbol: str = Field(..., example="EURUSD")
    timeframe: str = Field(..., example="5m")
    features: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    """Model prediction response"""
    symbol: str
    timeframe: str
    prediction: float
    confidence: float
    timestamp: str
    model_version: str


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ForexGPT API",
    description="Real-time forex trading intelligence API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.start_time = datetime.now()
        self.sentiment_service: Optional[SentimentService] = None
        self.calendar_service: Optional[EventCalendarService] = None
        self.db_service: Optional[DBService] = None
        self.initialized = False

state = AppState()


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting ForexGPT API...")

    try:
        # Initialize services
        state.db_service = DBService()
        state.sentiment_service = SentimentService()
        state.calendar_service = EventCalendarService()

        state.initialized = True
        logger.info("ForexGPT API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        state.initialized = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ForexGPT API...")


# ============================================================================
# Dependency Injection
# ============================================================================

def get_sentiment_service() -> SentimentService:
    """Get sentiment service instance"""
    if not state.initialized or state.sentiment_service is None:
        raise HTTPException(status_code=503, detail="Sentiment service not available")
    return state.sentiment_service


def get_calendar_service() -> EventCalendarService:
    """Get calendar service instance"""
    if not state.initialized or state.calendar_service is None:
        raise HTTPException(status_code=503, detail="Calendar service not available")
    return state.calendar_service


# ============================================================================
# Root & Health Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ForexGPT API",
        "version": "1.0.0",
        "status": "running" if state.initialized else "initializing",
        "docs": "/api/docs",
    }


@app.get("/api/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - state.start_time).total_seconds()

    services_status = {
        "sentiment": "ok" if state.sentiment_service else "unavailable",
        "calendar": "ok" if state.calendar_service else "unavailable",
        "database": "ok" if state.db_service else "unavailable",
    }

    overall_status = "healthy" if state.initialized else "degraded"

    return HealthStatus(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=uptime,
        services=services_status,
    )


# ============================================================================
# Sentiment Endpoints
# ============================================================================

@app.get("/api/sentiment/{symbol}", response_model=SentimentMetrics)
async def get_sentiment(
    symbol: str,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
):
    """
    Get latest sentiment metrics for a symbol.

    Args:
        symbol: Trading pair symbol (e.g., EURUSD, GBPUSD)

    Returns:
        Latest sentiment metrics with scores from multiple sources
    """
    try:
        # Get latest sentiment data
        sentiment_data = sentiment_service.get_latest_sentiment(symbol)

        if sentiment_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data found for {symbol}"
            )

        return SentimentMetrics(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            overall_score=sentiment_data.get("overall_score", 0.0),
            news_score=sentiment_data.get("news_score", 0.0),
            social_score=sentiment_data.get("social_score", 0.0),
            analyst_score=sentiment_data.get("analyst_score", 0.0),
            volume_weighted_score=sentiment_data.get("volume_weighted_score", 0.0),
            confidence=sentiment_data.get("confidence", 0.5),
            sources_count=sentiment_data.get("sources_count", 0),
            last_updated=sentiment_data.get("last_updated", datetime.now().isoformat()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sentiment/{symbol}/history", response_model=List[SentimentHistoryPoint])
async def get_sentiment_history(
    symbol: str,
    hours: int = Query(24, ge=1, le=720, description="Hours of history to fetch"),
    sentiment_service: SentimentService = Depends(get_sentiment_service),
):
    """
    Get historical sentiment data for a symbol.

    Args:
        symbol: Trading pair symbol
        hours: Number of hours of history (1-720, default 24)

    Returns:
        List of sentiment data points over time
    """
    try:
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Get historical data
        history = sentiment_service.get_sentiment_history(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
        )

        if not history:
            return []

        # Convert to response models
        return [
            SentimentHistoryPoint(
                timestamp=point.get("timestamp", ""),
                overall_score=point.get("overall_score", 0.0),
                news_score=point.get("news_score", 0.0),
                social_score=point.get("social_score", 0.0),
                analyst_score=point.get("analyst_score", 0.0),
            )
            for point in history
        ]

    except Exception as e:
        logger.error(f"Error fetching sentiment history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sentiment/{symbol}/signal", response_model=SentimentSignal)
async def get_sentiment_signal(
    symbol: str,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
):
    """
    Get trading signal derived from sentiment analysis.

    Args:
        symbol: Trading pair symbol

    Returns:
        Trading signal (BUY/SELL/NEUTRAL) with confidence and reasoning
    """
    try:
        # Get latest sentiment
        sentiment_data = sentiment_service.get_latest_sentiment(symbol)

        if sentiment_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data for {symbol}"
            )

        # Calculate signal
        overall_score = sentiment_data.get("overall_score", 0.0)
        confidence = sentiment_data.get("confidence", 0.5)

        # Determine signal
        if overall_score > 0.3 and confidence > 0.6:
            signal = "BUY"
            trend = "BULLISH"
            strength = min(1.0, overall_score * confidence)
            reasoning = f"Strong positive sentiment ({overall_score:.2f}) with high confidence ({confidence:.2f})"
        elif overall_score < -0.3 and confidence > 0.6:
            signal = "SELL"
            trend = "BEARISH"
            strength = min(1.0, abs(overall_score) * confidence)
            reasoning = f"Strong negative sentiment ({overall_score:.2f}) with high confidence ({confidence:.2f})"
        else:
            signal = "NEUTRAL"
            trend = "NEUTRAL"
            strength = 0.0
            reasoning = f"Neutral sentiment ({overall_score:.2f}) or low confidence ({confidence:.2f})"

        return SentimentSignal(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            signal=signal,
            strength=strength,
            sentiment_score=overall_score,
            trend=trend,
            confidence=confidence,
            reasoning=reasoning,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating sentiment signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Calendar Endpoints
# ============================================================================

@app.get("/api/calendar/upcoming", response_model=List[CalendarEvent])
async def get_upcoming_events(
    hours: int = Query(24, ge=1, le=168, description="Hours ahead to look"),
    impact: Optional[str] = Query(None, regex="^(LOW|MEDIUM|HIGH)$"),
    country: Optional[str] = Query(None, description="Filter by country code"),
    calendar_service: EventCalendarService = Depends(get_calendar_service),
):
    """
    Get upcoming economic calendar events.

    Args:
        hours: Hours ahead to fetch events (1-168, default 24)
        impact: Filter by impact level (LOW/MEDIUM/HIGH)
        country: Filter by country code (e.g., US, EU, GB)

    Returns:
        List of upcoming economic events
    """
    try:
        # Calculate time range
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)

        # Get events
        events = calendar_service.get_upcoming_events(
            start_time=start_time,
            end_time=end_time,
            impact=impact,
            country=country,
        )

        if not events:
            return []

        # Convert to response models
        result = []
        for event in events:
            result.append(CalendarEvent(
                event_id=event.get("id", ""),
                title=event.get("title", ""),
                country=event.get("country", ""),
                timestamp=event.get("timestamp", ""),
                impact=event.get("impact", "MEDIUM"),
                currency=event.get("currency", ""),
                actual=event.get("actual"),
                forecast=event.get("forecast"),
                previous=event.get("previous"),
                affected_symbols=event.get("affected_symbols", []),
            ))

        return result

    except Exception as e:
        logger.error(f"Error fetching upcoming events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calendar/risk/{symbol}", response_model=RiskScore)
async def get_risk_score(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Hours ahead to consider"),
    calendar_service: EventCalendarService = Depends(get_calendar_service),
):
    """
    Get risk score for a symbol based on upcoming events.

    Args:
        symbol: Trading pair symbol
        hours: Hours ahead to consider for risk calculation

    Returns:
        Risk score with upcoming events affecting the symbol
    """
    try:
        # Get upcoming events affecting this symbol
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)

        events = calendar_service.get_events_for_symbol(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
        )

        # Calculate risk score
        high_impact_count = sum(1 for e in events if e.get("impact") == "HIGH")
        medium_impact_count = sum(1 for e in events if e.get("impact") == "MEDIUM")

        # Risk score calculation
        risk_score = min(1.0, (high_impact_count * 0.5 + medium_impact_count * 0.2))

        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "CRITICAL"
        elif risk_score >= 0.5:
            risk_level = "HIGH"
        elif risk_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Find next event
        next_event_hours = None
        if events:
            next_event_time = datetime.fromisoformat(events[0].get("timestamp", ""))
            next_event_hours = (next_event_time - start_time).total_seconds() / 3600

        # Convert events to response models
        event_models = [
            CalendarEvent(
                event_id=e.get("id", ""),
                title=e.get("title", ""),
                country=e.get("country", ""),
                timestamp=e.get("timestamp", ""),
                impact=e.get("impact", "MEDIUM"),
                currency=e.get("currency", ""),
                actual=e.get("actual"),
                forecast=e.get("forecast"),
                previous=e.get("previous"),
                affected_symbols=e.get("affected_symbols", []),
            )
            for e in events
        ]

        return RiskScore(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            risk_score=risk_score,
            risk_level=risk_level,
            upcoming_events_count=len(events),
            high_impact_events_count=high_impact_count,
            next_event_hours=next_event_hours,
            events=event_models,
        )

    except Exception as e:
        logger.error(f"Error calculating risk score for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Prediction Endpoint (bonus)
# ============================================================================

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get model prediction for a symbol.

    Note: This is a placeholder - actual implementation would load
    trained models and make predictions.

    Args:
        request: Prediction request with symbol and features

    Returns:
        Model prediction with confidence
    """
    # Placeholder implementation
    # In production, this would:
    # 1. Load appropriate model for symbol/timeframe
    # 2. Preprocess features
    # 3. Make prediction
    # 4. Calculate confidence

    return PredictionResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        prediction=0.0,  # Placeholder
        confidence=0.5,  # Placeholder
        timestamp=datetime.now().isoformat(),
        model_version="v1.0.0-placeholder",
    )


# ============================================================================
# Main entry point (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ForexGPT API server...")
    uvicorn.run(
        "forex_diffusion.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
