# ForexGPT API Documentation

FastAPI service exposing sentiment analysis, economic calendar, and trading intelligence endpoints.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn pydantic loguru

# Optional: Redis for caching
pip install redis
```

### Running the API

**Development mode** (auto-reload):
```bash
python run_api.py
```

**Production mode** (multiple workers):
```bash
python run_api.py --production --workers 4
```

**Custom port**:
```bash
python run_api.py --port 8080
```

**With uvicorn directly**:
```bash
uvicorn src.forex_diffusion.api.main:app --reload --port 8000
```

### Access

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

---

## 📡 API Endpoints

### Health & Status

#### `GET /`
Root endpoint with basic info.

**Response**:
```json
{
  "service": "ForexGPT API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/api/docs"
}
```

#### `GET /api/health`
Health check with service status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T10:30:00",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "services": {
    "sentiment": "ok",
    "calendar": "ok",
    "database": "ok"
  }
}
```

---

### Sentiment Analysis

#### `GET /api/sentiment/{symbol}`
Get latest sentiment metrics for a symbol.

**Parameters**:
- `symbol` (path): Trading pair (e.g., EURUSD, GBPUSD)

**Example**:
```bash
curl http://localhost:8000/api/sentiment/EURUSD
```

**Response**:
```json
{
  "symbol": "EURUSD",
  "timestamp": "2025-10-05T10:30:00",
  "overall_score": 0.65,
  "news_score": 0.70,
  "social_score": 0.55,
  "analyst_score": 0.72,
  "volume_weighted_score": 0.68,
  "confidence": 0.85,
  "sources_count": 15,
  "last_updated": "2025-10-05T10:25:00"
}
```

**Sentiment Scores**:
- Range: -1.0 (very bearish) to +1.0 (very bullish)
- `overall_score`: Weighted average of all sources
- `news_score`: News sentiment
- `social_score`: Social media sentiment
- `analyst_score`: Professional analyst sentiment
- `volume_weighted_score`: Weighted by trading volume

---

#### `GET /api/sentiment/{symbol}/history`
Get historical sentiment data.

**Parameters**:
- `symbol` (path): Trading pair
- `hours` (query): Hours of history (1-720, default 24)

**Example**:
```bash
curl "http://localhost:8000/api/sentiment/EURUSD/history?hours=48"
```

**Response**:
```json
[
  {
    "timestamp": "2025-10-05T10:00:00",
    "overall_score": 0.65,
    "news_score": 0.70,
    "social_score": 0.55,
    "analyst_score": 0.72
  },
  {
    "timestamp": "2025-10-05T09:00:00",
    "overall_score": 0.60,
    "news_score": 0.65,
    "social_score": 0.50,
    "analyst_score": 0.68
  }
]
```

---

#### `GET /api/sentiment/{symbol}/signal`
Get trading signal derived from sentiment.

**Parameters**:
- `symbol` (path): Trading pair

**Example**:
```bash
curl http://localhost:8000/api/sentiment/EURUSD/signal
```

**Response**:
```json
{
  "symbol": "EURUSD",
  "timestamp": "2025-10-05T10:30:00",
  "signal": "BUY",
  "strength": 0.75,
  "sentiment_score": 0.65,
  "trend": "BULLISH",
  "confidence": 0.85,
  "reasoning": "Strong positive sentiment (0.65) with high confidence (0.85)"
}
```

**Signal Values**:
- `signal`: BUY, SELL, or NEUTRAL
- `strength`: 0.0-1.0 (signal strength)
- `trend`: BULLISH, BEARISH, or NEUTRAL
- `confidence`: 0.0-1.0 (confidence in signal)

**Signal Logic**:
- **BUY**: sentiment > 0.3 AND confidence > 0.6
- **SELL**: sentiment < -0.3 AND confidence > 0.6
- **NEUTRAL**: Otherwise

---

### Economic Calendar

#### `GET /api/calendar/upcoming`
Get upcoming economic calendar events.

**Parameters**:
- `hours` (query): Hours ahead to look (1-168, default 24)
- `impact` (query, optional): Filter by impact (LOW/MEDIUM/HIGH)
- `country` (query, optional): Filter by country code (US, EU, GB, etc.)

**Examples**:
```bash
# Next 24 hours
curl http://localhost:8000/api/calendar/upcoming

# Next week, high impact only
curl "http://localhost:8000/api/calendar/upcoming?hours=168&impact=HIGH"

# US events only
curl "http://localhost:8000/api/calendar/upcoming?country=US"
```

**Response**:
```json
[
  {
    "event_id": "evt_12345",
    "title": "Non-Farm Payrolls",
    "country": "US",
    "timestamp": "2025-10-05T12:30:00",
    "impact": "HIGH",
    "currency": "USD",
    "actual": null,
    "forecast": 180000,
    "previous": 175000,
    "affected_symbols": ["EURUSD", "GBPUSD", "USDJPY"]
  },
  {
    "event_id": "evt_12346",
    "title": "ECB Interest Rate Decision",
    "country": "EU",
    "timestamp": "2025-10-05T14:00:00",
    "impact": "HIGH",
    "currency": "EUR",
    "actual": null,
    "forecast": 4.50,
    "previous": 4.50,
    "affected_symbols": ["EURUSD", "EURGBP", "EURJPY"]
  }
]
```

---

#### `GET /api/calendar/risk/{symbol}`
Get risk score for a symbol based on upcoming events.

**Parameters**:
- `symbol` (path): Trading pair
- `hours` (query): Hours ahead to consider (1-168, default 24)

**Example**:
```bash
curl "http://localhost:8000/api/calendar/risk/EURUSD?hours=48"
```

**Response**:
```json
{
  "symbol": "EURUSD",
  "timestamp": "2025-10-05T10:30:00",
  "risk_score": 0.85,
  "risk_level": "CRITICAL",
  "upcoming_events_count": 5,
  "high_impact_events_count": 2,
  "next_event_hours": 2.5,
  "events": [
    {
      "event_id": "evt_12345",
      "title": "Non-Farm Payrolls",
      "country": "US",
      "timestamp": "2025-10-05T12:30:00",
      "impact": "HIGH",
      "currency": "USD",
      "actual": null,
      "forecast": 180000,
      "previous": 175000,
      "affected_symbols": ["EURUSD", "GBPUSD", "USDJPY"]
    }
  ]
}
```

**Risk Levels**:
- `CRITICAL`: risk_score ≥ 0.7 (avoid trading)
- `HIGH`: risk_score ≥ 0.5 (reduce position size)
- `MEDIUM`: risk_score ≥ 0.3 (normal caution)
- `LOW`: risk_score < 0.3 (normal conditions)

**Risk Score Calculation**:
```
risk_score = min(1.0, high_impact_count × 0.5 + medium_impact_count × 0.2)
```

---

### Model Predictions

#### `POST /api/predict`
Get model prediction (placeholder - requires trained model).

**Request Body**:
```json
{
  "symbol": "EURUSD",
  "timeframe": "5m",
  "features": {
    "rsi": 65.5,
    "macd": 0.0012,
    "sentiment": 0.65
  }
}
```

**Response**:
```json
{
  "symbol": "EURUSD",
  "timeframe": "5m",
  "prediction": 0.0015,
  "confidence": 0.75,
  "timestamp": "2025-10-05T10:30:00",
  "model_version": "v1.0.0"
}
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Server
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=False
API_WORKERS=4

# CORS
CORS_ORIGINS=*
CORS_CREDENTIALS=True

# Cache
REDIS_URL=redis://localhost:6379
CACHE_TTL=300

# Database
DATABASE_URL=sqlite:///forexgpt.db

# Services
SENTIMENT_UPDATE_INTERVAL=3600
CALENDAR_UPDATE_INTERVAL=1800

# Logging
LOG_LEVEL=INFO
```

### Redis Caching (Optional)

If Redis is available, the API will automatically cache responses:

```bash
# Install Redis client
pip install redis

# Start Redis (Docker)
docker run -d -p 6379:6379 redis:alpine

# Set REDIS_URL
export REDIS_URL=redis://localhost:6379
```

---

## 📊 Usage Examples

### Python Client

```python
import requests

# Get sentiment
response = requests.get("http://localhost:8000/api/sentiment/EURUSD")
sentiment = response.json()
print(f"EUR/USD sentiment: {sentiment['overall_score']}")

# Get trading signal
response = requests.get("http://localhost:8000/api/sentiment/EURUSD/signal")
signal = response.json()
print(f"Signal: {signal['signal']} (strength: {signal['strength']})")

# Get risk score
response = requests.get("http://localhost:8000/api/calendar/risk/EURUSD")
risk = response.json()
print(f"Risk level: {risk['risk_level']} ({risk['upcoming_events_count']} events)")
```

### JavaScript/Node.js

```javascript
// Get sentiment
const response = await fetch('http://localhost:8000/api/sentiment/EURUSD');
const sentiment = await response.json();
console.log(`EUR/USD sentiment: ${sentiment.overall_score}`);

// Get upcoming events
const events = await fetch('http://localhost:8000/api/calendar/upcoming?hours=24');
const data = await events.json();
console.log(`${data.length} events in next 24h`);
```

### cURL

```bash
# Get sentiment
curl http://localhost:8000/api/sentiment/EURUSD | jq

# Get signal
curl http://localhost:8000/api/sentiment/EURUSD/signal | jq '.signal'

# Get upcoming high-impact events
curl "http://localhost:8000/api/calendar/upcoming?impact=HIGH" | jq

# Get risk score
curl "http://localhost:8000/api/calendar/risk/EURUSD" | jq '.risk_level'
```

---

## 🔒 Production Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run_api.py", "--production", "--workers", "4"]
```

Build and run:
```bash
docker build -t forexgpt-api .
docker run -p 8000:8000 forexgpt-api
```

### Nginx Reverse Proxy

```nginx
upstream forexgpt_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.forexgpt.com;

    location / {
        proxy_pass http://forexgpt_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Systemd Service

Create `/etc/systemd/system/forexgpt-api.service`:

```ini
[Unit]
Description=ForexGPT API Service
After=network.target

[Service]
Type=simple
User=forexgpt
WorkingDirectory=/opt/forexgpt
ExecStart=/opt/forexgpt/venv/bin/python run_api.py --production --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable forexgpt-api
sudo systemctl start forexgpt-api
sudo systemctl status forexgpt-api
```

---

## 📈 Monitoring

### Prometheus Metrics (Future)

Add prometheus middleware:
```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Metrics available at: `http://localhost:8000/metrics`

### Logging

Logs are written to stdout in structured format:

```
2025-10-05 10:30:00 | INFO     | main:startup_event - ForexGPT API started successfully
2025-10-05 10:30:15 | INFO     | main:get_sentiment - Fetching sentiment for EURUSD
```

---

## 🧪 Testing

### Manual Testing

Use the interactive docs at http://localhost:8000/api/docs to test all endpoints.

### Automated Testing

```python
from fastapi.testclient import TestClient
from src.forex_diffusion.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_sentiment():
    response = client.get("/api/sentiment/EURUSD")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "EURUSD"
    assert -1.0 <= data["overall_score"] <= 1.0
```

---

## 📝 Notes

- **Sentiment data**: Requires `SentimentService` to be properly configured with data sources
- **Calendar data**: Requires `EventCalendarService` with economic calendar provider
- **Predictions**: Placeholder endpoint - requires trained models to be loaded
- **Rate limiting**: Not implemented - add for production use
- **Authentication**: Not implemented - add for production use

---

## 🤝 Contributing

The API is designed to be extended. To add new endpoints:

1. Define request/response models with Pydantic
2. Add endpoint function with proper typing
3. Use dependency injection for services
4. Add comprehensive docstrings
5. Update this README

---

## 📄 License

Part of ForexGPT project.

---

**API Version**: 1.0.0
**Last Updated**: October 5, 2025
