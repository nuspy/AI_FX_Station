"""
Crea dati di test per sentiment_data per verificare il SentimentPanel.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from sqlalchemy import create_engine, text

def create_test_data():
    """Crea dati sentiment di test."""
    logger.info("Creazione dati sentiment di test...")
    
    db_path = project_root / "forexgpt.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    now = datetime.now()
    
    # Create realistic sentiment scenarios for each symbol
    scenarios = {
        "EURUSD": {"long_pct": 65.0, "sentiment": "bullish"},   # Moderately bullish
        "GBPUSD": {"long_pct": 35.0, "sentiment": "bearish"},   # Moderately bearish
        "USDJPY": {"long_pct": 78.0, "sentiment": "bullish"},   # Extremely bullish (contrarian signal!)
    }
    
    with engine.begin() as conn:
        # Clear existing test data
        conn.execute(text("DELETE FROM sentiment_data WHERE provider = 'test_data'"))
        
        # Create 100 data points for each symbol (spanning last hour)
        for symbol in symbols:
            scenario = scenarios[symbol]
            base_long = scenario["long_pct"]
            
            for i in range(100):
                # Add some variation to make it realistic
                long_pct = base_long + random.uniform(-5, 5)
                long_pct = max(10, min(90, long_pct))  # Clamp between 10-90%
                short_pct = 100 - long_pct
                
                # Simulate volume
                buy_volume = long_pct * 10
                sell_volume = short_pct * 10
                
                # Calculate sentiment ratio
                ratio = (long_pct - 50.0) / 50.0  # -1 to +1
                
                # Classify sentiment
                if long_pct > 60:
                    sentiment = "bullish"
                elif long_pct < 40:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"
                
                # Confidence based on distance from 50%
                confidence = abs(long_pct - 50.0) / 50.0
                
                # Timestamp going back in time
                ts_utc = int((now - timedelta(minutes=100-i)).timestamp() * 1000)
                
                conn.execute(text("""
                    INSERT INTO sentiment_data 
                    (symbol, ts_utc, long_pct, short_pct, total_traders, confidence, 
                     sentiment, ratio, buy_volume, sell_volume, provider, ts_created_ms)
                    VALUES 
                    (:symbol, :ts_utc, :long_pct, :short_pct, :total_traders, :confidence,
                     :sentiment, :ratio, :buy_volume, :sell_volume, :provider, :ts_created_ms)
                """), {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "long_pct": long_pct,
                    "short_pct": short_pct,
                    "total_traders": int(buy_volume + sell_volume),
                    "confidence": confidence,
                    "sentiment": sentiment,
                    "ratio": ratio,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "provider": "test_data",
                    "ts_created_ms": ts_utc
                })
            
            logger.success(f"✓ Creati 100 record per {symbol} (scenario: {scenario['sentiment']})")
    
    logger.info("\n" + "=" * 80)
    logger.success("✓ Dati di test creati con successo!")
    logger.info("=" * 80)
    logger.info("\nScenari creati:")
    logger.info("  EURUSD: 65% Long (Moderately Bullish)")
    logger.info("  GBPUSD: 35% Long (Moderately Bearish)")  
    logger.info("  USDJPY: 78% Long (Extremely Bullish - CONTRARIAN SHORT SIGNAL!)")
    logger.info("\nEsegui test_sentiment_panel.py per verificare i dati")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")
    create_test_data()
