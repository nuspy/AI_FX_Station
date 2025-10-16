"""
Test per verificare il flusso dati verso i pannelli.
Controlla che DOM service e Sentiment service ricevano dati.
"""
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from sqlalchemy import create_engine, text

def test_dom_data():
    """Test dati DOM (Order Flow Panel)."""
    logger.info("=" * 80)
    logger.info("TEST DATI DOM (ORDER FLOW PANEL)")
    logger.info("=" * 80)
    
    db_path = project_root / "data" / "forex_diffusion.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    try:
        with engine.connect() as conn:
            # Check market_depth table
            result = conn.execute(text(
                "SELECT COUNT(*) FROM market_depth WHERE symbol = 'EURUSD'"
            ))
            count = result.scalar()
            logger.info(f"  Record DOM per EURUSD: {count}")
            
            if count == 0:
                logger.warning("  ⚠️ Nessun dato DOM - Order Flow Panel non riceverà aggiornamenti")
                return False
            
            # Get latest DOM snapshot
            result = conn.execute(text(
                "SELECT ts_utc, best_bid, best_ask, bid_depth, ask_depth, depth_imbalance "
                "FROM market_depth "
                "WHERE symbol = 'EURUSD' "
                "ORDER BY ts_utc DESC LIMIT 1"
            ))
            row = result.fetchone()
            
            if row:
                from datetime import datetime
                ts_dt = datetime.fromtimestamp(row[0] / 1000)
                logger.info(f"\n  Ultimo snapshot DOM:")
                logger.info(f"    Timestamp: {ts_dt}")
                logger.info(f"    Best Bid: {row[1]}")
                logger.info(f"    Best Ask: {row[2]}")
                logger.info(f"    Bid Depth: {row[3]}")
                logger.info(f"    Ask Depth: {row[4]}")
                logger.info(f"    Depth Imbalance: {row[5]:.2f}")
                
                # Check if data is recent (< 1 minute old)
                age_seconds = (time.time() * 1000 - row[0]) / 1000
                if age_seconds > 60:
                    logger.warning(f"  ⚠️ Dati DOM vecchi ({age_seconds:.0f}s) - potrebbero non aggiornarsi")
                    return False
                else:
                    logger.success(f"  ✓ Dati DOM recenti ({age_seconds:.0f}s)")
                    return True
            
    except Exception as e:
        logger.error(f"  Errore: {e}")
        return False


def test_sentiment_data():
    """Test dati Sentiment (Sentiment Panel)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST DATI SENTIMENT (SENTIMENT PANEL)")
    logger.info("=" * 80)
    
    db_path = project_root / "data" / "forex_diffusion.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    try:
        with engine.connect() as conn:
            # Check sentiment_data table
            result = conn.execute(text(
                "SELECT COUNT(*) FROM sentiment_data WHERE symbol = 'EURUSD'"
            ))
            count = result.scalar()
            logger.info(f"  Record Sentiment per EURUSD: {count}")
            
            if count == 0:
                logger.warning("  ⚠️ Nessun dato Sentiment - Sentiment Panel non riceverà aggiornamenti")
                logger.info("  💡 Esegui: python create_test_sentiment_data.py")
                return False
            
            # Get latest sentiment snapshot
            result = conn.execute(text(
                "SELECT ts_utc, sentiment, long_pct, short_pct, confidence, ratio "
                "FROM sentiment_data "
                "WHERE symbol = 'EURUSD' "
                "ORDER BY ts_utc DESC LIMIT 1"
            ))
            row = result.fetchone()
            
            if row:
                from datetime import datetime
                ts_dt = datetime.fromtimestamp(row[0] / 1000)
                logger.info(f"\n  Ultimo snapshot Sentiment:")
                logger.info(f"    Timestamp: {ts_dt}")
                logger.info(f"    Sentiment: {row[1]}")
                logger.info(f"    Long%: {row[2]:.1f}%")
                logger.info(f"    Short%: {row[3]:.1f}%")
                logger.info(f"    Confidence: {row[4]:.2f}")
                logger.info(f"    Ratio: {row[5]:.2f}")
                
                # Calculate contrarian signal
                long_pct = row[2]
                if long_pct > 70:
                    contrarian = -(long_pct - 50) / 50
                elif long_pct < 30:
                    contrarian = (50 - long_pct) / 50
                else:
                    contrarian = 0.0
                
                logger.info(f"    Contrarian Signal: {contrarian:.2f}")
                
                # Check if data is recent
                age_seconds = (time.time() * 1000 - row[0]) / 1000
                if age_seconds > 3600:  # 1 hour
                    logger.warning(f"  ⚠️ Dati Sentiment vecchi ({age_seconds/60:.0f}min)")
                    logger.info("  💡 I dati di test non si auto-aggiornano")
                    logger.info("  💡 Con cTrader attivo, i dati si aggiornano automaticamente")
                    return True  # Still OK for testing
                else:
                    logger.success(f"  ✓ Dati Sentiment disponibili ({age_seconds:.0f}s)")
                    return True
            
    except Exception as e:
        logger.error(f"  Errore: {e}")
        return False


def test_services():
    """Test servizi aggregatori."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SERVIZI AGGREGATORI")
    logger.info("=" * 80)
    
    db_path = project_root / "data" / "forex_diffusion.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Test DOM Aggregator
    logger.info("\n  1. DOMAggregatorService:")
    try:
        from forex_diffusion.services.dom_aggregator import DOMAggregatorService
        dom_service = DOMAggregatorService(
            engine=engine,
            symbols=["EURUSD"],
            interval_seconds=2
        )
        snapshot = dom_service.get_latest_dom_snapshot("EURUSD")
        if snapshot:
            logger.success(f"     ✓ Restituisce snapshot: spread={snapshot.get('spread'):.5f}")
            logger.info(f"       Depth imbalance: {snapshot.get('depth_imbalance'):.2f}")
        else:
            logger.warning("     ⚠️ Nessun snapshot disponibile")
    except Exception as e:
        logger.error(f"     ✗ Errore: {e}")
    
    # Test Sentiment Aggregator
    logger.info("\n  2. SentimentAggregatorService:")
    try:
        from forex_diffusion.services.sentiment_aggregator import SentimentAggregatorService
        sentiment_service = SentimentAggregatorService(
            engine=engine,
            symbols=["EURUSD"],
            interval_seconds=30
        )
        metrics = sentiment_service.get_latest_sentiment_metrics("EURUSD")
        if metrics:
            logger.success(f"     ✓ Restituisce metriche: sentiment={metrics.get('sentiment')}")
            logger.info(f"       Long: {metrics.get('long_pct')}%")
            logger.info(f"       Contrarian: {metrics.get('contrarian_signal'):.2f}")
        else:
            logger.warning("     ⚠️ Nessuna metrica disponibile")
    except Exception as e:
        logger.error(f"     ✗ Errore: {e}")


def main():
    """Main test function."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")
    
    logger.info("🔍 VERIFICA FLUSSO DATI PANNELLI\n")
    
    dom_ok = test_dom_data()
    sentiment_ok = test_sentiment_data()
    test_services()
    
    logger.info("\n" + "=" * 80)
    logger.info("RIEPILOGO")
    logger.info("=" * 80)
    
    if dom_ok:
        logger.success("✓ Order Flow Panel riceverà aggiornamenti")
    else:
        logger.warning("⚠️ Order Flow Panel: NESSUN DATO o dati vecchi")
        logger.info("  Soluzione: Avvia ForexGPT con cTrader configurato")
    
    if sentiment_ok:
        logger.success("✓ Sentiment Panel riceverà aggiornamenti")
    else:
        logger.warning("⚠️ Sentiment Panel: NESSUN DATO")
        logger.info("  Soluzione: python create_test_sentiment_data.py")
    
    logger.info("\n💡 Per dati real-time:")
    logger.info("  1. Configura cTrader in Settings")
    logger.info("  2. Avvia ForexGPT")
    logger.info("  3. I pannelli si aggiorneranno automaticamente")


if __name__ == "__main__":
    main()
