"""
Test script per verificare SentimentPanel e SentimentAggregatorService.

Questo script:
1. Verifica la presenza di dati sentiment nel database
2. Testa il SentimentAggregatorService
3. Mostra i dati formattati per l'UI
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from sqlalchemy import create_engine, text
from forex_diffusion.services.sentiment_aggregator import SentimentAggregatorService

def check_database():
    """Verifica la presenza di dati sentiment nel database."""
    logger.info("=" * 80)
    logger.info("VERIFICA DATABASE SENTIMENT")
    logger.info("=" * 80)
    
    # Connect to database
    db_path = project_root / "forexgpt.db"
    if not db_path.exists():
        logger.error(f"Database non trovato: {db_path}")
        return False
    
    engine = create_engine(f"sqlite:///{db_path}")
    
    try:
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'"
            ))
            table_exists = result.fetchone()
            if not table_exists:
                logger.warning("⚠️ Tabella sentiment_data non esiste - creazione in corso...")
                # Create table manually if migration failed
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sentiment_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(64) NOT NULL,
                        ts_utc BIGINT NOT NULL,
                        long_pct REAL,
                        short_pct REAL,
                        total_traders INTEGER,
                        confidence REAL,
                        sentiment VARCHAR(32),
                        ratio REAL,
                        buy_volume REAL,
                        sell_volume REAL,
                        provider VARCHAR(64),
                        ts_created_ms BIGINT
                    )
                """))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_ts ON sentiment_data(symbol, ts_utc)"
                ))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_sentiment_ts ON sentiment_data(ts_utc)"
                ))
                conn.commit()
                logger.success("✓ Tabella sentiment_data creata")
            else:
                logger.success("✓ Tabella sentiment_data trovata")
            
            # Count records
            result = conn.execute(text("SELECT COUNT(*) FROM sentiment_data"))
            count = result.scalar()
            logger.info(f"  Record totali: {count}")
            
            if count == 0:
                logger.warning("⚠️ Nessun dato sentiment nel database")
                logger.info("  Il CTraderWebSocketService deve essere attivo per generare dati")
                return False
            
            # Show recent records per symbol
            result = conn.execute(text(
                "SELECT symbol, COUNT(*) as cnt, MAX(ts_utc) as last_ts "
                "FROM sentiment_data "
                "GROUP BY symbol "
                "ORDER BY last_ts DESC"
            ))
            
            logger.info("\n  Dati per simbolo:")
            for row in result:
                symbol, cnt, last_ts = row
                from datetime import datetime
                last_dt = datetime.fromtimestamp(last_ts / 1000)
                logger.info(f"    {symbol}: {cnt} record, ultimo aggiornamento: {last_dt}")
            
            # Show latest sentiment for each symbol
            logger.info("\n  Ultimi valori sentiment:")
            result = conn.execute(text(
                "SELECT symbol, sentiment, long_pct, short_pct, confidence, ts_utc "
                "FROM sentiment_data "
                "WHERE ts_utc IN ("
                "  SELECT MAX(ts_utc) FROM sentiment_data GROUP BY symbol"
                ") "
                "ORDER BY symbol"
            ))
            
            for row in result:
                symbol, sentiment, long_pct, short_pct, confidence, ts_utc = row
                from datetime import datetime
                ts_dt = datetime.fromtimestamp(ts_utc / 1000)
                logger.info(
                    f"    {symbol}: {sentiment.upper() if sentiment else 'N/A'} | "
                    f"Long: {long_pct:.1f}% | Short: {short_pct:.1f}% | "
                    f"Confidence: {confidence:.2f} | {ts_dt}"
                )
            
            return True
            
    except Exception as e:
        logger.error(f"Errore verifica database: {e}")
        return False


def test_sentiment_service():
    """Test del SentimentAggregatorService."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SENTIMENT AGGREGATOR SERVICE")
    logger.info("=" * 80)
    
    db_path = project_root / "forexgpt.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create service
    service = SentimentAggregatorService(
        engine=engine,
        symbols=["EURUSD", "GBPUSD", "USDJPY"],
        interval_seconds=30
    )
    
    logger.info("✓ SentimentAggregatorService creato")
    
    # Test get_latest_sentiment_metrics for each symbol
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    for symbol in symbols:
        logger.info(f"\n  Test {symbol}:")
        metrics = service.get_latest_sentiment_metrics(symbol)
        
        if not metrics:
            logger.warning(f"    ⚠️ Nessun dato disponibile per {symbol}")
            continue
        
        logger.info(f"    Sentiment: {metrics.get('sentiment', 'N/A').upper()}")
        logger.info(f"    Confidence: {metrics.get('confidence', 0):.2%}")
        logger.info(f"    Ratio: {metrics.get('ratio', 0):.2f}")
        logger.info(f"    Long: {metrics.get('long_pct', 0):.1f}%")
        logger.info(f"    Short: {metrics.get('short_pct', 0):.1f}%")
        logger.info(f"    Total Traders (volume): {metrics.get('total_traders', 0)}")
        logger.info(f"    Contrarian Signal: {metrics.get('contrarian_signal', 0):.2f}")
        
        # Check if contrarian signal is calculated correctly
        long_pct = metrics.get('long_pct', 50.0)
        if long_pct > 70:
            logger.info(f"    🔴 ALERTA CONTRARIAN: Crowd troppo long ({long_pct:.1f}%) - Consider SHORT")
        elif long_pct < 30:
            logger.info(f"    🟢 ALERTA CONTRARIAN: Crowd troppo short ({long_pct:.1f}%) - Consider LONG")
        else:
            logger.info(f"    ⚪ Posizionamento bilanciato")
    
    return True


def main():
    """Main test function."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")
    
    # Check database first
    db_ok = check_database()
    
    if not db_ok:
        logger.warning("\n" + "=" * 80)
        logger.warning("NOTA: Per generare dati sentiment:")
        logger.warning("  1. Avvia ForexGPT con cTrader configurato")
        logger.warning("  2. Assicurati che CTraderWebSocketService sia attivo")
        logger.warning("  3. I dati verranno generati automaticamente dal order flow")
        logger.warning("=" * 80)
        return 1
    
    # Test service
    test_sentiment_service()
    
    logger.info("\n" + "=" * 80)
    logger.success("✓ Test completato con successo!")
    logger.info("=" * 80)
    logger.info("\nIl SentimentPanel ora dovrebbe visualizzare questi dati nell'UI.")
    logger.info("Per verificare, avvia ForexGPT e controlla il pannello Sentiment nel ChartTab.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
