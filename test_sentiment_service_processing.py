"""
Test del processing del SentimentAggregatorService.
Simula il ciclo di elaborazione per verificare il logging.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from sqlalchemy import create_engine
from forex_diffusion.services.sentiment_aggregator import SentimentAggregatorService

def test_processing():
    """Test processing del servizio."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format="<level>{level: <8}</level> | {message}")
    
    logger.info("=" * 80)
    logger.info("TEST PROCESSING SENTIMENT AGGREGATOR")
    logger.info("=" * 80)
    
    db_path = project_root / "data" / "forex_diffusion.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create service
    service = SentimentAggregatorService(
        engine=engine,
        symbols=["EURUSD", "GBPUSD", "USDJPY"],
        interval_seconds=30
    )
    
    logger.info("✓ Servizio creato\n")
    
    # Manually trigger one processing iteration
    logger.info("Esecuzione singola iterazione di processing...\n")
    service._process_iteration()
    
    logger.info("\n" + "=" * 80)
    logger.success("✓ Processing completato senza errori!")
    logger.info("=" * 80)
    
    # Show cached data
    logger.info("\nDati in cache dopo processing:")
    for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
        if symbol in service._sentiment_history:
            history_len = len(service._sentiment_history[symbol])
            logger.info(f"  {symbol}: {history_len} record in cache")
        else:
            logger.warning(f"  {symbol}: Nessun dato in cache")


if __name__ == "__main__":
    test_processing()
