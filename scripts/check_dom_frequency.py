"""
Check DOM Event Frequency in Database

Analyzes the market_depth table to show DOM data frequency and recent activity.
Useful for verifying that DOM events are being saved correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.forex_diffusion.utils.user_settings import get_setting


def analyze_dom_data():
    """Analyze DOM data frequency and recent activity."""
    
    # Load database path
    db_path = get_setting("database.path", "forexgpt.db")
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    logger.info(f"Analyzing DOM data in: {db_path}")
    
    engine = create_engine(f"sqlite:///{db_path}")
    
    with engine.connect() as conn:
        # Total records
        result = conn.execute(text("SELECT COUNT(*) FROM market_depth")).fetchone()
        total_records = result[0]
        
        logger.info(f"=" * 80)
        logger.info(f"MARKET DEPTH (DOM) DATA ANALYSIS")
        logger.info(f"=" * 80)
        logger.info(f"Total records: {total_records:,}")
        
        if total_records == 0:
            logger.warning("⚠️  No DOM data found in database")
            logger.info("\nPossible reasons:")
            logger.info("1. Account doesn't support DOM (common for demo accounts)")
            logger.info("2. WebSocket service not running")
            logger.info("3. Symbols not subscribed correctly")
            logger.info("\nRun: python scripts/test_ctrader_dom_subscription.py")
            return
        
        # Records per symbol
        result = conn.execute(text("""
            SELECT symbol, COUNT(*) as count
            FROM market_depth
            GROUP BY symbol
            ORDER BY count DESC
        """)).fetchall()
        
        logger.info("\nRecords per symbol:")
        for row in result:
            logger.info(f"  {row[0]:10s}: {row[1]:,} records")
        
        # Time range
        result = conn.execute(text("""
            SELECT 
                MIN(timestamp) as first_record,
                MAX(timestamp) as last_record
            FROM market_depth
        """)).fetchone()
        
        first_ts = datetime.fromisoformat(result[0])
        last_ts = datetime.fromisoformat(result[1])
        duration = last_ts - first_ts
        
        logger.info(f"\nTime range:")
        logger.info(f"  First record: {first_ts}")
        logger.info(f"  Last record:  {last_ts}")
        logger.info(f"  Duration:     {duration}")
        
        # Calculate frequency
        if duration.total_seconds() > 0:
            records_per_hour = total_records / (duration.total_seconds() / 3600)
            logger.info(f"\nEvent frequency:")
            logger.info(f"  {records_per_hour:.1f} records/hour")
            logger.info(f"  {records_per_hour/60:.1f} records/minute")
        
        # Recent activity (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        result = conn.execute(text("""
            SELECT COUNT(*) 
            FROM market_depth 
            WHERE timestamp > :cutoff
        """), {"cutoff": one_hour_ago.isoformat()}).fetchone()
        
        recent_count = result[0]
        logger.info(f"\nRecent activity (last hour):")
        logger.info(f"  {recent_count} records")
        
        if recent_count == 0:
            logger.warning("⚠️  No DOM data in last hour - WebSocket may not be running")
        else:
            logger.success(f"✓ DOM data is actively streaming ({recent_count} events/hour)")
        
        # Sample recent records
        result = conn.execute(text("""
            SELECT timestamp, symbol, side, price, size
            FROM market_depth
            ORDER BY timestamp DESC
            LIMIT 10
        """)).fetchall()
        
        logger.info(f"\nRecent records (last 10):")
        logger.info(f"  {'Timestamp':<20} {'Symbol':<10} {'Side':<5} {'Price':<10} {'Size':<10}")
        logger.info(f"  {'-'*20} {'-'*10} {'-'*5} {'-'*10} {'-'*10}")
        for row in result:
            ts = datetime.fromisoformat(row[0]).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"  {ts:<20} {row[1]:<10} {row[2]:<5} {row[3]:<10.5f} {row[4]:<10.2f}")
        
        # Storage estimate
        result = conn.execute(text("""
            SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()
        """)).fetchone()
        
        db_size_mb = result[0] / (1024 * 1024)
        logger.info(f"\nDatabase size: {db_size_mb:.2f} MB")
        
        if total_records > 0:
            bytes_per_record = (result[0] / total_records)
            records_per_mb = 1024 * 1024 / bytes_per_record
            logger.info(f"Average: {bytes_per_record:.0f} bytes/record ({records_per_mb:.0f} records/MB)")
            
            # Daily estimate
            if duration.total_seconds() > 3600:
                daily_records = total_records * (86400 / duration.total_seconds())
                daily_mb = daily_records / records_per_mb
                logger.info(f"\nEstimated daily growth:")
                logger.info(f"  {daily_records:,.0f} records/day")
                logger.info(f"  {daily_mb:.2f} MB/day")
                logger.info(f"  {daily_mb * 30:.1f} MB/month")
        
        logger.info(f"=" * 80)


if __name__ == "__main__":
    try:
        analyze_dom_data()
    except Exception as e:
        logger.exception(f"Error analyzing DOM data: {e}")
