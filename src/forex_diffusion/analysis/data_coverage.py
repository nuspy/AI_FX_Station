"""
Data Coverage Analysis Tool

Comprehensive analysis of available market data to inform training strategy.
Generates reports on symbols, timeframes, volume quality, and data gaps.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from forex_diffusion.services.marketdata import MarketDataService
except ImportError:
    try:
        from ..services.marketdata import MarketDataService
    except ImportError:
        MarketDataService = None


class DataCoverageAnalyzer:
    """
    Analyze data coverage across symbols, timeframes, and time periods.

    Provides recommendations for training data sufficiency and identifies gaps.
    """

    # Minimum candles required for reliable training (empirically derived)
    MIN_CANDLES_FOR_TRAINING = {
        '1m': 100000,   # ~70 days of 1-minute data
        '5m': 50000,    # ~175 days
        '15m': 20000,   # ~200 days
        '30m': 10000,   # ~200 days
        '1h': 8000,     # ~330 days
        '4h': 2000,     # ~330 days
        '1d': 500,      # ~500 days
        '1w': 100       # ~2 years
    }

    def __init__(self):
        """Initialize with database connection."""
        if MarketDataService is None:
            raise RuntimeError("MarketDataService not available")

        self.service = MarketDataService()
        self.engine = getattr(self.service, "engine", None)

        if self.engine is None:
            raise RuntimeError("Database engine not available")

    def analyze_symbols_coverage(self) -> pd.DataFrame:
        """
        Analyze coverage across all symbols.

        Returns:
            DataFrame with columns: symbol, total_candles, timeframes_count
        """
        query = text("""
            SELECT
                symbol,
                COUNT(*) as total_candles,
                COUNT(DISTINCT timeframe) as timeframes_count
            FROM market_data_candles
            GROUP BY symbol
            ORDER BY total_candles DESC
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=['symbol', 'total_candles', 'timeframes_count'])

        df = pd.DataFrame(rows, columns=['symbol', 'total_candles', 'timeframes_count'])
        return df

    def analyze_timeframes_coverage(self) -> pd.DataFrame:
        """
        Analyze coverage for each symbol-timeframe pair.

        Returns:
            DataFrame with: symbol, timeframe, first_candle, last_candle,
                           total_candles, months_coverage, status
        """
        query = text("""
            SELECT
                symbol,
                timeframe,
                MIN(ts_utc) as first_candle_ms,
                MAX(ts_utc) as last_candle_ms,
                COUNT(*) as total_candles
            FROM market_data_candles
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=[
                'symbol', 'timeframe', 'first_candle', 'last_candle',
                'total_candles', 'months_coverage', 'status'
            ])

        data = []
        for row in rows:
            symbol, timeframe, first_ms, last_ms, total_candles = row

            # Convert timestamps
            first_candle = pd.Timestamp(first_ms, unit='ms', tz='UTC')
            last_candle = pd.Timestamp(last_ms, unit='ms', tz='UTC')

            # Calculate months of coverage
            date_range = last_candle - first_candle
            months_coverage = date_range.days / 30.0

            # Determine status (sufficient/marginal/insufficient)
            required = self.MIN_CANDLES_FOR_TRAINING.get(timeframe, 10000)

            if total_candles >= required:
                status = "✅ SUFFICIENT"
            elif total_candles >= required * 0.7:
                status = "⚠️  MARGINAL"
            else:
                status = "❌ INSUFFICIENT"

            data.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'first_candle': first_candle,
                'last_candle': last_candle,
                'total_candles': total_candles,
                'months_coverage': round(months_coverage, 1),
                'required_candles': required,
                'coverage_pct': round(100 * total_candles / required, 1),
                'status': status
            })

        return pd.DataFrame(data)

    def analyze_volume_quality(self) -> pd.DataFrame:
        """
        Analyze quality of volume data.

        Returns:
            DataFrame with: symbol, timeframe, volume_coverage_pct,
                           avg_volume, std_volume, quality_score
        """
        query = text("""
            SELECT
                symbol,
                timeframe,
                SUM(CASE WHEN volume > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
                    as volume_coverage_pct,
                AVG(CASE WHEN volume > 0 THEN volume ELSE NULL END) as avg_volume,
                STDDEV(CASE WHEN volume > 0 THEN volume ELSE NULL END) as std_volume,
                COUNT(*) as total_candles
            FROM market_data_candles
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=[
                'symbol', 'timeframe', 'volume_coverage_pct',
                'avg_volume', 'std_volume', 'quality_score'
            ])

        data = []
        for row in rows:
            symbol, timeframe, vol_coverage, avg_vol, std_vol, total = row

            # Calculate quality score (0-100)
            # Based on: coverage %, avg volume, and consistency (CV)
            coverage_score = vol_coverage if vol_coverage else 0

            # Coefficient of variation (lower is more consistent)
            if avg_vol and avg_vol > 0 and std_vol:
                cv = std_vol / avg_vol
                consistency_score = max(0, 100 - cv * 10)  # Lower CV = higher score
            else:
                consistency_score = 0

            quality_score = (coverage_score * 0.7 + consistency_score * 0.3)

            # Quality rating
            if quality_score >= 80:
                quality = "✅ EXCELLENT"
            elif quality_score >= 60:
                quality = "🟢 GOOD"
            elif quality_score >= 40:
                quality = "⚠️  FAIR"
            else:
                quality = "❌ POOR"

            data.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'volume_coverage_pct': round(vol_coverage, 1) if vol_coverage else 0,
                'avg_volume': round(avg_vol, 2) if avg_vol else 0,
                'std_volume': round(std_vol, 2) if std_vol else 0,
                'quality_score': round(quality_score, 1),
                'quality': quality
            })

        return pd.DataFrame(data)

    def analyze_tick_data(self) -> Optional[pd.DataFrame]:
        """
        Analyze tick data coverage (if available).

        Returns:
            DataFrame with tick data info or None if no tick table exists
        """
        # Check if ticks table exists
        try:
            query = text("""
                SELECT
                    symbol,
                    MIN(timestamp) as first_tick,
                    MAX(timestamp) as last_tick,
                    COUNT(*) as total_ticks
                FROM ticks
                GROUP BY symbol
                ORDER BY symbol
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()

            if not rows:
                return None

            data = []
            for row in rows:
                symbol, first_tick, last_tick, total_ticks = row
                data.append({
                    'symbol': symbol,
                    'first_tick': pd.Timestamp(first_tick) if first_tick else None,
                    'last_tick': pd.Timestamp(last_tick) if last_tick else None,
                    'total_ticks': total_ticks
                })

            return pd.DataFrame(data)

        except Exception:
            # Table doesn't exist or query failed
            return None

    def analyze_features_coverage(self) -> Dict[str, Any]:
        """
        Analyze features table coverage.

        Returns:
            Dict with features count and breakdown
        """
        try:
            # Count unique features
            query_count = text("""
                SELECT COUNT(DISTINCT feature_name) as unique_features
                FROM features
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query_count)
                row = result.fetchone()
                unique_features = row[0] if row else 0

            # Get feature breakdown
            query_breakdown = text("""
                SELECT
                    feature_name,
                    COUNT(*) as occurrences
                FROM features
                GROUP BY feature_name
                ORDER BY occurrences DESC
                LIMIT 50
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query_breakdown)
                rows = result.fetchall()

            feature_breakdown = {row[0]: row[1] for row in rows} if rows else {}

            return {
                'unique_features': unique_features,
                'feature_breakdown': feature_breakdown
            }

        except Exception:
            return {
                'unique_features': 0,
                'feature_breakdown': {}
            }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data coverage report.

        Returns:
            Dict containing all analysis results and summary
        """
        print("Analyzing data coverage...")

        # Gather all analyses
        symbols_df = self.analyze_symbols_coverage()
        timeframes_df = self.analyze_timeframes_coverage()
        volume_df = self.analyze_volume_quality()
        ticks_df = self.analyze_tick_data()
        features_info = self.analyze_features_coverage()

        # Calculate summary statistics
        summary = {
            'total_symbols': len(symbols_df),
            'total_symbol_timeframe_pairs': len(timeframes_df),
            'total_candles': int(symbols_df['total_candles'].sum()) if len(symbols_df) > 0 else 0,
            'avg_volume_coverage': float(volume_df['volume_coverage_pct'].mean()) if len(volume_df) > 0 else 0,
            'date_range': {}
        }

        if len(timeframes_df) > 0:
            summary['date_range'] = {
                'earliest': timeframes_df['first_candle'].min(),
                'latest': timeframes_df['last_candle'].max()
            }

        # Count sufficient/marginal/insufficient pairs
        if len(timeframes_df) > 0:
            summary['data_sufficiency'] = {
                'sufficient': len(timeframes_df[timeframes_df['status'].str.contains('SUFFICIENT')]),
                'marginal': len(timeframes_df[timeframes_df['status'].str.contains('MARGINAL')]),
                'insufficient': len(timeframes_df[timeframes_df['status'].str.contains('INSUFFICIENT')])
            }

        report = {
            'summary': summary,
            'symbols': symbols_df,
            'timeframes': timeframes_df,
            'volume_quality': volume_df,
            'ticks': ticks_df,
            'features': features_info,
            'generated_at': datetime.now(timezone.utc)
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted coverage report."""
        print("\n" + "=" * 80)
        print("DATA COVERAGE ANALYSIS REPORT")
        print("=" * 80)

        # Summary
        summary = report['summary']
        print(f"\n📊 SUMMARY")
        print("-" * 80)
        print(f"Total Symbols: {summary['total_symbols']}")
        print(f"Total Symbol-Timeframe Pairs: {summary['total_symbol_timeframe_pairs']}")
        print(f"Total Candles: {summary['total_candles']:,}")
        print(f"Average Volume Coverage: {summary['avg_volume_coverage']:.1f}%")

        if summary['date_range']:
            print(f"Date Range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")

        if 'data_sufficiency' in summary:
            suff = summary['data_sufficiency']
            print(f"\nData Sufficiency:")
            print(f"  ✅ Sufficient: {suff['sufficient']}")
            print(f"  ⚠️  Marginal: {suff['marginal']}")
            print(f"  ❌ Insufficient: {suff['insufficient']}")

        # Symbols
        symbols_df = report['symbols']
        if len(symbols_df) > 0:
            print(f"\n📈 SYMBOLS COVERAGE")
            print("-" * 80)
            print(symbols_df.to_string(index=False))

        # Timeframes
        timeframes_df = report['timeframes']
        if len(timeframes_df) > 0:
            print(f"\n⏰ TIMEFRAMES COVERAGE")
            print("-" * 80)
            # Show only key columns
            display_cols = ['symbol', 'timeframe', 'total_candles', 'months_coverage', 'coverage_pct', 'status']
            print(timeframes_df[display_cols].to_string(index=False))

        # Volume Quality
        volume_df = report['volume_quality']
        if len(volume_df) > 0:
            print(f"\n📦 VOLUME QUALITY")
            print("-" * 80)
            display_cols = ['symbol', 'timeframe', 'volume_coverage_pct', 'quality_score', 'quality']
            print(volume_df[display_cols].to_string(index=False))

        # Features
        features_info = report['features']
        print(f"\n🔧 FEATURES")
        print("-" * 80)
        print(f"Unique Features: {features_info['unique_features']}")
        if features_info['feature_breakdown']:
            print(f"\nTop 10 Most Common Features:")
            for i, (feat, count) in enumerate(list(features_info['feature_breakdown'].items())[:10], 1):
                print(f"  {i}. {feat}: {count:,} occurrences")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 80)

        if len(timeframes_df) > 0:
            insufficient = timeframes_df[timeframes_df['status'].str.contains('INSUFFICIENT')]
            marginal = timeframes_df[timeframes_df['status'].str.contains('MARGINAL')]

            if len(insufficient) > 0:
                print("\n⚠️  INSUFFICIENT DATA - Acquire more data for:")
                for _, row in insufficient.iterrows():
                    print(f"  • {row['symbol']} {row['timeframe']}: {row['total_candles']:,} candles "
                          f"(need {row['required_candles']:,}, {row['coverage_pct']:.0f}% coverage)")

            if len(marginal) > 0:
                print("\n⚠️  MARGINAL DATA - Consider more data for:")
                for _, row in marginal.iterrows():
                    print(f"  • {row['symbol']} {row['timeframe']}: {row['total_candles']:,} candles "
                          f"({row['coverage_pct']:.0f}% of recommended)")

            sufficient = timeframes_df[timeframes_df['status'].str.contains('SUFFICIENT')]
            if len(sufficient) > 0:
                print(f"\n✅ READY FOR TRAINING:")
                for _, row in sufficient.iterrows():
                    print(f"  • {row['symbol']} {row['timeframe']}: {row['total_candles']:,} candles "
                          f"({row['months_coverage']:.1f} months)")

        # Volume quality recommendations
        if len(volume_df) > 0:
            poor_volume = volume_df[volume_df['quality'].str.contains('POOR')]
            if len(poor_volume) > 0:
                print(f"\n⚠️  POOR VOLUME QUALITY:")
                for _, row in poor_volume.iterrows():
                    print(f"  • {row['symbol']} {row['timeframe']}: {row['volume_coverage_pct']:.1f}% coverage")

        print("\n" + "=" * 80)
        print(f"Report generated at: {report['generated_at']}")
        print("=" * 80)


def main():
    """Main entry point for data coverage analysis."""
    try:
        analyzer = DataCoverageAnalyzer()
        report = analyzer.generate_comprehensive_report()
        analyzer.print_report(report)

        # Optionally save to JSON
        import json

        # Convert DataFrames to dict for JSON serialization
        report_serializable = {
            'summary': report['summary'],
            'symbols': report['symbols'].to_dict('records'),
            'timeframes': report['timeframes'].to_dict('records'),
            'volume_quality': report['volume_quality'].to_dict('records'),
            'features': report['features'],
            'generated_at': report['generated_at'].isoformat()
        }

        # Convert timestamps
        for key in ['symbols', 'timeframes', 'volume_quality']:
            for record in report_serializable[key]:
                for field in ['first_candle', 'last_candle', 'first_tick', 'last_tick']:
                    if field in record and record[field]:
                        if isinstance(record[field], pd.Timestamp):
                            record[field] = record[field].isoformat()

        output_path = Path(__file__).parent.parent.parent.parent / 'data_coverage_report.json'
        with open(output_path, 'w') as f:
            json.dump(report_serializable, f, indent=2, default=str)

        print(f"\n📄 Report saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"❌ Error analyzing data coverage: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
