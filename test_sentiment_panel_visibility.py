"""
Test visibilità SentimentPanel nell'UI.
Apre una finestra minimale per verificare che il widget sia visibile.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from loguru import logger

from sqlalchemy import create_engine
from forex_diffusion.services.sentiment_aggregator import SentimentAggregatorService
from forex_diffusion.ui.sentiment_panel import SentimentPanel


class TestWindow(QMainWindow):
    """Finestra di test per SentimentPanel."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test SentimentPanel Visibility")
        self.resize(800, 600)
        
        # Create service
        db_path = project_root / "data" / "forex_diffusion.db"
        engine = create_engine(f"sqlite:///{db_path}")
        
        self.sentiment_service = SentimentAggregatorService(
            engine=engine,
            symbols=["EURUSD", "GBPUSD", "USDJPY"],
            interval_seconds=30
        )
        
        # Process once to populate cache
        self.sentiment_service._process_iteration()
        
        # Create central widget with splitter
        central = QWidget()
        layout = QVBoxLayout(central)
        
        # Add title
        title = QLabel("Test Visibilità SentimentPanel")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Create splitter (simulating right_splitter in ChartTab)
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Add placeholder widgets
        placeholder1 = QLabel("Area Grafico (placeholder)")
        placeholder1.setStyleSheet("background: #2b2b2b; color: white; padding: 20px;")
        placeholder1.setMinimumHeight(200)
        splitter.addWidget(placeholder1)
        
        placeholder2 = QLabel("Tabella Ordini (placeholder)")
        placeholder2.setStyleSheet("background: #1e1e1e; color: white; padding: 20px;")
        placeholder2.setMinimumHeight(100)
        splitter.addWidget(placeholder2)
        
        # Add OrderFlowPanel placeholder
        order_flow_placeholder = QLabel("OrderFlowPanel (placeholder)")
        order_flow_placeholder.setStyleSheet("background: #2a2a2a; color: white; padding: 20px;")
        order_flow_placeholder.setMinimumHeight(150)
        splitter.addWidget(order_flow_placeholder)
        
        # Add REAL SentimentPanel
        logger.info("Creazione SentimentPanel...")
        self.sentiment_panel = SentimentPanel(
            parent=self,
            sentiment_service=self.sentiment_service
        )
        splitter.addWidget(self.sentiment_panel)
        
        # Set stretch factors
        splitter.setStretchFactor(0, 3)  # Chart area gets most space
        splitter.setStretchFactor(1, 1)  # Orders table
        splitter.setStretchFactor(2, 2)  # Order Flow
        splitter.setStretchFactor(3, 2)  # Sentiment Panel
        
        layout.addWidget(splitter)
        
        # Info label
        info = QLabel(
            "Il SentimentPanel dovrebbe essere visibile in fondo.\n"
            "Se non lo vedi, trascina i separatori dello splitter."
        )
        info.setStyleSheet("padding: 10px; background: #333; color: #ccc;")
        layout.addWidget(info)
        
        self.setCentralWidget(central)
        
        # Log widget visibility
        logger.info(f"SentimentPanel creato: visible={self.sentiment_panel.isVisible()}")
        logger.info(f"SentimentPanel size: {self.sentiment_panel.size()}")
        logger.info(f"SentimentPanel minimumSize: {self.sentiment_panel.minimumSize()}")
        
        # Check if service is connected
        if self.sentiment_panel.sentiment_service:
            logger.success("✓ sentiment_service collegato al pannello")
            
            # Test getting data
            metrics = self.sentiment_service.get_latest_sentiment_metrics("EURUSD")
            if metrics:
                logger.success(f"✓ Dati disponibili: {metrics}")
                # Update panel with data
                self.sentiment_panel.update_metrics(metrics)
            else:
                logger.warning("⚠️ Nessun dato disponibile per EURUSD")
        else:
            logger.error("✗ sentiment_service NON collegato!")


def main():
    """Main test function."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")
    
    logger.info("=" * 80)
    logger.info("TEST VISIBILITÀ SENTIMENT PANEL")
    logger.info("=" * 80)
    
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = TestWindow()
    window.show()
    
    logger.info("\n" + "=" * 80)
    logger.info("Finestra aperta - verifica visibilità SentimentPanel")
    logger.info("Premi Ctrl+C nel terminale per chiudere")
    logger.info("=" * 80)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
