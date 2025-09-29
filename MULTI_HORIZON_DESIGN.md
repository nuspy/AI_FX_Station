# 🎯 Multi-Horizon Prediction System Design

## Requisito
**Usare lo stesso modello per predire diversi orizzonti temporali**
- Da una sola previsione generare multiple previsioni a orizzonti diversi
- Esempi: 10 minuti, 30 minuti, 1 ora, 4 ore, ecc.
- Basato sui timeframes, non su associazioni generative/patterns

---

## 🔍 Analisi Sistema Attuale

### Horizon Converter Esistente
Il sistema ha già `utils/horizon_converter.py` che gestisce:
```python
# Conversione bars → time labels
horizon_bars_to_time_labels(5, "1m") → ["1m", "5m", "15m", "1h", "4h"]

# Conversione per inference
convert_horizons_for_inference(["10m", "30m", "1h"], "1m")
```

### Limitazioni Attuali
1. **Scaling lineare semplice**: `base_pred * scale_factor`
2. **Non considera non-linearità temporali**
3. **Manca calibrazione per diversi timeframes**

---

## 🎨 Design Proposto: Smart Multi-Horizon System

### 1. Architettura Multi-Horizon

```python
class MultiHorizonPredictor:
    """
    Genera predizioni multiple da un singolo modello usando
    scaling intelligente basato su timeframe relationships.
    """

    def __init__(self, base_model, calibration_data=None):
        self.base_model = base_model
        self.horizon_calibrators = {}  # Calibratori per ogni horizon
        self.timeframe_relationships = {}  # Relazioni non-lineari

    def predict_multi_horizon(self, features, target_horizons):
        """
        Genera predizioni per multiple orizzonti temporali.

        Args:
            features: Features del modello base
            target_horizons: ["10m", "30m", "1h", "4h", "1d"]

        Returns:
            Dict[horizon, prediction_with_uncertainty]
        """
```

### 2. Scaling Non-Lineare Intelligente

```python
class HorizonScaler:
    """
    Scala predizioni considerando caratteristiche non-lineari del mercato.
    """

    SCALING_MODES = {
        "linear": lambda base, ratio: base * ratio,
        "sqrt": lambda base, ratio: base * np.sqrt(ratio),
        "log": lambda base, ratio: base * np.log1p(ratio),
        "volatility_adjusted": lambda base, ratio, vol: base * ratio * vol_factor,
        "regime_aware": lambda base, ratio, regime: base * ratio * regime_factor
    }

    def scale_prediction(self, base_pred, base_tf, target_tf, mode="smart"):
        """
        Scala intelligentemente da base_tf a target_tf.

        Considera:
        - Volatility clustering su timeframes diversi
        - Mean reversion vs trend persistence
        - Market regime (trending/range-bound)
        - Session caratteristiche (Asian/European/US)
        """
```

### 3. Calibrazione Automatica

```python
class HorizonCalibrator:
    """
    Calibra le predizioni multi-horizon usando dati storici.
    """

    def calibrate_horizons(self, historical_data, base_model):
        """
        Analizza performance storica del modello su diversi orizzonti
        per ottimizzare i fattori di scaling.
        """

        calibration_results = {}

        for target_horizon in self.target_horizons:
            # Calcola accuracy/correlation per ogni horizon
            metrics = self._evaluate_horizon_performance(
                historical_data, base_model, target_horizon
            )

            # Ottimizza scaling factors
            optimal_scaler = self._optimize_scaling_factors(
                historical_data, target_horizon, metrics
            )

            calibration_results[target_horizon] = {
                'scaler': optimal_scaler,
                'metrics': metrics,
                'confidence_intervals': self._compute_confidence_bands(...)
            }

        return calibration_results
```

---

## 🛠️ Implementazione Pratica

### Fase 1: Estensione Horizon Converter

```python
# In utils/horizon_converter.py

class SmartMultiHorizonConverter:
    """Converter avanzato con scaling intelligente."""

    def __init__(self):
        self.scaling_factors = self._load_calibrated_factors()
        self.market_regime_detector = MarketRegimeDetector()

    def convert_single_to_multi_horizon(
        self,
        base_prediction: float,
        base_timeframe: str,
        target_horizons: List[str],
        market_data: pd.DataFrame = None,
        uncertainty_bands: bool = True
    ) -> Dict[str, Dict]:
        """
        Converte una predizione base a multiple orizzonti.

        Returns:
            {
                "10m": {"prediction": 1.0523, "lower": 1.0515, "upper": 1.0531},
                "30m": {"prediction": 1.0547, "lower": 1.0532, "upper": 1.0562},
                "1h":  {"prediction": 1.0578, "lower": 1.0556, "upper": 1.0600},
                "4h":  {"prediction": 1.0645, "lower": 1.0598, "upper": 1.0692}
            }
        """

        results = {}

        # Detect current market regime
        regime = self.market_regime_detector.detect_regime(market_data)
        current_volatility = self._estimate_current_volatility(market_data)

        for target_horizon in target_horizons:
            # Calculate time ratio
            base_minutes = timeframe_to_minutes(base_timeframe)
            target_minutes = timeframe_to_minutes(target_horizon)
            time_ratio = target_minutes / base_minutes

            # Smart scaling based on regime and volatility
            scaled_prediction = self._smart_scale(
                base_prediction, time_ratio, regime, current_volatility
            )

            # Calculate uncertainty bands
            uncertainty = self._calculate_uncertainty(
                target_horizon, time_ratio, current_volatility
            )

            results[target_horizon] = {
                "prediction": scaled_prediction,
                "lower": scaled_prediction - uncertainty,
                "upper": scaled_prediction + uncertainty,
                "confidence": self._calculate_confidence(target_horizon),
                "regime": regime
            }

        return results
```

### Fase 2: Integrazione nel Sistema Esistente

```python
# In ui/controllers.py - modifica del metodo inference

def _enhanced_multi_horizon_inference(self):
    """Inference con multi-horizon intelligente."""

    # 1. Ottieni predizione base dal modello
    base_prediction = self._get_base_model_prediction()

    # 2. Configura orizzonti target dalla UI
    target_horizons = self._get_target_horizons_from_ui()

    # 3. Converti a multi-horizon
    multi_horizon_converter = SmartMultiHorizonConverter()

    multi_predictions = multi_horizon_converter.convert_single_to_multi_horizon(
        base_prediction=base_prediction,
        base_timeframe=self.payload.get("timeframe", "1m"),
        target_horizons=target_horizons,
        market_data=self._get_recent_market_data(),
        uncertainty_bands=True
    )

    # 4. Format risultati per display
    return self._format_multi_horizon_results(multi_predictions)
```

### Fase 3: UI Enhancement

```python
# In prediction_settings_dialog.py

class MultiHorizonSettings:
    """Sezione UI per configurare multi-horizon predictions."""

    def _create_multi_horizon_section(self, layout):
        """Crea sezione UI per multi-horizon."""

        mh_box = QGroupBox("Multi-Horizon Predictions")
        mh_layout = QVBoxLayout(mh_box)

        # Scenario selection
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems([
            "Scalping (1m-15m)",
            "Intraday 4h (1m-4h)",
            "Intraday 8h (1m-8h)",
            "Short-term (1m-2d)",
            "Medium-term (1m-5d)",
            "Long-term (1m-15d)",
            "Custom"
        ])

        # Custom horizons
        self.custom_horizons = QLineEdit("10m, 30m, 1h, 4h")

        # Scaling method
        self.scaling_method = QComboBox()
        self.scaling_method.addItems([
            "Smart Adaptive",
            "Linear",
            "Volatility Adjusted",
            "Regime Aware"
        ])

        mh_layout.addWidget(QLabel("Trading Scenario:"))
        mh_layout.addWidget(self.scenario_combo)
        mh_layout.addWidget(QLabel("Custom Horizons:"))
        mh_layout.addWidget(self.custom_horizons)
        mh_layout.addWidget(QLabel("Scaling Method:"))
        mh_layout.addWidget(self.scaling_method)

        layout.addWidget(mh_box)
```

---

## 📊 Scenari di Trading Predefiniti

### Scalping (High Frequency)
```python
SCALPING_HORIZONS = ["1m", "3m", "5m", "10m", "15m"]
# Focus: Micro-movements, alta frequenza
# Scaling: Non-lineare con decadimento rapido dell'accuracy
```

### Intraday 4h
```python
INTRADAY_4H_HORIZONS = ["5m", "15m", "30m", "1h", "2h", "4h"]
# Focus: Trend intraday, session-based
# Scaling: Volatility-adjusted per session changes
```

### Intraday 8h
```python
INTRADAY_8H_HORIZONS = ["15m", "30m", "1h", "2h", "4h", "6h", "8h"]
# Focus: Full trading session
# Scaling: Considera overlap sessions
```

### Short/Medium/Long-term
```python
SHORT_TERM_HORIZONS = ["1h", "4h", "8h", "12h", "1d", "2d"]
MEDIUM_TERM_HORIZONS = ["4h", "12h", "1d", "2d", "3d", "5d"]
LONG_TERM_HORIZONS = ["1d", "2d", "3d", "5d", "10d", "15d"]

# Exclude closed markets per giorni > 1
MARKET_AWARE_ADJUSTMENTS = {
    "exclude_weekends": True,
    "adjust_for_holidays": True,
    "consider_session_gaps": True
}
```

---

## 🎯 Vantaggi della Soluzione

### 1. **Single Model Efficiency**
- Un solo modello addestrato
- Inference veloce con scaling intelligente
- Meno memoria e computational overhead

### 2. **Adaptive Scaling**
- Considera regime di mercato corrente
- Volatility-aware adjustments
- Non-linear relationships

### 3. **Scenario-Based**
- Configurazioni predefinite per use cases comuni
- Customizable per strategie specifiche
- Market-hours aware

### 4. **Uncertainty Quantification**
- Confidence bands per ogni horizon
- Regime-dependent uncertainty
- Calibrated sulla performance storica

---

## 🚀 Piano di Implementazione

### Fase 1 (Immediate):
- [ ] Estendi `horizon_converter.py` con smart scaling
- [ ] Implementa scaling modes (linear, sqrt, log, volatility-adjusted)
- [ ] Basic UI per scenario selection

### Fase 2 (Short-term):
- [ ] Market regime detection
- [ ] Historical calibration system
- [ ] Uncertainty quantification

### Fase 3 (Medium-term):
- [ ] Advanced scenarios (scalping, intraday, long-term)
- [ ] Market-hours awareness
- [ ] Performance monitoring e auto-tuning

Questa soluzione mantiene la semplicità di un singolo modello mentre fornisce predizioni intelligenti su multiple scale temporali, perfetto per diversi scenari di trading.