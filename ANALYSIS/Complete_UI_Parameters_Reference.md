# ForexGPT - Complete UI Parameters Reference Guide

**Document Version**: 2.0  
**Date**: 2025-01-08  
**Purpose**: Exhaustive documentation of ALL tabs, sub-tabs, parameters, tooltips, and configuration options  
**Scope**: Complete replacement for user manual - every configurable element documented

---

## Table of Contents

### LEVEL 1 TABS (Main Navigation)
1. [Chart Tab](#1-chart-tab) - No nested tabs, direct chart interface
2. [Trading Intelligence Tab](#2-trading-intelligence-tab) - Contains Portfolio + Signals
3. [Generative Forecast Tab](#3-generative-forecast-tab) - Contains Forecast Settings + Training + Backtesting
4. [Patterns Tab](#4-patterns-tab) - Pattern training and configuration
5. [Logs Tab](#5-logs-tab) - System logs and diagnostics
6. [3D Reports Tab](#6-3d-reports-tab) - 3D visualization reports

### ADDITIONAL WINDOWS
7. [Live Trading Window](#7-live-trading-window) - Separate window for live trading
8. [Settings/Configuration Dialogs](#8-settings-dialogs) - Global application settings

---

## STRUCTURE OVERVIEW

```
ForexGPT Main Window
│
├── [LEVEL 1] Chart (No nested tabs - direct chart content)
│   ├── Left Panel
│   │   ├── Market Watch Widget
│   │   ├── VIX Widget (compact)
│   │   ├── Order Books Widget
│   │   └── Order Flow Widget
│   ├── Center Area
│   │   ├── Drawing Toolbar
│   │   ├── Main Chart (PyQtGraph/finplot)
│   │   ├── Volume Subplot
│   │   └── Indicator Subplots (dynamic)
│   └── Right Panel
│       ├── Chart Controls
│       ├── Timeframe Selector
│       ├── Symbol Selector
│       ├── Indicators Manager
│       ├── Sentiment Panel
│       └── Pattern Recognition Panel
│
├── [LEVEL 1] Trading Intelligence
│   ├── [LEVEL 2] Portfolio Tab
│   │   ├── Portfolio Optimization Panel
│   │   ├── Efficient Frontier Visualization
│   │   ├── Asset Allocation Controls
│   │   └── Risk Metrics Display
│   └── [LEVEL 2] Signals Tab
│       ├── Signal Quality Monitor
│       ├── Signal History Table
│       ├── Signal Filters
│       └── Signal Analytics
│
├── [LEVEL 1] Generative Forecast
│   ├── [LEVEL 2] Forecast Settings Tab
│   │   ├── Model Selection
│   │   ├── Horizon Configuration
│   │   ├── Uncertainty Controls
│   │   └── Ensemble Settings
│   ├── [LEVEL 2] Training Tab
│   │   ├── Manual Training Panel
│   │   ├── Multi-Timeframe Ensemble Training
│   │   ├── Feature Selection
│   │   ├── Algorithm Selection
│   │   ├── Hyperparameter Configuration
│   │   └── Training Progress Monitor
│   └── [LEVEL 2] Backtesting Tab
│       ├── Backtest Configuration
│       ├── Strategy Parameters
│       ├── Risk Management Settings
│       ├── Results Display
│       └── Performance Metrics
│
├── [LEVEL 1] Patterns
│   ├── Pattern Training Panel
│   │   ├── Chart Patterns Training
│   │   ├── Candlestick Patterns Training
│   │   ├── Genetic Algorithm Configuration
│   │   ├── Parameter Space Definition
│   │   └── Optimization Results
│   └── Pattern Settings Panel
│       ├── Pattern Detection Parameters
│       ├── Pattern Sensitivity Controls
│       └── Pattern Display Options
│
├── [LEVEL 1] Logs
│   ├── Application Logs Panel
│   ├── Data Sources Status
│   ├── Trading Engine Logs
│   └── System Diagnostics
│
├── [LEVEL 1] 3D Reports
│   ├── 3D Equity Curve
│   ├── 3D Risk Surface
│   ├── 3D Performance Landscape
│   └── Visualization Controls
│
└── [SEPARATE WINDOW] Live Trading
    ├── Trading Controls
    ├── Position Management
    ├── Order Execution
    ├── Risk Monitor
    └── P&L Dashboard
```

---

# DETAILED PARAMETER DOCUMENTATION

---

## 1. Chart Tab

**Location**: Main Window → Chart Tab (Level 1, no nested tabs)

**Description**: Primary chart interface with real-time price visualization, technical indicators, drawing tools, and pattern recognition. This is the main workspace for market analysis.

### 1.1 Left Panel Widgets

#### 1.1.1 Market Watch Widget

**Purpose**: Monitors multiple currency pairs simultaneously with real-time bid/ask prices and pip changes.

**Parameters**:

##### Symbol List
- **What it is**: List of actively monitored currency pairs
- **What it serves**: Quick access to multiple instruments for switching charts and monitoring correlations
- **Default symbols**: EUR/USD, GBP/USD, USD/JPY
- **How to use**: Click on any symbol to switch the main chart to that pair
- **Tooltip**:
  ```
  Market Watch - Multi-Symbol Monitor
  
  What it is:
  Real-time price monitoring for multiple currency pairs simultaneously.
  Shows bid/ask prices, spread, and pip change for each symbol.
  
  What it serves:
  - Quick switching between instruments (click to load chart)
  - Correlation awareness (see related pairs moving together)
  - Opportunity scanning (identify which pairs are moving)
  - Market sentiment gauge (see overall USD strength, etc.)
  
  How to use:
  - Left-click symbol: Load that pair on main chart
  - Right-click symbol: Add/remove from watchlist
  - Green values: Price increased since last update
  - Red values: Price decreased since last update
  
  Default symbols: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF
  Updates: Real-time (every tick received from broker)
  ```

##### Spread Display

**Parametro**: Spread (Bid-Ask Difference) - Visualizzazione in tempo reale
- **Tooltip**:
  ```
  Spread (Bid-Ask Difference)
  
  1) COSA È:
  Lo spread è la differenza tra il prezzo bid (prezzo al quale puoi VENDERE) 
  e il prezzo ask (prezzo al quale puoi COMPRARE), misurato in pips. 
  Rappresenta il costo immediato di ogni transazione.
  
  Esempio: EUR/USD Bid 1.0850 / Ask 1.0851 = Spread di 1.0 pip
  
  2) COME E QUANDO SI USA:
  - Monitoralo SEMPRE prima di aprire un trade
  - Usalo per scegliere il timing ottimale di ingresso
  - Confronta spread tra diversi broker
  - Evita trading quando spread >3× normale (eventi news, bassa liquidità)
  
  Quando è rilevante:
  - Scalping: CRITICO (spread mangia profitti su piccoli movimenti)
  - Day trading: IMPORTANTE (impatta breakeven e target)
  - Swing trading: MODERATO (meno rilevante su hold multi-day)
  
  3) PERCHÉ SI USA:
  - Calcolare il costo reale di ogni trade
  - Identificare condizioni di mercato (spread basso = alta liquidità)
  - Ottimizzare timing (evitare spread alti)
  - Valutare se un trade è conveniente (target deve superare spread)
  
  4) EFFETTI:
  
  Spread levels and interpretation:
  
  VERY LOW (0.0-0.5 pips):
  - Excellent liquidity, institutional-grade pricing
  - Ideal for scalping and high-frequency strategies
  - Typical during London/NY session overlap (13:00-17:00 GMT)
  - Major pairs: EUR/USD, GBP/USD with ECN brokers
  
  LOW (0.5-1.5 pips):
  - Good liquidity, standard retail pricing
  - Suitable for day trading and swing trading
  - Normal spread during active trading hours
  - Example: EUR/USD 0.8 pips, GBP/USD 1.2 pips
  
  MEDIUM (1.5-3.0 pips):
  - Moderate liquidity, acceptable for swing trades
  - Not ideal for scalping (costs eat into profits)
  - Common during Asian session or minor pairs
  - Example: EUR/JPY 2.0 pips, AUD/USD 1.8 pips
  
  HIGH (3.0-10.0 pips):
  - Low liquidity or volatile conditions
  - Significantly reduces profit potential
  - Common during news releases or exotic pairs
  - Avoid scalping, only swing trades with strong setups
  
  VERY HIGH (>10 pips):
  - Illiquid market or broker connectivity issues
  - Trading NOT recommended - costs too high
  - Typical during weekend gaps or major news shocks
  - Example: EUR/USD 15 pips during NFP announcement
  
  Impact on trading:
  - Scalping: Requires spreads <1 pip to be profitable
  - Day trading: Spreads <2 pips preferred
  - Swing trading: Can tolerate spreads up to 3-5 pips
  - Position trading: Spread impact minimal (holding days/weeks)
  
  Real-time updates: Yes (changes with market liquidity)
  ```

##### Pip Change Indicator
- **What it is**: Price movement since last update (or session start)
- **What it serves**: Momentum indicator showing which pairs are trending
- **Color coding**:
  - **Green (positive)**: Price moving up - bullish momentum
  - **Red (negative)**: Price moving down - bearish momentum
  - **Gray (zero)**: No significant movement - ranging/consolidation
- **Magnitude interpretation**:
  - **< 10 pips**: Low volatility, quiet market
  - **10-30 pips**: Normal intraday movement
  - **30-60 pips**: High volatility, trending market
  - **> 60 pips**: Extreme movement, news event or strong trend
- **Tooltip**:
  ```
  Pip Change Indicator
  
  What it is:
  The price change in pips since the last reference point (market open, 
  session start, or last chart reload). Shows momentum and direction.
  
  What it serves:
  - Quick momentum scanning (which pairs are moving)
  - Trend strength indication
  - Correlation confirmation (related pairs should move together)
  - Entry timing (wait for pullbacks in strong trends)
  
  Color coding:
  - GREEN: Positive change (price increased) - bullish momentum
  - RED: Negative change (price decreased) - bearish momentum
  - GRAY: No significant change - market consolidating
  
  Magnitude interpretation:
  
  VERY LOW (<10 pips):
  - Quiet market, low volatility
  - Typical during Asian session or holiday periods
  - Not ideal for day trading (insufficient movement)
  - Good for range-bound strategies (support/resistance bounces)
  
  LOW (10-30 pips):
  - Normal intraday movement
  - Typical for major pairs during active sessions
  - Suitable for day trading with standard targets
  - Example: EUR/USD +25 pips during London session
  
  MEDIUM (30-60 pips):
  - High volatility, strong trending movement
  - Typical during overlapping sessions or economic data releases
  - Excellent for trend-following strategies
  - Caution: Higher risk, wider stops needed
  
  HIGH (60-100 pips):
  - Very high volatility, major news event likely
  - Strong directional bias, momentum likely to continue
  - Ideal for breakout and trend strategies
  - Risk: Potential for sharp reversals if news-driven
  
  EXTREME (>100 pips):
  - Exceptional movement, major fundamental event
  - Central bank decision, geopolitical shock, or crisis
  - Experienced traders only - extreme volatility
  - Risk: Gap risk, slippage, broker requotes likely
  
  Trading strategies by pip change:
  - <10 pips: Range trading, fade extremes, scalp reversals
  - 10-30 pips: Standard day trading, momentum continuation
  - 30-60 pips: Trend following, wait for pullbacks to enter
  - >60 pips: Breakout trading, ride strong momentum (expert only)
  
  Update frequency: Real-time (every tick)
  Reset period: Configurable (daily session, market open, manual)
  ```

#### 1.1.2 VIX Widget (Volatility Index)

**Purpose**: Displays CBOE Volatility Index (VIX) as a market fear gauge and position sizing adjustment factor.

**Parameters**:

##### VIX Level Display
- **What it is**: Current VIX index value (0-50+ scale)
- **What it serves**: Market volatility and risk sentiment indicator - used to adjust position sizing
- **Classification levels**:
  - **Complacency (VIX < 12)**: Extremely low volatility, market calm - increase position size slightly (0.95× multiplier)
  - **Normal (VIX 12-20)**: Standard volatility, balanced risk - standard position size (1.0× multiplier)
  - **Concern (VIX 20-30)**: Elevated volatility, market anxiety - reduce position size (0.85× multiplier)
  - **Fear (VIX > 30)**: High volatility, panic conditions - significantly reduce position size (0.7× multiplier)
- **Tooltip**:
  ```
  VIX - Volatility Index (Fear Gauge)
  
  What it is:
  The CBOE Volatility Index (VIX) measures expected 30-day volatility in 
  the S&P 500 based on options prices. Often called the "fear gauge" or 
  "fear index" because it spikes during market stress.
  
  What it serves:
  - Market risk sentiment indicator
  - Position sizing adjustment (reduce size in high VIX environments)
  - Crisis detection (VIX >40 signals extreme stress)
  - Correlation with FX volatility (high VIX often means high FX volatility)
  
  Why it matters for Forex:
  Although VIX measures stock market volatility, it strongly correlates with:
  - USD strength (flight to safety increases USD demand when VIX spikes)
  - Risk appetite (low VIX = risk-on, high VIX = risk-off)
  - Carry trade activity (high VIX kills carry trades due to risk aversion)
  - Overall FX volatility (turbulent stocks often mean turbulent currencies)
  
  VIX levels and trading implications:
  
  COMPLACENCY (VIX < 12):
  What it means:
  - Extremely low volatility and investor complacency
  - Markets in "Goldilocks" mode - stable, predictable
  - Historically precedes major corrections (calm before storm)
  
  Market conditions:
  - Very tight trading ranges
  - Low risk premium in all assets
  - Carry trades and risk-on strategies thriving
  - Investors overly confident, underestimating risk
  
  Trading strategy:
  - Slightly INCREASE position size (0.95× → modest boost)
  - Low volatility means tighter stops work well
  - Good for mean-reversion strategies
  - WARNING: Be prepared for sudden volatility spike
  
  Position sizing: 0.95× (5% increase from normal)
  Risk: Complacency can end abruptly - use tight stops
  
  NORMAL (VIX 12-20):
  What it means:
  - Standard market volatility, balanced risk environment
  - Healthy two-way price action
  - Normal economic conditions, no major crises
  
  Market conditions:
  - Moderate intraday price swings
  - News events move markets predictably
  - Technical analysis works well
  - Risk/reward calculations reliable
  
  Trading strategy:
  - STANDARD position size (1.0× baseline)
  - Most strategies work in this environment
  - Both trend and range strategies viable
  - Stick to your trading plan without adjustments
  
  Position sizing: 1.0× (no adjustment)
  Risk: Normal market risk - standard risk management applies
  
  CONCERN (VIX 20-30):
  What it means:
  - Elevated volatility, investors becoming nervous
  - Uncertainty about economic outlook or geopolitics
  - Increased hedging activity
  - Risk premium rising across all asset classes
  
  Market conditions:
  - Wider daily ranges, more choppy price action
  - News events cause exaggerated reactions
  - Increased correlation (everything moves together)
  - Stop hunts and false breakouts more common
  
  Trading strategy:
  - REDUCE position size (0.85× - down 15%)
  - Widen stop losses (more volatility means more noise)
  - Favor trend-following over mean-reversion
  - Be selective - only take highest-conviction setups
  
  Position sizing: 0.85× (15% reduction)
  Risk: Heightened volatility - larger swings, more unpredictability
  
  FEAR (VIX > 30):
  What it means:
  - High volatility, panic conditions in markets
  - Major crisis, economic shock, or geopolitical event
  - Flight to safety (USD and safe havens bid)
  - Institutional de-risking and margin calls
  
  Market conditions:
  - Extreme intraday swings (100+ pip moves in minutes)
  - Liquidity drying up (wider spreads, gaps)
  - Technical levels frequently violated
  - Correlations break down (chaos mode)
  
  Trading strategy:
  - SIGNIFICANTLY REDUCE position size (0.7× - down 30%)
  - Use very wide stops or avoid trading altogether
  - Only trade with trend (don't fight panic)
  - Cash is a position - preservation of capital paramount
  
  Position sizing: 0.7× (30% reduction)
  Risk: EXTREME - potential for flash crashes, gaps, broker issues
  
  EXTREME FEAR (VIX > 40):
  - Market panic, systemic crisis (2008, COVID-19, major war)
  - Normal trading strategies fail
  - Consider stopping all trading until VIX < 30
  - Capital preservation mode only
  
  Historical VIX examples:
  - VIX 9-11 (2017): Record low volatility, calm markets
  - VIX 15-18 (2019): Normal bull market conditions
  - VIX 25-35 (2020 pre-COVID): Concerns building
  - VIX 82 (March 2020): COVID-19 peak panic
  - VIX 35-45 (2022): Ukraine war, inflation fears
  
  How ForexGPT uses VIX:
  1. Automatic position sizing adjustment (multiplier applied to all trades)
  2. Risk management layer (scales down exposure in volatile conditions)
  3. Regime detection input (high VIX = risk-off regime)
  4. Signal filtering (may skip signals when VIX > threshold)
  
  Update frequency: Every 5 minutes (fetched from Yahoo Finance)
  Data source: CBOE (Chicago Board Options Exchange)
  Calculation: 30-day implied volatility from S&P 500 options
  ```

##### VIX Progress Bar
- **What it is**: Visual representation of current VIX level on 0-50 scale
- **What it serves**: Quick visual reference for volatility level
- **Color coding**:
  - **Green**: VIX < 12 or 12-20 (Complacency/Normal) - calm market
  - **Yellow**: VIX 20-30 (Concern) - elevated volatility
  - **Orange**: VIX 30-40 (Fear) - high volatility
  - **Red**: VIX > 40 (Extreme Fear) - panic conditions
- **Tooltip**:
  ```
  VIX Progress Bar - Visual Volatility Gauge
  
  What it is:
  A horizontal bar showing the current VIX level on a 0-50 scale.
  Bar fills from left to right as VIX increases.
  
  What it serves:
  - Quick visual check of market stress level
  - At-a-glance volatility assessment (no need to read numbers)
  - Color-coded risk alerts
  
  Color interpretation:
  
  GREEN (VIX < 20):
  - Calm market, normal trading conditions
  - Standard risk management applies
  - All strategies viable
  
  YELLOW (VIX 20-30):
  - Market showing concern, volatility rising
  - Adjust position sizes down 15%
  - Be more selective with trade entries
  
  ORANGE (VIX 30-40):
  - Fear setting in, high volatility regime
  - Reduce positions by 30%
  - Consider defensive strategies only
  
  RED (VIX > 40):
  - Extreme panic, crisis mode
  - Minimize trading or stop completely
  - Focus on capital preservation
  
  Bar fill percentage:
  - 0-24%: Complacency/Normal (VIX 0-12)
  - 24-40%: Normal (VIX 12-20)
  - 40-60%: Concern (VIX 20-30)
  - 60-80%: Fear (VIX 30-40)
  - 80-100%: Extreme Fear (VIX 40-50+)
  
  Note: Scale tops at 50, but VIX can exceed this during extreme events.
  Historical peak: VIX 82.69 (March 16, 2020 - COVID panic)
  
  Use this for: Quick glance risk assessment before opening trades
  ```

##### VIX Classification Label
- **What it is**: Text label showing VIX classification (Complacency/Normal/Concern/Fear)
- **What it serves**: Named category for quick interpretation of volatility regime
- **Categories**:
  - **"Complacency"**: VIX < 12 (green text)
  - **"Normal"**: VIX 12-20 (green text)
  - **"Concern"**: VIX 20-30 (yellow text)
  - **"Fear"**: VIX > 30 (red text)
- **Tooltip**: (Same as VIX Level Display above - categories explained in detail)

#### 1.1.3 Order Books Widget (DOM - Depth of Market)

**Purpose**: Displays aggregated bid/ask order book depth from broker, showing liquidity and potential support/resistance.

**Parameters**:

##### Order Book Depth Display
- **What it is**: Aggregated order sizes at each price level (bid and ask side)
- **What it serves**: Liquidity analysis, large order detection, support/resistance levels
- **Display format**:
  ```
  Ask Side (Sell Orders)
  1.0854 | 45.2 lots
  1.0853 | 67.8 lots
  1.0852 | 123.4 lots ← Best Ask (current market price to BUY)
  ──────────────────
  1.0851 | 98.7 lots  ← Best Bid (current market price to SELL)
  1.0850 | 54.3 lots
  1.0849 | 32.1 lots
  Bid Side (Buy Orders)
  ```
- **Interpretation**:
  - **Large orders (>100 lots)**: Potential support/resistance, institutional activity
  - **Thin book (small orders)**: Low liquidity, risk of slippage
  - **Bid > Ask depth**: Buying pressure (bullish)
  - **Ask > Bid depth**: Selling pressure (bearish)
- **Tooltip**:
  ```
  Order Books (Depth of Market - DOM)
  
  What it is:
  A real-time display of all pending buy (bid) and sell (ask) orders at 
  each price level, aggregated from the broker's liquidity providers. 
  Shows how much volume is waiting to be traded at each price.
  
  What it serves:
  - Liquidity analysis (can your order be filled without moving the market?)
  - Support/resistance detection (large orders act as price barriers)
  - Order flow imbalance (more bids = buying pressure, more asks = selling)
  - Institutional activity detection (very large orders = smart money)
  
  How to read the display:
  
  ASK SIDE (top, red/orange):
  - Sell orders waiting to be matched
  - Price levels above current market price
  - If you want to BUY, you "lift the offer" from this side
  - Example: 1.0852 | 123.4 lots = 123.4 lots available to buy at 1.0852
  
  BID SIDE (bottom, green):
  - Buy orders waiting to be matched
  - Price levels below current market price
  - If you want to SELL, you "hit the bid" from this side
  - Example: 1.0851 | 98.7 lots = 98.7 lots available to sell at 1.0851
  
  ──────────────── (separator line):
  - Marks the best bid/ask spread
  - Current market price is between these levels
  
  Order size interpretation:
  
  VERY SMALL (<10 lots):
  - Retail traders or small institutional orders
  - Low liquidity at this level
  - Your order may cause slippage if >5 lots
  - Typical in exotic pairs or off-peak hours
  
  SMALL (10-50 lots):
  - Normal retail/small institutional activity
  - Moderate liquidity, acceptable for most retail traders
  - Orders <20 lots should fill without issues
  - Typical in minor pairs during active sessions
  
  MEDIUM (50-100 lots):
  - Good liquidity, institutional interest
  - Large retail or small bank orders
  - Orders <50 lots will fill cleanly
  - Common in major pairs during London/NY sessions
  
  LARGE (100-500 lots):
  - HIGH liquidity level, strong institutional presence
  - Likely bank or fund orders
  - Price level may act as support/resistance
  - Your order will NOT move the market
  - Ideal for scalping and high-frequency strategies
  
  VERY LARGE (>500 lots):
  - VERY HIGH liquidity, major institutional activity
  - Central bank, sovereign fund, or large hedge fund
  - Strong support/resistance likely (price may bounce here)
  - "Iceberg order" possible (hidden size, only partial shown)
  - Price absorption level - market may stall or reverse here
  
  Order flow imbalance (bid vs ask depth):
  
  BID DEPTH > ASK DEPTH (e.g., Bid 300 lots vs Ask 150 lots):
  What it means:
  - More buyers than sellers at current levels
  - Buying pressure building
  - Bullish imbalance
  
  Trading implications:
  - Price likely to move UP (buyers will lift offers)
  - Support building below (large bid orders prevent drops)
  - Good for LONG entries if confirmed by other signals
  - Imbalance >30%: Strong bullish pressure
  - Imbalance >50%: Very strong bullish pressure (buy surge likely)
  
  ASK DEPTH > BID DEPTH (e.g., Ask 300 lots vs Bid 150 lots):
  What it means:
  - More sellers than buyers at current levels
  - Selling pressure building
  - Bearish imbalance
  
  Trading implications:
  - Price likely to move DOWN (sellers will hit bids)
  - Resistance forming above (large ask orders prevent rallies)
  - Good for SHORT entries if confirmed by other signals
  - Imbalance >30%: Strong bearish pressure
  - Imbalance >50%: Very strong bearish pressure (sell-off likely)
  
  BALANCED (Bid ≈ Ask, within 10%):
  What it means:
  - Equilibrium, no strong directional bias
  - Market in consolidation or indecision
  
  Trading implications:
  - Range-bound conditions, wait for breakout
  - Support and resistance levels balanced
  - Mean-reversion strategies favored
  - Avoid momentum strategies until imbalance develops
  
  Advanced order book patterns:
  
  SPOOFING (pattern to watch for):
  - Large order appears suddenly
  - Order pulls away when price approaches
  - Manipulation tactic (illegal but still happens)
  - Don't rely solely on large orders for S/R
  
  ICEBERGS (hidden orders):
  - Order book shows 50 lots
  - After 50 lots fill, another 50 appears (and repeats)
  - True size hidden (could be 500+ lots total)
  - Signature: Orders refill instantly after being hit
  
  ORDER ABSORPTION:
  - Price approaches large order
  - Large order gets filled (absorbed by aggressive traders)
  - If refills: Strong support/resistance confirmed
  - If disappears: Fake liquidity, spoof detected
  
  How ForexGPT uses Order Book data:
  1. Liquidity scoring (prefer trading when DOM is thick)
  2. Imbalance signals (>30% imbalance triggers order flow indicator)
  3. Support/resistance levels (large orders create automatic S/R zones)
  4. Slippage prediction (thin book = widen acceptable slippage)
  5. Entry timing (wait for imbalance to build before entering)
  
  DOM update frequency: Every 1-3 seconds (depends on broker feed)
  Data source: Broker DOM feed (cTrader, MT5, etc.)
  Aggregation: All liquidity providers summed per price level
  Depth levels shown: Configurable (default: 10 levels each side)
  
  Caution:
  - DOM is NOT predictive (orders can be pulled anytime)
  - Use as confluence with technical analysis, not standalone
  - In fast markets, DOM changes rapidly (less reliable)
  - Some brokers show partial book (not true market depth)
  ```

##### Spread Indicator (in Order Books)
- **What it is**: Current bid-ask spread displayed above order book
- **What it serves**: Transaction cost awareness (same as Market Watch spread)
- **Tooltip**: (Same detailed spread explanation as in Market Watch section)

##### Imbalance Percentage
- **What it is**: Calculated bid/ask imbalance as percentage
- **Formula**: `(Bid Depth - Ask Depth) / (Bid Depth + Ask Depth) × 100`
- **Range**: -100% (all asks) to +100% (all bids)
- **Interpretation**:
  - **Negative (e.g., -35%)**: Bearish imbalance, selling pressure
  - **Zero (0%)**: Balanced, no directional bias
  - **Positive (e.g., +45%)**: Bullish imbalance, buying pressure
- **Tooltip**:
  ```
  Order Flow Imbalance Percentage
  
  What it is:
  A percentage calculation showing the difference between total bid depth 
  and total ask depth in the order book. Positive = more bids (buying pressure), 
  Negative = more asks (selling pressure).
  
  Formula:
  Imbalance% = ((Total Bid Depth - Total Ask Depth) / (Total Bid + Total Ask)) × 100
  
  Example calculation:
  - Total Bid Depth: 300 lots
  - Total Ask Depth: 200 lots
  - Imbalance = ((300 - 200) / (300 + 200)) × 100 = (100 / 500) × 100 = +20%
  
  What it serves:
  - Quick measure of buying vs selling pressure
  - Entry timing signal (enter when imbalance confirms your bias)
  - Reversal warning (extreme imbalance often precedes reversal)
  - Confluence with other indicators
  
  Imbalance levels and trading implications:
  
  EXTREME SELLING PRESSURE (-100% to -50%):
  What it means:
  - Order book overwhelmingly filled with sell orders
  - Very few buyers willing to step in
  - Panic selling or major distribution event
  
  Market behavior:
  - Price likely to continue falling (sellers will hit lower bids)
  - Bid side thin = low support, price can drop fast
  - Potential capitulation or major news event
  
  Trading strategy:
  - SHORT bias confirmed (if you're already short, hold)
  - DO NOT try to catch falling knife (buy) until imbalance reverses
  - Wait for imbalance to improve to at least -30% before considering longs
  - Set alerts for imbalance improvement (reversal signal)
  
  Risk: Trying to buy into extreme selling usually fails
  
  STRONG SELLING PRESSURE (-50% to -30%):
  What it means:
  - Significant selling pressure, but not panic
  - Sellers outnumber buyers 2:1 or more
  - Bearish bias, but potential for bounces
  
  Market behavior:
  - Downtrend likely to continue
  - Rallies will be sold (resistance from ask orders)
  - Support levels may break easily
  
  Trading strategy:
  - Short entries favored (trade with the imbalance)
  - Long entries only at major support with confirmation
  - Use tight stops on longs (selling pressure can resume)
  - Scale into shorts on rallies (fade strength)
  
  MODERATE SELLING PRESSURE (-30% to -10%):
  What it means:
  - Mild bearish bias, but not extreme
  - Sellers have slight edge
  - Market leaning down but buyers still present
  
  Market behavior:
  - Slight downward drift likely
  - Not strong enough to ignore bullish technical setups
  - Consolidation or slow decline possible
  
  Trading strategy:
  - Neutral to slight short bias
  - Both longs and shorts viable (follow technicals)
  - Imbalance can flip easily (not a strong signal)
  - Use as tie-breaker when other signals mixed
  
  BALANCED (-10% to +10%):
  What it means:
  - No significant buying or selling pressure
  - Bid and ask depth roughly equal
  - Market indecision or equilibrium
  
  Market behavior:
  - Range-bound price action likely
  - Price oscillating around fair value
  - Waiting for catalyst (news, breakout, etc.)
  
  Trading strategy:
  - NO directional bias from order flow
  - Use mean-reversion strategies (buy dips, sell rallies)
  - Wait for imbalance to develop before momentum trades
  - Fade extremes within range
  - Good for scalping narrow ranges
  
  MODERATE BUYING PRESSURE (+10% to +30%):
  What it means:
  - Mild bullish bias, buyers have slight edge
  - Support building as bids accumulate
  - Market leaning up but sellers still present
  
  Market behavior:
  - Slight upward drift likely
  - Not strong enough to ignore bearish technical setups
  - Consolidation or slow climb possible
  
  Trading strategy:
  - Neutral to slight long bias
  - Both longs and shorts viable (follow technicals)
  - Imbalance can flip easily (not a strong signal)
  - Use as tie-breaker when other signals mixed
  
  STRONG BUYING PRESSURE (+30% to +50%):
  What it means:
  - Significant buying pressure, buyers outnumber sellers 2:1+
  - Bullish bias, strong support from large bid orders
  - Dips likely to be bought
  
  Market behavior:
  - Uptrend likely to continue
  - Selloffs will be bought (support from bid orders)
  - Resistance levels may break easily
  
  Trading strategy:
  - Long entries favored (trade with the imbalance)
  - Short entries only at major resistance with confirmation
  - Use tight stops on shorts (buying pressure can resume)
  - Buy dips (strong bids will support price)
  
  EXTREME BUYING PRESSURE (+50% to +100%):
  What it means:
  - Order book overwhelmingly filled with buy orders
  - Very few sellers willing to step in
  - Euphoric buying or major accumulation event
  
  Market behavior:
  - Price likely to continue rising (buyers will lift offers)
  - Ask side thin = low resistance, price can spike fast
  - Potential blow-off top or major FOMO event
  
  Trading strategy:
  - LONG bias confirmed (if you're already long, hold)
  - DO NOT try to short into extreme buying (will get squeezed)
  - Wait for imbalance to cool to at least +30% before considering shorts
  - WARNING: Extreme imbalances often reverse sharply (parabolic blow-off)
  
  Risk: Shorting into extreme buying usually fails (until it doesn't)
  
  Imbalance persistence:
  - If imbalance persists >5 minutes: Strong signal (trade with it)
  - If imbalance flips rapidly (<1 minute): Weak signal (ignore)
  - If imbalance oscillates around zero: Balanced market (range-bound)
  
  Imbalance + Price action:
  - Strong imbalance + price moving opposite direction: DIVERGENCE (reversal signal)
    Example: +40% bullish imbalance but price dropping = sellers overwhelming buyers
  - Strong imbalance + price moving same direction: CONFIRMATION (trend continuation)
    Example: +40% bullish imbalance and price rising = healthy uptrend
  
  How ForexGPT uses Imbalance:
  1. Signal filtering (skip trades against strong imbalance >30%)
  2. Entry timing (wait for imbalance to align with trade direction)
  3. Position sizing (increase size when imbalance >40% in trade direction)
  4. Stop placement (tighten stops when imbalance opposed to position)
  5. Divergence detection (imbalance disagrees with price = reversal warning)
  
  Update frequency: Real-time (every DOM update, typically 1-3 seconds)
  Calculation window: All visible order book levels (typically 10 levels each side)
  ```

#### 1.1.4 Order Flow Panel

**Purpose**: Displays real-time order flow imbalance with visual progress bar and detailed bid/ask metrics.

**Parameters**:

##### Order Flow Progress Bar
- **What it is**: Horizontal bar showing imbalance from -100% (all selling) to +100% (all buying)
- **What it serves**: Visual representation of order flow bias
- **Color gradient**:
  - **Red (left)**: Bearish imbalance (selling pressure)
  - **Gray (center)**: Balanced (neutral)
  - **Green (right)**: Bullish imbalance (buying pressure)
- **Tooltip**:
  ```
  Order Flow Progress Bar
  
  What it is:
  A visual representation of order flow imbalance as a horizontal progress bar.
  Bar fills from center to left (red) for selling pressure, or center to right 
  (green) for buying pressure.
  
  What it serves:
  - At-a-glance order flow bias assessment
  - Quick confirmation of directional pressure
  - Visual complement to imbalance percentage
  
  Color gradient meaning:
  
  FAR LEFT (Red, -100% to -50%):
  - Extreme selling pressure
  - Order book dominated by sell orders
  - Bearish extreme - consider SHORT bias
  - WARNING: May signal capitulation (reversal soon)
  
  CENTER-LEFT (Orange, -50% to -10%):
  - Moderate selling pressure
  - More sellers than buyers, but not extreme
  - Slight bearish bias
  
  CENTER (Gray, -10% to +10%):
  - Balanced market
  - No strong directional bias from order flow
  - Range-bound conditions likely
  
  CENTER-RIGHT (Light Green, +10% to +50%):
  - Moderate buying pressure
  - More buyers than sellers, but not extreme
  - Slight bullish bias
  
  FAR RIGHT (Green, +50% to +100%):
  - Extreme buying pressure
  - Order book dominated by buy orders
  - Bullish extreme - consider LONG bias
  - WARNING: May signal euphoria (reversal soon)
  
  How to use:
  - Quick glance before entering trade (does order flow agree with your bias?)
  - Combined with price action (divergences signal reversals)
  - Entry timing (wait for bar to align with your direction)
  
  Update frequency: Real-time (every 1-3 seconds)
  ```

##### Bid/Ask Volume Labels
- **What it is**: Detailed breakdown of total bid and ask volume
- **What it serves**: Raw data behind the imbalance calculation
- **Display format**: `Bid: 300.5 lots | Ask: 215.3 lots`
- **Tooltip**:
  ```
  Bid/Ask Volume Breakdown
  
  What it is:
  The total volume (in lots) of all pending bid orders vs all pending ask orders 
  currently visible in the order book. Raw data used to calculate imbalance percentage.
  
  What it serves:
  - Transparency into imbalance calculation
  - Absolute volume assessment (not just percentage)
  - Liquidity depth check
  
  How to interpret:
  
  Bid Volume:
  - Total lots waiting to BUY at all bid price levels
  - Higher bid volume = more buying interest = support
  - Example: "Bid: 300.5 lots" = 300.5 lots willing to buy at various prices
  
  Ask Volume:
  - Total lots waiting to SELL at all ask price levels
  - Higher ask volume = more selling interest = resistance
  - Example: "Ask: 215.3 lots" = 215.3 lots willing to sell at various prices
  
  Volume levels interpretation:
  
  VERY LOW (<50 lots total):
  - Illiquid market, thin order book
  - High risk of slippage
  - Typical in exotic pairs or off-hours
  - Avoid large orders (will move market)
  
  LOW (50-200 lots total):
  - Moderate liquidity
  - Acceptable for small retail orders (<10 lots)
  - Common in minor pairs or Asian session
  
  MEDIUM (200-500 lots total):
  - Good liquidity
  - Suitable for most retail trading (<50 lots)
  - Normal for major pairs during active sessions
  
  HIGH (500-1000 lots total):
  - High liquidity
  - Institutional presence
  - Orders up to 100 lots will fill cleanly
  - Ideal for larger retail and small institutional
  
  VERY HIGH (>1000 lots total):
  - Very high liquidity
  - Major institutional activity
  - Even 500+ lot orders won't move market
  - Excellent for scalping and HFT strategies
  
  Bid vs Ask comparison:
  
  Bid > Ask (e.g., Bid 300 vs Ask 200):
  - More buyers than sellers (bullish)
  - Imbalance = +20% (calculated as explained earlier)
  - Buying pressure building
  
  Ask > Bid (e.g., Ask 300 vs Bid 200):
  - More sellers than buyers (bearish)
  - Imbalance = -20%
  - Selling pressure building
  
  Bid ≈ Ask (e.g., Bid 250 vs Ask 255):
  - Balanced market
  - Imbalance ≈ -1% (nearly neutral)
  - No strong directional bias
  
  Combined with price action:
  - Bid 300 + Ask 200 + price rising = healthy uptrend (confirmation)
  - Bid 300 + Ask 200 + price falling = divergence (reversal signal)
  - Bid 200 + Ask 300 + price rising = struggling uptrend (weak, may reverse)
  - Bid 200 + Ask 300 + price falling = healthy downtrend (confirmation)
  
  Update frequency: Real-time (every DOM update)
  Depth: Typically 10 levels each side (configurable)
  ```

### 1.2 Center Area - Main Chart

#### 1.2.1 Drawing Toolbar

**Purpose**: Provides drawing tools for technical analysis (trendlines, channels, Fibonacci, etc.)

**Tools Available**:

##### Trendline Tool
- **What it is**: Draw straight lines connecting price points
- **What it serves**: Identify support/resistance trends, breakout levels
- **How to use**: Click start point, click end point
- **Tooltip**:
  ```
  Trendline Drawing Tool
  
  What it is:
  A straight line connecting two or more price points, used to visualize 
  the direction and strength of a trend. Most fundamental technical analysis tool.
  
  What it serves:
  - Trend identification (uptrend = higher lows, downtrend = lower highs)
  - Support/resistance detection (price bounces off trendline)
  - Breakout signals (price breaking trendline = trend change)
  - Entry/exit timing (enter at trendline bounces, exit at breaks)
  
  How to draw:
  1. Select Trendline tool (icon: diagonal line)
  2. Click on first price point (swing low for uptrend, swing high for downtrend)
  3. Click on second price point (another swing low/high in same direction)
  4. Line extends automatically into the future
  
  Trendline types:
  
  UPTREND LINE (ascending):
  - Connect series of higher lows
  - Acts as dynamic support (price bounces up from line)
  - Buy signals when price approaches line and bounces
  - Break below line = trend reversal signal (exit longs or go short)
  
  DOWNTREND LINE (descending):
  - Connect series of lower highs
  - Acts as dynamic resistance (price bounces down from line)
  - Sell signals when price approaches line and bounces
  - Break above line = trend reversal signal (exit shorts or go long)
  
  Trendline strength (reliability):
  
  WEAK (2 touches):
  - Minimum for valid trendline
  - May be coincidence, not yet proven
  - Use with caution, wait for 3rd touch to confirm
  
  MODERATE (3 touches):
  - Confirmed trendline
  - Market respecting this level
  - Reliable for trading bounces
  
  STRONG (4+ touches):
  - Very reliable trendline
  - Institutionally recognized level
  - Breaks are significant events
  - High probability of bounce on next approach
  
  Trendline angle:
  
  STEEP (>45° angle):
  - Unsustainable pace
  - Likely to break or flatten
  - Parabolic move, risk of sharp reversal
  - Don't rely on steep trendlines for long
  
  MODERATE (30-45° angle):
  - Healthy sustainable trend
  - Ideal for trend-following strategies
  - Most reliable trendlines
  
  SHALLOW (<30° angle):
  - Weak trend, more like sideways consolidation
  - Less reliable, price drifts through easily
  - Consider range-bound strategies instead
  
  Trendline break confirmation:
  
  FALSE BREAK (wick through, closes back):
  - Price briefly penetrates but candle closes on original side
  - Trendline still valid (bear trap or bull trap)
  - Often precedes strong move in original trend direction
  
  TRUE BREAK (body closes beyond):
  - Candle body closes beyond trendline (not just wick)
  - Trendline invalidated
  - Wait for retest (price returns to line) before entering reversal trade
  
  How to use in trading:
  
  Bounce strategy (trend continuation):
  1. Identify valid trendline (3+ touches)
  2. Wait for price to approach trendline
  3. Look for reversal patterns (hammer, engulfing, etc.)
  4. Enter in trend direction when price bounces
  5. Stop loss just beyond trendline
  6. Target: next swing high (uptrend) or swing low (downtrend)
  
  Breakout strategy (trend reversal):
  1. Identify mature trendline (4+ touches)
  2. Watch for price approaching line with momentum
  3. Enter when candle closes beyond line (confirmed break)
  4. Wait for retest (price pulls back to line) for better entry
  5. Stop loss back inside original trend
  6. Target: measured move (distance of previous swing)
  
  Common mistakes:
  - Forcing trendline to fit (cherry-picking points)
  - Drawing on wicks instead of closes
  - Not waiting for confirmation before trading
  - Ignoring breaks (holding losing trades hoping for bounce)
  
  Best practices:
  - Use logarithmic scale for long-term trendlines
  - Draw multiple trendlines (upper/lower bounds of channel)
  - Combine with other indicators (RSI, MACD) for confirmation
  - Respect breaks decisively (don't fight trend changes)
  
  Keyboard shortcuts:
  - T: Select trendline tool
  - Delete: Remove selected trendline
  - Ctrl+Click: Extend existing trendline
  ```

##### Horizontal Line Tool
- **What it is**: Draw horizontal support/resistance lines
- **What it serves**: Mark key price levels (S/R, entry, exit, stop loss)
- **Tooltip**:
  ```
  Horizontal Line Tool
  
  What it is:
  A horizontal line drawn at a specific price level, used to mark support, 
  resistance, entry points, stop losses, take profits, and other key levels.
  
  What it serves:
  - Support/resistance identification (price bounces at horizontal levels)
  - Entry/exit planning (mark intended entry and target prices)
  - Risk management (visualize stop loss and take profit levels)
  - Psychological levels (round numbers like 1.1000, 1.2000)
  
  How to draw:
  1. Select Horizontal Line tool
  2. Click at desired price level
  3. Line extends across entire chart at that price
  4. Drag up/down to adjust, double-click to edit exact price
  
  Use cases:
  
  SUPPORT LEVELS:
  What it is:
  - Price level where buying interest historically emerged
  - Previous swing lows, consolidation bottoms, breakout origins
  
  How to identify:
  - Mark previous low points where price bounced 2+ times
  - Round numbers (1.0500, 1.1000) often act as support
  - Previous resistance becomes support after breakout (role reversal)
  
  How to trade:
  - Buy near support (anticipate bounce)
  - Stop loss just below support (invalidation level)
  - If support breaks, becomes resistance (flip)
  
  RESISTANCE LEVELS:
  What it is:
  - Price level where selling interest historically emerged
  - Previous swing highs, consolidation tops, breakdown origins
  
  How to identify:
  - Mark previous high points where price rejected 2+ times
  - Round numbers (1.2000, 1.3000) often act as resistance
  - Previous support becomes resistance after breakdown (role reversal)
  
  How to trade:
  - Sell near resistance (anticipate rejection)
  - Stop loss just above resistance (invalidation level)
  - If resistance breaks, becomes support (flip)
  
  ENTRY LEVELS:
  - Mark your intended entry price before placing order
  - Visual reference for limit orders
  - Helps avoid emotional entry (stick to plan)
  
  STOP LOSS LEVELS:
  - Mark where you'll exit if trade goes against you
  - Typically below support (longs) or above resistance (shorts)
  - Visualize risk before entering trade
  
  TAKE PROFIT LEVELS:
  - Mark your profit target (exit when price reaches)
  - Based on risk/reward ratio (e.g., 2× or 3× risk distance)
  - Multiple TP levels possible (scale out at TP1, TP2, TP3)
  
  Level strength (how reliable):
  
  WEAK (1 touch):
  - Not yet proven
  - May be coincidence
  - Use with caution
  
  MODERATE (2-3 touches):
  - Confirmed S/R level
  - Market respecting this price
  - Reliable for trading
  
  STRONG (4+ touches):
  - Major S/R zone
  - Institutionally recognized
  - High probability of bounce
  - Breaks are very significant
  
  Support/Resistance zones (not exact lines):
  - Price rarely bounces at exact same level
  - Think of S/R as zones (e.g., 1.0850-1.0870)
  - Draw multiple horizontal lines to mark zone boundaries
  - Price often oscillates within zone before breaking out
  
  Round number psychology:
  
  MAJOR ROUND NUMBERS (e.g., 1.0000, 1.5000):
  - Very strong psychological S/R
  - Option strike prices cluster here
  - Institutions and algorithms key off these levels
  - Often see significant bounces or breakouts
  
  MINOR ROUND NUMBERS (e.g., 1.0850, 1.0900):
  - Moderate psychological S/R
  - Traders naturally place orders at "clean" levels
  - Still relevant but weaker than major rounds
  
  How ForexGPT uses horizontal lines:
  1. Automatic S/R detection algorithm draws these for you
  2. Manual lines you draw are saved and persist across sessions
  3. Price alerts can be set at horizontal line levels
  4. Backtesting engine respects your drawn S/R levels
  
  Best practices:
  - Don't clutter chart with too many lines (keep it clean)
  - Use different colors for S (green), R (red), entries (blue), stops (orange)
  - Label your lines (right-click → Add Label)
  - Review and delete outdated lines periodically
  - Focus on most recent and most-tested levels
  
  Keyboard shortcuts:
  - H: Select horizontal line tool
  - Delete: Remove selected line
  - Ctrl+C, Ctrl+V: Copy/paste lines to other charts
  ```

##### Fibonacci Retracement Tool
- **What it is**: Draw Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) between swing high and low
- **What it serves**: Identify potential reversal levels during pullbacks
- **Tooltip**:
  ```
  Fibonacci Retracement Tool
  
  What it is:
  A technical analysis tool based on the Fibonacci sequence (mathematical ratios). 
  Draws horizontal lines at key percentage levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) 
  between a swing high and swing low to identify potential reversal levels.
  
  What it serves:
  - Pullback/retracement prediction (where will price bounce in a trend?)
  - Entry timing (buy dips at Fib levels in uptrend, sell rallies in downtrend)
  - Confluence with other indicators (Fib + S/R = high-probability zone)
  - Extension targets (project price targets using Fib extensions)
  
  How to draw:
  
  For UPTRENDS (predicting pullback lows):
  1. Select Fibonacci tool
  2. Click on swing LOW (start of uptrend)
  3. Drag to swing HIGH (top of uptrend)
  4. Tool automatically draws retracement levels
  5. Price will likely bounce at one of these levels during pullback
  
  For DOWNTRENDS (predicting rally highs):
  1. Click on swing HIGH (start of downtrend)
  2. Drag to swing LOW (bottom of downtrend)
  3. Levels show where price may reverse during rally
  
  Fibonacci levels and their meaning:
  
  0.0% LEVEL (100% of move):
  - The swing high (uptrend) or swing low (downtrend)
  - Starting point of retracement
  - If price returns here, entire move is erased
  
  23.6% RETRACEMENT:
  What it is:
  - Shallow pullback, only 23.6% of prior move retraced
  - Very strong trend continuation signal
  
  When it holds:
  - Trend is very healthy and strong
  - Momentum is powerful
  - Ideal for aggressive trend-following entries
  
  Trading strategy:
  - Enter long (uptrend) or short (downtrend) if price bounces here
  - Stop loss below next Fib level (38.2%)
  - Target: New high (uptrend) or new low (downtrend)
  - Lower probability than deeper retracements but faster move
  
  38.2% RETRACEMENT:
  What it is:
  - Moderate pullback, 38.2% of move retraced
  - Most common retracement in strong trends
  - "Goldilocks zone" for entries
  
  When it holds:
  - Trend is healthy and likely to continue
  - Typical correction in strong uptrend/downtrend
  - High probability reversal zone
  
  Trading strategy:
  - Primary entry zone for trend continuation trades
  - Stop loss below 50% level
  - Target: 161.8% extension (new extreme)
  - BEST risk/reward ratio (not too shallow, not too deep)
  
  50.0% RETRACEMENT (not true Fibonacci, but widely used):
  What it is:
  - Halfway point of prior move
  - Psychological level (traders naturally think "half back")
  - Not mathematically Fibonacci but extremely relevant in practice
  
  When it holds:
  - Trend still intact but weakening
  - Last chance for trend continuation
  - If breaks, trend likely reversing
  
  Trading strategy:
  - Conservative entry for trend continuation
  - Stop loss below 61.8% level
  - Smaller position size (trend weaker than 38.2% entry)
  - Watch for confirmation (reversal pattern, volume, etc.)
  
  61.8% RETRACEMENT (THE GOLDEN RATIO):
  What it is:
  - The "golden ratio" (1.618), most famous Fibonacci level
  - Deep pullback, trend is weakening
  - Critical make-or-break level
  
  When it holds:
  - Trend hanging by a thread but still alive
  - Last viable support/resistance before reversal
  - High-risk, high-reward entry
  
  When it breaks:
  - Trend is REVERSING (not just correcting)
  - Previous trend now invalid
  - New opposite trend likely forming
  
  Trading strategy:
  - High-risk entry (only with strong confirmation)
  - Tight stop loss just below level
  - Reduced position size
  - If breaks, FLIP BIAS (consider counter-trend trade)
  
  78.6% RETRACEMENT:
  What it is:
  - Very deep pullback, almost entire move retraced
  - Trend essentially dead
  - Square root of 0.618 (mathematical relevance)
  
  When price reaches here:
  - Original trend is over
  - Consolidation or reversal forming
  - NOT a good trend continuation entry
  
  Trading strategy:
  - DO NOT enter trend continuation trades here
  - Wait for new pattern to form
  - Consider range-bound strategies
  - If bounces here, very weak trend (don't trust it)
  
  100.0% RETRACEMENT:
  - Entire prior move erased
  - Trend completely invalidated
  - Full reversal or deep consolidation
  
  Fibonacci Extensions (for targets):
  
  After price bounces at a Fib retracement level, where will it go?
  Use Fibonacci extensions to project targets:
  
  127.2% EXTENSION:
  - Conservative target
  - Slight overshoot of prior high/low
  - Take partial profits here
  
  161.8% EXTENSION (Golden ratio extension):
  - Primary target for trend continuation
  - Most common profit target
  - Price often reaches here before major correction
  
  200.0% EXTENSION:
  - Aggressive target
  - Parabolic move, strong trend
  - Only hit in very strong trends
  
  261.8% EXTENSION:
  - Extreme target
  - Rarely hit, requires exceptional trend
  - If reached, expect imminent reversal (overextended)
  
  Combining Fibonacci with other tools:
  
  Fib + Support/Resistance:
  - If 61.8% Fib aligns with previous S/R level = VERY HIGH probability zone
  - Confluence increases reliability dramatically
  - Example: 61.8% Fib at 1.0850 AND previous low at 1.0850 = strong support
  
  Fib + Trendlines:
  - Fib level coinciding with trendline = powerful confluence
  - Double confirmation of reversal zone
  
  Fib + Moving Averages:
  - 50% or 61.8% Fib near 200 EMA = major support/resistance
  - Institutional traders watch these levels closely
  
  Fib + Candlestick Patterns:
  - Hammer or engulfing pattern AT Fib level = strong entry signal
  - Doji at 61.8% = indecision, wait for confirmation
  
  Common Fibonacci mistakes:
  
  Wrong swing points:
  - Using minor highs/lows instead of major swings
  - Solution: Use higher timeframe to identify major swings (4H, Daily)
  
  Drawing backwards:
  - Must draw from OLD swing to NEW swing (chronological order)
  - Solution: Bottom-to-top (uptrend), top-to-bottom (downtrend)
  
  Expecting exact bounces:
  - Price rarely hits exact Fib level and bounces
  - Solution: Treat Fib levels as ZONES (±10 pips buffer)
  
  Over-reliance on Fibonacci:
  - Fib is NOT predictive, it's probabilistic
  - Solution: Combine with price action, patterns, indicators
  
  How ForexGPT uses Fibonacci:
  1. Automatic Fib detection on every major swing
  2. Confluence scoring (Fib + S/R + MA = high-quality signal)
  3. Entry suggestions at 38.2% and 61.8% levels
  4. Extension targets automatically calculated for TP placement
  
  Best practices:
  - Use Daily or 4H timeframe for drawing (ignore noise)
  - Redraw Fib levels as new swings form (update regularly)
  - Focus on 38.2%, 50%, 61.8% (most reliable levels)
  - Wait for confirmation before entering (don't blindly buy at Fib)
  - Use multiple timeframe Fib for confluence (e.g., 61.8% on 4H AND Daily)
  
  Keyboard shortcuts:
  - F: Select Fibonacci tool
  - Shift+F: Fibonacci extension tool
  - Delete: Remove selected Fibonacci drawing
  ```

##### Channel Tool
- **What it is**: Draw parallel trendlines (channel) containing price action
- **What it serves**: Identify range boundaries, anticipate breakouts
- **Tooltip**:
  ```
  Channel Tool (Parallel Trendlines)
  
  What it is:
  Two parallel trendlines (upper and lower boundary) that contain price action 
  within a defined range. Price oscillates between these boundaries until breakout.
  
  What it serves:
  - Range identification (market trading within defined boundaries)
  - Entry/exit timing (buy at lower boundary, sell at upper boundary)
  - Breakout detection (price exiting channel = trend change)
  - Volatility measurement (channel width = volatility)
  
  How to draw:
  
  ASCENDING CHANNEL (uptrend):
  1. Select Channel tool
  2. Draw lower trendline connecting swing lows (higher lows)
  3. Tool auto-draws parallel upper line through swing highs
  4. Adjust upper line if needed to fit highs
  
  DESCENDING CHANNEL (downtrend):
  1. Draw upper trendline connecting swing highs (lower highs)
  2. Tool auto-draws parallel lower line through swing lows
  3. Adjust lower line to fit
  
  HORIZONTAL CHANNEL (range):
  1. Draw horizontal line at support (lower boundary)
  2. Draw parallel horizontal line at resistance (upper boundary)
  
  Channel types and trading strategies:
  
  ASCENDING CHANNEL (bullish trend):
  What it is:
  - Price making higher highs and higher lows
  - Uptrend with defined boundaries
  - Both boundaries sloping upward
  
  Characteristics:
  - Lower boundary = dynamic support (buy zone)
  - Upper boundary = dynamic resistance (sell/TP zone)
  - Trend is UP but within controlled range
  
  Trading strategies:
  
  TREND CONTINUATION (buying the dips):
  1. Wait for price to approach lower channel line
  2. Look for bounce signal (hammer, bullish engulfing, etc.)
  3. Enter long when price bounces off lower boundary
  4. Stop loss below channel (confirmation of break)
  5. Target: Upper channel boundary
  6. Risk/Reward: Typically 1:2 or better
  
  When to use:
  - Clear uptrend with 3+ touches on each boundary
  - Channel width at least 50-100 pips (sufficient range)
  - Confirmed bounces at lower boundary previously
  
  BREAKOUT TRADING (channel break):
  1. Watch for price approaching upper boundary with momentum
  2. Enter long when price closes ABOVE upper channel line
  3. Stop loss: Re-entry into channel (close back below)
  4. Target: Measured move (channel width projected upward)
  
  When to use:
  - Price making increasingly strong pushes toward upper boundary
  - Volume increasing (if available)
  - Higher timeframe showing bullish pattern
  
  DESCENDING CHANNEL (bearish trend):
  What it is:
  - Price making lower highs and lower lows
  - Downtrend with defined boundaries
  - Both boundaries sloping downward
  
  Characteristics:
  - Upper boundary = dynamic resistance (sell zone)
  - Lower boundary = dynamic support (buy/TP zone for shorts)
  - Trend is DOWN but within controlled range
  
  Trading strategies:
  
  TREND CONTINUATION (selling the rallies):
  1. Wait for price to approach upper channel line
  2. Look for rejection signal (shooting star, bearish engulfing)
  3. Enter short when price bounces off upper boundary
  4. Stop loss above channel
  5. Target: Lower channel boundary
  
  BREAKOUT TRADING (channel break):
  1. Watch for price approaching lower boundary with momentum
  2. Enter short when price closes BELOW lower channel line
  3. Stop loss: Re-entry into channel
  4. Target: Measured move (channel width projected downward)
  
  HORIZONTAL CHANNEL (range-bound, sideways):
  What it is:
  - Price oscillating between horizontal support and resistance
  - No trend, consolidation phase
  - Equal highs and lows (no higher highs or lower lows)
  
  Characteristics:
  - Upper boundary = resistance (sell zone)
  - Lower boundary = support (buy zone)
  - Mean reversion dominates (price returns to middle)
  
  Trading strategies:
  
  RANGE TRADING (buy low, sell high):
  1. Buy at lower boundary (support)
     - Entry: When price touches support
     - Confirmation: Bullish reversal pattern
     - Stop: Just below support (20-30 pips)
     - Target: Upper boundary (resistance)
  
  2. Sell at upper boundary (resistance)
     - Entry: When price touches resistance
     - Confirmation: Bearish reversal pattern
     - Stop: Just above resistance (20-30 pips)
     - Target: Lower boundary (support)
  
  3. Fade the middle (advanced):
     - If price is in middle of channel, wait for extremes
     - Don't chase (let price come to boundaries)
  
  BREAKOUT TRADING (range breakout):
  1. Monitor for tightening range (consolidation before explosion)
  2. Enter when price closes outside channel (either direction)
  3. Direction of breakout = direction of new trend
  4. Stop loss: Re-entry into range
  5. Target: Channel width projected in breakout direction
  
  Breakout confirmation:
  - Candle CLOSE beyond boundary (not just wick)
  - Volume spike (if available)
  - Retest of broken boundary (now S/R flip)
  
  Channel width interpretation:
  
  VERY NARROW (<30 pips):
  - Extremely tight consolidation
  - Low volatility compression
  - Imminent breakout likely (spring coiling)
  - Don't range trade (risk/reward too small)
  - Prepare for breakout in either direction
  
  NARROW (30-60 pips):
  - Tight range, moderate consolidation
  - Can range trade but tight stops needed
  - Breakouts tend to be explosive
  
  MEDIUM (60-120 pips):
  - Ideal for range trading strategies
  - Sufficient room for profit after spread/commission
  - Scalp 2-3 times per boundary touch
  
  WIDE (120-200 pips):
  - Large range, good for swing trading
  - Multiple entry opportunities
  - Can scale in/out at different levels
  
  VERY WIDE (>200 pips):
  - Massive range, consider multiple sub-ranges
  - Long-term consolidation or slow trend
  - Break into smaller channels within the larger one
  
  Channel touches (reliability):
  
  2 TOUCHES (minimum):
  - Channel not yet confirmed
  - May be coincidence
  - Wait for 3rd touch before trading
  
  3-4 TOUCHES (confirmed):
  - Valid channel, market respecting boundaries
  - Safe to trade bounces
  - High probability of continuation
  
  5+ TOUCHES (mature):
  - Very strong channel
  - Institutional recognition
  - Reliable for trading
  - BUT: Mature channels more likely to break (watch for breakout)
  
  False breakouts (fakeouts):
  
  STOP HUNT:
  - Price briefly breaks boundary (triggers stops)
  - Immediately reverses back into channel
  - Common manipulation tactic
  
  How to avoid:
  - Wait for CLOSE outside channel (not just wick)
  - Require 2 consecutive closes outside for confirmation
  - Place stops wider (beyond likely stop hunt zones)
  
  EXHAUSTION WICK:
  - Long wick protruding from boundary
  - Body remains inside channel
  - Signal: Attempted break failed (trend still constrained)
  - Action: Fade the wick (enter opposite direction)
  
  Channel slope (angle):
  
  STEEP (>45° for ascending, <-45° for descending):
  - Unsustainable pace
  - Channel likely to break or flatten soon
  - Reduce position size (higher risk of reversal)
  
  MODERATE (20-45°):
  - Healthy, sustainable trend
  - Most reliable channels
  - Best for channel trading strategies
  
  SHALLOW (<20°):
  - Weak trend, nearly horizontal
  - Consider horizontal channel strategy instead
  
  Combining channels with other tools:
  
  Channel + Fibonacci:
  - Draw Fib retracement from channel bottom to top
  - 50% or 61.8% Fib often aligns with middle of channel
  - Strong confluence for mid-channel bounces
  
  Channel + Moving Averages:
  - 20 or 50 EMA often runs through channel middle
  - MA acts as dynamic midline support/resistance
  - Bounces off MA within channel = high-probability entries
  
  Channel + RSI:
  - RSI overbought (>70) at upper channel = strong sell signal
  - RSI oversold (<30) at lower channel = strong buy signal
  - Divergence at channel boundary = reversal warning
  
  How ForexGPT uses channels:
  1. Automatic channel detection on every timeframe
  2. Alerts when price approaches channel boundaries
  3. Breakout signals when channel breaks with confirmation
  4. Position sizing adjusted by channel width (wider = more volatile = smaller size)
  
  Best practices:
  - Use higher timeframes (4H, Daily) for more reliable channels
  - Redraw channels as new swings form
  - Don't force channels (if doesn't fit, market not channeling)
  - Combine with volume (if available) - breakouts need volume
  - Multiple timeframe channels (Daily channel + 4H sub-channels)
  
  Keyboard shortcuts:
  - C: Select channel tool
  - Alt+C: Equidistant channel (auto-parallel)
  - Delete: Remove selected channel
  ```

##### Rectangle Tool
- **What it is**: Draw rectangular boxes to highlight consolidation zones or price ranges
- **What it serves**: Mark support/resistance zones, flag patterns, accumulation areas
- **Tooltip**:
  ```
  Rectangle Tool (Consolidation Box)
  
  What it is:
  A rectangular box drawn around a period of consolidation or price range, 
  highlighting a zone where price is oscillating between defined support and 
  resistance levels without a clear trend.
  
  What it serves:
  - Consolidation zone identification
  - Range trading setup (buy at bottom of box, sell at top)
  - Breakout preparation (box breaks often lead to strong moves)
  - Flag/pennant pattern recognition
  - Accumulation/distribution zone marking
  
  How to draw:
  1. Select Rectangle tool
  2. Click at top-left corner (resistance + leftmost point of range)
  3. Drag to bottom-right corner (support + rightmost point of range)
  4. Rectangle drawn automatically
  
  Use cases:
  
  CONSOLIDATION BOXES (sideways range):
  What it is:
  - Price trading sideways between horizontal S/R levels
  - Market in equilibrium, buyers and sellers balanced
  - "Coiling" before next directional move
  
  Characteristics:
  - Horizontal top and bottom boundaries
  - Multiple touches on both boundaries
  - Duration: Several hours to several days
  - Typically follows strong trend (pause before continuation or reversal)
  
  Trading strategies:
  
  Range Trading (inside the box):
  1. BUY at bottom of rectangle (support):
     - Entry: Price touches lower boundary
     - Confirmation: Bullish reversal pattern (hammer, engulfing)
     - Stop loss: Just below rectangle (20-30 pips)
     - Target: Top of rectangle (resistance)
     - Scale out: Take 50% profit at middle, rest at top
  
  2. SELL at top of rectangle (resistance):
     - Entry: Price touches upper boundary
     - Confirmation: Bearish reversal pattern (shooting star, engulfing)
     - Stop loss: Just above rectangle (20-30 pips)
     - Target: Bottom of rectangle (support)
  
  When to avoid range trading:
  - Rectangle too narrow (<50 pips) - not enough profit after costs
  - Few touches on boundaries (<3 each side) - not confirmed range
  - High volatility (VIX >30) - risk of sudden breakout
  
  Breakout Trading (box break):
  1. Monitor for compression (price oscillations getting tighter)
  2. Watch for volume buildup (if available) - precedes breakouts
  3. Enter when price CLOSES outside rectangle (up or down)
     - Upward break: BUY
     - Downward break: SELL
  4. Confirmation: Candle body closes beyond box (not just wick)
  5. Stop loss: Re-entry into box (if back inside, breakout failed)
  6. Target: Measured move (box height projected in breakout direction)
  
  Example:
  - Rectangle: 1.0850 (bottom) to 1.0950 (top) = 100 pips height
  - Breakout: Price closes at 1.0960 (above top)
  - Target: 1.0950 + 100 pips = 1.1050
  
  Rectangle height and implications:
  
  VERY NARROW (<30 pips):
  - Tight consolidation, volatility compression
  - Imminent explosive breakout likely
  - Don't range trade (risk/reward poor)
  - Prepare breakout orders in BOTH directions
  
  NARROW (30-60 pips):
  - Moderate consolidation
  - Can range trade with tight stops
  - Breakout target = 60-120 pips (1:1 to 2:1 box height)
  
  MEDIUM (60-120 pips):
  - Ideal for range trading
  - Multiple scalp opportunities
  - Breakout target = 120-240 pips
  
  WIDE (120-200 pips):
  - Large range, swing trading ideal
  - Can break into sub-ranges (mini boxes inside)
  - Breakout target = 200-400 pips (big move expected)
  
  VERY WIDE (>200 pips):
  - Major consolidation or slow trend
  - Consider weekly timeframe pattern
  - Breakout = major trend change (months-long move)
  
  Rectangle duration:
  
  SHORT (<1 day):
  - Intraday consolidation
  - Typical flag or pennant pattern
  - Breakout usually continues prior trend
  
  MEDIUM (1-7 days):
  - Weekly consolidation
  - Reaccumulation or redistribution
  - Breakout can go either direction (50/50)
  
  LONG (>7 days):
  - Major consolidation
  - Market indecision or awaiting catalyst
  - Breakout direction determines new long-term trend
  
  Specific rectangle patterns:
  
  FLAG PATTERN (consolidation after strong move):
  What it is:
  - Small rectangle (30-60 pips) tilted slightly against prior trend
  - Follows strong directional move (flagpole)
  - Brief pause before trend continuation
  
  Example (bullish flag):
  - Strong rally (flagpole): 1.0700 → 1.0900 (200 pips)
  - Consolidation (flag): 1.0850-1.0900 rectangle (50 pips), slight downward tilt
  - Breakout: Above 1.0900 → Target 1.1100 (flagpole height projected)
  
  Trading:
  - Bias: CONTINUATION of prior trend (flag = pause, not reversal)
  - Entry: Breakout in trend direction (upward break for bullish flag)
  - Target: Prior move distance added to breakout point
  
  PENNANT PATTERN (triangle inside rectangle):
  - Similar to flag but triangle shape (consolidating range narrows)
  - Even stronger continuation signal
  - Breakout typically explosive
  
  ACCUMULATION ZONE (smart money buying):
  What it is:
  - Long rectangle at bottom of downtrend
  - Institutions quietly buying without pushing price up
  - Public still bearish, smart money accumulating
  
  Identification:
  - Downtrend stops at rectangle
  - Price oscillates inside box for weeks
  - Volume steady (not declining like typical bottom)
  - Small wicks on both sides (tight control)
  
  Trading:
  - Wait for upward breakout (accumulation complete, markup begins)
  - Enter on breakout or pullback to broken resistance (now support)
  - Target: Major resistance levels or measured move
  
  DISTRIBUTION ZONE (smart money selling):
  - Long rectangle at top of uptrend
  - Institutions quietly selling without crashing price
  - Public still bullish, smart money distributing
  
  Identification:
  - Uptrend stops at rectangle
  - Price oscillates inside box for weeks
  - Volume erratic (churning)
  - Large wicks on both sides (fighting)
  
  Trading:
  - Wait for downward breakout (distribution complete, markdown begins)
  - Enter short on breakout or pullback to broken support (now resistance)
  - Target: Major support levels or measured move
  
  Rectangle false breakouts:
  
  STOP HUNT (fakeout):
  - Price briefly pokes outside rectangle
  - Triggers breakout traders' stops
  - Immediately reverses back inside
  
  How to avoid:
  - Wait for CLOSE outside rectangle (not just wick)
  - Require 2 consecutive closes outside for strong confirmation
  - Use wider stops (beyond typical stop hunt zones)
  
  EXHAUSTION:
  - Breakout occurs but lacks follow-through
  - Price returns inside rectangle within 1-2 candles
  - Sign: Weak breakout, no momentum
  
  How to avoid:
  - Wait for retest (price pulls back to box, holds, then resumes breakout)
  - Enter on retest, not initial break (better risk/reward)
  
  Combining rectangles with other tools:
  
  Rectangle + Volume:
  - Low volume inside rectangle = calm accumulation/distribution
  - Volume spike on breakout = genuine move (not fake)
  - No volume on breakout = suspect (likely fails)
  
  Rectangle + Fibonacci:
  - Draw Fib from bottom to top of rectangle
  - 50% level often acts as midpoint support/resistance
  - Price oscillating around 50% Fib = balanced
  
  Rectangle + Moving Averages:
  - MAs (20, 50, 200 EMA) often run through middle of rectangle
  - Breakout confirmed when price AND MA exit rectangle
  
  How ForexGPT uses rectangles:
  1. Automatic consolidation detection (draws boxes for you)
  2. Breakout alerts when price exits rectangle
  3. Measured move targets calculated automatically
  4. Range trading signals (buy at bottom, sell at top)
  
  Best practices:
  - Use higher timeframes (4H, Daily) for reliable rectangles
  - Don't force boxes (if price not ranging, don't draw)
  - Wait for at least 3 touches on each boundary before trading
  - Label your rectangles (accumulation, distribution, flag, etc.)
  - Delete old rectangles to avoid chart clutter
  
  Keyboard shortcuts:
  - R: Select rectangle tool
  - Shift+R: Ellipse tool (circular consolidation)
  - Delete: Remove selected rectangle
  - Ctrl+D: Duplicate rectangle to another chart
  ```

### 1.2.2 Main Chart Display

#### Chart Type Selection
- **What it is**: Choose between candlestick, bar, line, or Heikin-Ashi charts
- **What it serves**: Different visualizations for different analysis styles
- **Options**:
  - **Candlestick**: Shows OHLC with colored bodies (default)
  - **Bar Chart**: OHLC as vertical bars
  - **Line Chart**: Close prices connected
  - **Heikin-Ashi**: Smoothed candlesticks for trend clarity
- **Tooltip**:
  ```
  Chart Type Selection
  
  What it is:
  Choose how price data is visually represented on the chart. Different chart 
  types emphasize different aspects of price action.
  
  Options:
  
  CANDLESTICK (Default, Most Popular):
  What it is:
  - Japanese candlestick chart
  - Each candle shows 4 data points: Open, High, Low, Close (OHLC)
  - Body (rectangle): Open to Close distance
  - Wicks (lines): High and Low extremes
  - Color: Green (close > open), Red (close < open)
  
  What it serves:
  - Most information-rich visualization
  - Shows buyer/seller battle within each period
  - Enables candlestick pattern recognition (doji, hammer, engulfing, etc.)
  - Industry standard for technical analysis
  
  When to use:
  - ALL trading styles (scalping to position trading)
  - Pattern recognition (requires candlesticks)
  - Precise entry/exit timing
  - Default choice for 95% of traders
  
  Candlestick components:
  - LONG GREEN BODY: Strong buying, bulls in control
  - SHORT GREEN BODY: Weak buying, bulls barely winning
  - LONG RED BODY: Strong selling, bears in control
  - SHORT RED BODY: Weak selling, bears barely winning
  - LONG UPPER WICK: Rejection of highs (sellers stepped in)
  - LONG LOWER WICK: Rejection of lows (buyers stepped in)
  - NO WICKS (MARUBOZU): Extreme conviction, no hesitation
  - SMALL BODY + LONG WICKS (DOJI): Indecision, equilibrium
  
  BAR CHART (OHLC Bars):
  What it is:
  - Vertical line showing High to Low
  - Left tick: Opening price
  - Right tick: Closing price
  - No colored bodies, just lines
  
  What it serves:
  - Clean, minimalist visualization
  - Same data as candlesticks but less visual noise
  - Preferred by some professional traders
  
  When to use:
  - When candlesticks feel too cluttered
  - Focusing on price levels rather than patterns
  - Old-school technical analysis
  - Western trading tradition (pre-candlestick popularity)
  
  Advantages:
  - Less visually distracting than candlesticks
  - Easier to see price levels through clutter
  - Lighter on screen rendering (faster)
  
  Disadvantages:
  - Harder to spot candlestick patterns
  - Less intuitive for beginners
  - Can't see "body" strength as easily
  
  LINE CHART (Close-Only):
  What it is:
  - Connects closing prices of each period
  - Smoothest, simplest visualization
  - Only shows close (ignores open, high, low)
  
  What it serves:
  - Trend identification (removes intrabar noise)
  - Smoothed price action
  - Beginner-friendly (least intimidating)
  
  When to use:
  - Identifying overall trend direction
  - When intrabar volatility is noise (not signal)
  - Presenting to non-traders (simplest to understand)
  - Smoothing erratic markets
  
  Advantages:
  - Clearest trend visualization
  - No visual clutter whatsoever
  - Easy to draw trendlines and channels
  - Fastest chart rendering
  
  Disadvantages:
  - LOSES DATA (no open, high, low info)
  - Can't see intrabar volatility (dangerous for stops)
  - No pattern recognition possible
  - Hides important price rejections (wicks)
  
  NOT recommended for:
  - Active trading (need OHLC data)
  - Stop loss placement (can't see real volatility)
  - Entry timing (need to see intrabar action)
  
  HEIKIN-ASHI (Japanese "Average Bar"):
  What it is:
  - Modified candlesticks using averaged prices
  - Formula smooths out noise, emphasizes trends
  - Calculation:
    * HA Close = (Open + High + Low + Close) / 4
    * HA Open = (Previous HA Open + Previous HA Close) / 2
    * HA High = Max(High, HA Open, HA Close)
    * HA Low = Min(Low, HA Open, HA Close)
  
  What it serves:
  - Trend identification (filters noise)
  - Smoother price action than regular candlesticks
  - Reduces false signals in choppy markets
  - Shows trend strength via candle bodies
  
  Visual characteristics:
  - Long green candles with small/no lower wicks: STRONG UPTREND
  - Long red candles with small/no upper wicks: STRONG DOWNTREND
  - Candles with long wicks on both sides: TREND WEAKENING
  - Alternating colors (green-red-green-red): SIDEWAYS/CHOPPY
  
  When to use:
  - Trend-following strategies (filters whipsaws)
  - Volatile/choppy markets (smooths out noise)
  - Staying in winning trades longer (less false exits)
  - Beginners who get shaken out by normal volatility
  
  Advantages:
  - Smoother than regular candlesticks
  - Trends appear clearer
  - Fewer false signals in choppy conditions
  - Easier to hold winning trades (less noise)
  
  Disadvantages:
  - NOT REAL PRICES (averaged/calculated)
  - Delayed signals (smoothing = lag)
  - Can't use for precise entry/exit (not actual prices)
  - Hides important wicks (price rejections)
  - NOT suitable for market orders (use regular candles for entry)
  
  WARNING about Heikin-Ashi:
  - Use for ANALYSIS only (identifying trends)
  - Switch to regular candlesticks for ENTRY/EXIT
  - Do NOT place stops based on HA candles (not real prices)
  - Do NOT backtest on HA (results will be misleading)
  
  How to use Heikin-Ashi properly:
  1. Use HA on higher timeframe (Daily, 4H) to see overall trend
  2. Switch to regular candlesticks on lower TF (1H, 15m) for entries
  3. Trade in direction of HA trend (HA Daily green = only longs)
  4. Exit when HA candles change color (trend reversing)
  
  Which chart type should YOU use?
  
  BEGINNER:
  - Start with CANDLESTICK (industry standard)
  - Learn to read candlestick patterns
  - Switch to line chart temporarily if feeling overwhelmed
  
  INTERMEDIATE:
  - CANDLESTICK for most trading
  - LINE chart for quick trend checks (1-minute glance)
  - Experiment with HEIKIN-ASHI for trend confirmation
  
  ADVANCED:
  - CANDLESTICK primary (95% of time)
  - HEIKIN-ASHI on secondary chart for trend filter
  - BAR chart if personal preference (some pros prefer it)
  - LINE chart rarely (only for presentations)
  
  How ForexGPT uses chart types:
  1. Default: Candlestick (optimized for pattern recognition AI)
  2. Heikin-Ashi mode available for trend filtering
  3. Chart type saved per symbol/timeframe (your preference remembered)
  4. Indicators work on all chart types (calculations same)
  
  Keyboard shortcuts:
  - Alt+1: Candlestick chart
  - Alt+2: Bar chart
  - Alt+3: Line chart
  - Alt+4: Heikin-Ashi chart
  ```

---

#### Chart Timeframe Selector
- **What it is**: Dropdown to select candlestick timeframe
- **Options**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- **What it serves**: Changes granularity of price action displayed
- **Tooltip**:
  ```
  Chart Timeframe Selector
  
  What it is:
  Select the time period represented by each candlestick on the chart.
  1m = 1-minute candles, 1d = daily candles, etc.
  
  What it serves:
  - Multi-timeframe analysis (view same market at different scales)
  - Strategy alignment (scalping = 1m/5m, swing = 4h/1d)
  - Zoom in/out without losing context
  
  Timeframe selection guide:
  
  1m (1-minute candles):
  - Each candle = 1 minute of price action
  - ~1400 candles per trading day (24h FX market)
  - Data coverage: Typically 7-30 days stored locally
  
  Use for:
  - Scalping (5-15 minute trades)
  - Precise entry timing within day trading setups
  - High-frequency pattern recognition
  - Tick-level noise visible (use with caution)
  
  Pros:
  - Maximum granularity, see every small move
  - Perfect for intraday precision entries
  - Catch micro-patterns (1-5 minute setups)
  
  Cons:
  - High noise-to-signal ratio
  - Requires constant monitoring
  - Many false breakouts and whipsaws
  - Spread costs significant on small moves
  
  5m (5-minute candles):
  - Each candle = 5 minutes
  - ~280 candles per day
  - Data coverage: 30-90 days typical
  
  Use for:
  - Scalping (15-60 minute trades)
  - Day trading with slightly less noise than 1m
  - Intraday trend confirmation
  
  Pros:
  - Good balance of granularity vs noise
  - Still fast enough for scalping
  - Filters out some 1m noise
  
  Cons:
  - Still noisy, requires attention
  - Many intraday reversals
  - Can miss very fast moves (1-3 minute spikes)
  
  15m (15-minute candles):
  - Each candle = 15 minutes
  - ~96 candles per day
  - Data coverage: 90 days - 1 year
  
  Use for:
  - Day trading (1-4 hour trades)
  - Short-term swing trading
  - Key timeframe for day trader decisions
  
  Pros:
  - Significantly less noise than 1m/5m
  - Patterns more reliable (3+ candle patterns)
  - Good for trend identification within day
  
  Cons:
  - Can miss quick intraday reversals
  - Still requires intraday monitoring
  - Overnight gaps if trading stocks (less issue in 24h FX)
  
  30m (30-minute candles):
  - Each candle = 30 minutes
  - ~48 candles per day
  - Data coverage: 6 months - 2 years
  
  Use for:
  - Day trading (2-8 hour trades)
  - Swing trading (overnight to 3 days)
  - Institutional timeframe (banks watch this)
  
  Pros:
  - Clean price action, low noise
  - Reliable patterns and trends
  - Good for part-time traders (check every 30min)
  
  Cons:
  - Slower entries/exits vs 15m
  - May miss intraday opportunities
  - Requires patience (setups take hours)
  
  1h (1-hour candles):
  - Each candle = 1 hour
  - 24 candles per day
  - Data coverage: 1-5 years typical
  
  Use for:
  - Swing trading (1-7 day holds)
  - Position trading (weeks to months)
  - PRIMARY timeframe for most retail traders
  
  Pros:
  - Very reliable patterns (proven over hours)
  - Low noise, strong trend signals
  - Check once per hour (lifestyle-friendly)
  - Widely followed (institutional support/resistance)
  
  Cons:
  - Slower reaction to news events
  - Larger stop losses required (more pips)
  - Fewer trading opportunities (1-2 setups per week)
  
  4h (4-hour candles):
  - Each candle = 4 hours
  - 6 candles per day
  - Data coverage: 3-10 years
  
  Use for:
  - Swing trading (3-30 day holds)
  - Position trading (months)
  - Long-term trend identification
  
  Pros:
  - Extremely reliable (patterns span full trading sessions)
  - Major support/resistance levels highly respected
  - Check 2-3 times per day (very lifestyle-friendly)
  - Low noise, high signal
  
  Cons:
  - Very slow to react (4 hours per candle update)
  - Wide stop losses (50-200 pips)
  - Very few setups (1-2 per month)
  - Requires large account (wider stops = more capital)
  
  1d (Daily candles):
  - Each candle = 1 full trading day
  - 1 candle per day (obvious)
  - Data coverage: 5-50 years available
  
  Use for:
  - Position trading (weeks to years)
  - Long-term trend analysis
  - Weekly swing trades
  - End-of-day trading (EOD strategy)
  
  Pros:
  - Maximum reliability (full day of price action)
  - Historical data rich (decades available)
  - Check ONCE per day (ultimate lifestyle freedom)
  - Major patterns very significant (months in making)
  
  Cons:
  - Extremely slow (1 candle per day)
  - Massive stop losses (100-500 pips)
  - Very few setups (1-2 per quarter)
  - Requires patience and discipline
  
  1w (Weekly candles):
  - Each candle = 1 full week (5-7 days)
  - ~52 candles per year
  - Data coverage: Decades
  
  Use for:
  - Long-term position trading (months to years)
  - Macro trend analysis
  - Pension fund / institutional level
  
  Pros:
  - Macro trends crystal clear
  - Noise completely filtered out
  - Check once per week (weekend review)
  
  Cons:
  - Glacially slow (1 candle per week!)
  - Huge stop losses (500-2000 pips)
  - Setups extremely rare (1-2 per year)
  - Not practical for most retail traders
  
  1M (Monthly candles):
  - Each candle = 1 full month
  - 12 candles per year
  - Data coverage: Decades to centuries
  
  Use for:
  - Multi-year position trading
  - Economic cycle analysis
  - Central bank and sovereign fund level
  
  Pros:
  - Secular trends visible (decades-long moves)
  - Ultimate big picture
  
  Cons:
  - Completely impractical for trading (1 candle/month!)
  - Only useful for macro analysis
  - Never use for entries/exits
  
  Multi-timeframe analysis (how to use multiple together):
  
  TOP-DOWN APPROACH (recommended):
  1. Start on Daily (1d): Identify major trend and key S/R levels
  2. Drop to 4H: Confirm trend, find swing structure
  3. Drop to 1H: Identify entry setups aligned with 4H/Daily trend
  4. Drop to 15m: Fine-tune exact entry point
  
  Example:
  - Daily: Strong uptrend, approaching resistance at 1.2000
  - 4H: Uptrend intact, small pullback forming
  - 1H: Pullback to 38.2% Fib, bullish engulfing forming
  - 15m: Entry long on break of 15m resistance with tight stop
  
  COMMON MISTAKES:
  - Trading lower TF against higher TF trend (fighting the trend)
  - Using only 1 timeframe (missing big picture)
  - Too many timeframes (analysis paralysis)
  
  BEST PRACTICE:
  - Primary timeframe: Where you make entry decision (1H typical)
  - Higher timeframe: Trend filter (4H/Daily confirms direction)
  - Lower timeframe: Entry precision (15m for exact entry timing)
  
  How ForexGPT uses timeframes:
  1. Indicators calculated on ALL timeframes simultaneously
  2. Multi-timeframe ensemble model (combines 1m, 5m, 15m, 1H, 4H, Daily predictions)
  3. Regime detection analyzes 4H and Daily for macro trend
  4. Signals generated on 1H, refined on 15m for entry
  
  Keyboard shortcuts:
  - Ctrl+Up: Next higher timeframe
  - Ctrl+Down: Next lower timeframe
  - Ctrl+1 through Ctrl+9: Jump to specific timeframes
  ```

#### Symbol Selector
- **What it is**: Dropdown to choose currency pair or instrument
- **Options**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY, etc.
- **What it serves**: Switch between different markets/instruments
- **Tooltip**:
  ```
  Symbol Selector
  
  What it is:
  Choose which currency pair (or financial instrument) to display and analyze.
  Symbols formatted as BASE/QUOTE (e.g., EUR/USD = Euro vs US Dollar).
  
  What it serves:
  - Trade different markets
  - Diversification across pairs
  - Correlation analysis (watch related pairs)
  - Opportunity scanning (which pair is moving?)
  
  Major Pairs (most liquid, tightest spreads):
  
  EUR/USD (Euro vs US Dollar):
  - Most traded pair globally (~30% of all FX volume)
  - Tightest spreads (0.0-1.0 pips typically)
  - Best for: Beginners, scalping, technical analysis
  - Correlation: Moves opposite USD/CHF (>90% inverse correlation)
  - Characteristics: Smooth trends, respects technical levels
  - Trading hours: Best during London/NY overlap (13:00-17:00 GMT)
  - Volatility: Medium (50-100 pips/day average)
  
  GBP/USD (British Pound vs US Dollar):
  - "The Cable" (historical nickname)
  - 2nd most traded major pair
  - Spreads: 1-3 pips typical
  - Best for: Experienced traders (higher volatility)
  - Characteristics: Sharp moves, frequent reversals, news-sensitive
  - Trading hours: London session (07:00-16:00 GMT)
  - Volatility: HIGH (100-200 pips/day), can spike 50+ pips in minutes
  - Caution: Brexit-related news causes extreme volatility
  
  USD/JPY (US Dollar vs Japanese Yen):
  - Major safe-haven pair
  - Spreads: 1-2 pips
  - Best for: Range trading, carry trades
  - Characteristics: Smooth trends, strong support at 100/105/110/115
  - Trading hours: Tokyo open (00:00-09:00 GMT), NY session
  - Volatility: Medium-Low (60-80 pips/day)
  - Unique: Intervention risk (Bank of Japan manipulates this pair)
  
  AUD/USD (Australian Dollar vs US Dollar):
  - "Aussie" - commodity currency
  - Spreads: 1-2 pips
  - Best for: Swing trading, commodity traders
  - Characteristics: Trends well, correlated with gold/copper prices
  - Trading hours: Sydney/Asian session (22:00-08:00 GMT)
  - Volatility: Medium (70-100 pips/day)
  - Driver: China economy (Australia's #1 trade partner)
  
  USD/CAD (US Dollar vs Canadian Dollar):
  - "Loonie" - oil currency
  - Spreads: 1.5-2.5 pips
  - Best for: Oil traders (strong inverse correlation with crude)
  - Characteristics: Smooth trends, technical levels respected
  - Trading hours: NY session (13:00-22:00 GMT)
  - Volatility: Medium (60-90 pips/day)
  - Driver: Oil prices (Canada is major oil exporter)
  
  USD/CHF (US Dollar vs Swiss Franc):
  - "Swissie" - safe haven
  - Spreads: 1.5-3 pips
  - Best for: Conservative traders, risk-off plays
  - Characteristics: Inverse of EUR/USD (>90% negative correlation)
  - Trading hours: European session (07:00-17:00 GMT)
  - Volatility: Low (50-80 pips/day)
  - Unique: Swiss National Bank (SNB) heavily intervenes
  
  NZD/USD (New Zealand Dollar vs US Dollar):
  - "Kiwi" - commodity currency
  - Spreads: 2-4 pips
  - Best for: Swing trading
  - Characteristics: Similar to AUD/USD but more volatile
  - Trading hours: Sydney/Asian session
  - Volatility: Medium-High (80-120 pips/day)
  
  Cross Pairs (no USD involved, wider spreads):
  
  EUR/GBP (Euro vs British Pound):
  - European cross
  - Spreads: 1.5-3 pips
  - Best for: Range trading (historically tight range)
  - Characteristics: Mean-reverting, rarely trends
  - Range: 0.8000-0.9500 historical (breaks are rare)
  - Volatility: Low (30-60 pips/day)
  
  EUR/JPY (Euro vs Japanese Yen):
  - Risk sentiment barometer
  - Spreads: 2-3 pips
  - Best for: Trend trading, risk-on/risk-off plays
  - Characteristics: Strong trends (up in risk-on, down in risk-off)
  - Volatility: HIGH (100-150 pips/day)
  - Correlation: Stocks (risk-on = EUR/JPY up)
  
  GBP/JPY (British Pound vs Japanese Yen):
  - "The Beast" or "The Widow Maker"
  - Spreads: 3-5 pips
  - Best for: EXPERIENCED ONLY (extreme volatility)
  - Characteristics: Huge swings, 200+ pip days common
  - Volatility: EXTREME (150-300 pips/day, can be 500+ during news)
  - Caution: Can wipe accounts with poor risk management
  
  Exotic Pairs (emerging markets, very wide spreads):
  
  EUR/TRY, USD/MXN, USD/ZAR, etc.:
  - Spreads: 10-100+ pips (!!!)
  - Best for: Advanced traders only
  - Characteristics: Huge volatility, poor liquidity
  - Caution: Overnight gaps, flash crashes, broker manipulation risk
  - NOT recommended for most traders
  
  How to choose a pair to trade:
  
  BEGINNER TRADERS:
  - Start with EUR/USD only (master 1 pair first)
  - Low spreads, high liquidity, predictable
  - Learn patterns on this pair for 6-12 months
  - Then expand to 1-2 more majors
  
  INTERMEDIATE TRADERS:
  - 2-4 major pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD)
  - Diversification without overload
  - Watch correlations (don't trade EUR/USD and USD/CHF simultaneously)
  
  ADVANCED TRADERS:
  - 5-8 pairs including some crosses
  - Correlation-aware portfolio management
  - Sector rotation (majors vs crosses vs commodities)
  
  Pair correlations (important for risk management):
  
  POSITIVE CORRELATION (move together):
  - EUR/USD ↔ GBP/USD (~80%) - both USD pairs
  - AUD/USD ↔ NZD/USD (~85%) - both commodity currencies
  - EUR/USD ↔ AUD/USD (~70%) - both risk-on pairs
  
  NEGATIVE CORRELATION (move opposite):
  - EUR/USD ↔ USD/CHF (~-95%) - nearly perfect inverse
  - EUR/USD ↔ USD/JPY (~-60%) - moderate inverse
  
  Risk management with correlations:
  - DON'T trade EUR/USD long + USD/CHF long (cancels out, double spread cost)
  - DO trade EUR/USD long + GBP/USD long IF conviction high (magnified exposure)
  - DO trade EUR/USD long + USD/JPY short (uncorrelated diversification)
  
  Volatility comparison (average daily range):
  - Low (50-80 pips): USD/CHF, EUR/GBP
  - Medium (80-120 pips): EUR/USD, USD/JPY, AUD/USD
  - High (120-180 pips): GBP/USD, EUR/JPY, NZD/USD
  - Extreme (180-300+ pips): GBP/JPY
  
  Trading session characteristics:
  
  ASIAN SESSION (22:00-08:00 GMT):
  - Best pairs: USD/JPY, AUD/USD, NZD/USD
  - Characteristics: Low volatility, ranging
  - Strategy: Range trading, fade extremes
  
  EUROPEAN SESSION (07:00-16:00 GMT):
  - Best pairs: EUR/USD, GBP/USD, EUR/GBP
  - Characteristics: Trending, high volume
  - Strategy: Breakout and trend following
  
  NEW YORK SESSION (13:00-22:00 GMT):
  - Best pairs: All USD pairs (especially EUR/USD, GBP/USD)
  - Characteristics: High volatility, reversals
  - Strategy: News trading, momentum
  
  OVERLAP (London+NY, 13:00-17:00 GMT):
  - Highest volume and volatility of the day
  - All majors active
  - Tightest spreads
  - Best trading window for most strategies
  
  How ForexGPT handles symbols:
  1. Each symbol has independent trained models
  2. Correlation matrix updated real-time
  3. Portfolio risk calculated across all open positions
  4. Alerts when correlation >80% (warn user of concentrated risk)
  
  Symbol data requirements:
  - Minimum: 7 days of 1m data (for basic analysis)
  - Recommended: 90 days of 1m data (for reliable training)
  - Optimal: 1+ year of 1m data (for walk-forward validation)
  
  Keyboard shortcuts:
  - Ctrl+S: Open symbol selector
  - Type first letters: Quick search (e.g., "EU" → EUR/USD)
  - Ctrl+→/←: Cycle through symbols in watchlist
  ```

---

## 2. Trading Intelligence Tab

**Location**: Main Window → Trading Intelligence Tab (Level 1)

**Description**: Container tab with nested Portfolio and Signals sub-tabs for trading analytics and opportunity scanning.

### 2.1 Portfolio Tab

**Location**: Trading Intelligence → Portfolio (Level 2)

**Description**: Portfolio optimization using Modern Portfolio Theory (MPT), efficient frontier calculation, and multi-symbol allocation.

---

## 3. Generative Forecast Tab

**Location**: Main Window → Generative Forecast Tab (Level 1)

**Description**: Container tab for forecast configuration, model training, and backtesting.

### 3.1 Forecast Settings Tab

**Location**: Generative Forecast → Forecast Settings (Level 2)

**Description**: Configure diffusion model parameters for probabilistic forecasting.

### 3.2 Training Tab

**Location**: Generative Forecast → Training (Level 2)

**Description**: Comprehensive model training interface with feature selection, hyperparameter tuning, and optimization.

---

**STATUS**: Documentation continues beyond token limit. This is section 1 of document.  
**Next**: Continue with complete Training Tab parameters (100+ parameters to document).

---

