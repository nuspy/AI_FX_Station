# ForexGPT - Missing Features Implementation Specifications
**Document Version:** 1.0  
**Date:** October 7, 2025  
**Status:** Draft - Awaiting Additional Specifications

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [Database Schema Extensions](#database-schema-extensions)
4. [Data Acquisition Layer](#data-acquisition-layer)
5. [Processing and Analysis Layer](#processing-and-analysis-layer)
6. [User Interface Components](#user-interface-components)
7. [Integration Points](#integration-points)
8. [Configuration and Settings](#configuration-and-settings)
9. [Version Control Strategy](#version-control-strategy)

---

## 1. Overview

### 1.1 Purpose
This document specifies the implementation requirements for completing the Volume and Depth of Market (DOM) functionality within the ForexGPT trading system. The current implementation contains partial infrastructure that requires completion, integration, and visualization.

### 1.2 Scope
The implementation encompasses:
- Completion of Depth of Market (DOM) data acquisition and storage
- Enhancement of volume analysis visualization
- Creation of dedicated UI components for order book and market depth
- Integration of existing analysis engines (VSA, Volume Profile, Order Flow) with the GUI
- Establishment of proper data flow between providers, storage, processing, and visualization layers

### 1.3 Current State Analysis
**Functional Components:**
- Volume data acquisition from providers (Tiingo, cTrader)
- Volume storage in database (market_data_ticks table)
- Volume Profile calculation engine (POC, VA, HVN, LVN)
- VSA (Volume Spread Analysis) pattern detection
- Order Flow analysis framework

**Incomplete Components:**
- DOM data streaming and real-time updates
- DOM database schema and persistence
- DOM aggregation service activation
- Order book visualization widget
- DOM heatmap/depth chart visualization
- Volume analysis indicators on main charts
- VSA signal overlays

---

## 2. Architecture Principles

### 2.1 Design Philosophy
All implementations must adhere to the existing architectural patterns within ForexGPT:
- **Separation of Concerns**: Data acquisition, processing, storage, and presentation remain distinct layers
- **Provider Abstraction**: All market data sources operate through the provider interface
- **Service-Oriented**: Background services handle continuous data processing
- **Event-Driven**: Components communicate via event bus where appropriate
- **Configuration-Driven**: User-adjustable parameters exposed through settings

### 2.2 Integration Requirements
Every new component must:
- Connect to existing workflows without breaking current functionality
- Respect existing naming conventions and code structure
- Utilize established utility functions and helpers
- Maintain backward compatibility with existing data
- Integrate with the current configuration system

### 2.3 Quality Standards
- No orphaned files or unused methods
- All database changes managed through proper migration mechanisms
- Dependencies declared explicitly in project configuration
- Each logical unit committed to version control with descriptive messages
- Error handling and logging at appropriate levels

---

## 3. Database Schema Extensions

### 3.1 Market Depth Table Creation

#### 3.1.1 Purpose
Store order book snapshots for analysis and historical reference. This table captures bid/ask levels with associated volumes at specific timestamps.

#### 3.1.2 Schema Requirements
The table must accommodate:
- Symbol identification
- Timestamp in milliseconds (UTC)
- Bid levels (price and volume pairs)
- Ask levels (price and volume pairs)
- Calculated metrics (mid price, spread, imbalance)
- Provider source identification

#### 3.1.3 Data Format Considerations
- Bid and ask levels stored as structured data (JSON or similar)
- Support for variable depth (5, 10, 20 levels)
- Efficient querying by symbol and time range
- Unique constraint preventing duplicate snapshots

#### 3.1.4 Integration with Existing Schema
- Leverage SQLAlchemy ORM for table definition
- Create appropriate indexes for query performance
- Establish foreign key relationships where applicable
- Maintain consistency with existing table naming conventions

### 3.2 Volume Analysis Cache Tables (Optional)

#### 3.2.1 Purpose
Pre-calculated volume profile metrics for performance optimization.

#### 3.2.2 Cached Metrics
- Point of Control (POC) values per window
- Value Area High/Low boundaries
- High Volume Nodes (HVN) locations
- Low Volume Nodes (LVN) locations
- Calculation window metadata (start, end, timeframe)

#### 3.2.3 Invalidation Strategy
Cache entries must invalidate when:
- New candles arrive in the calculation window
- Window parameters change
- Data corrections occur

### 3.3 Migration Strategy

#### 3.3.1 Schema Evolution
- Utilize Alembic or similar migration framework
- Create reversible migration scripts
- Test migrations on development database before production
- Document migration dependencies and prerequisites

#### 3.3.2 Data Migration
If existing data requires transformation:
- Define transformation logic clearly
- Implement idempotent migration scripts
- Provide rollback procedures
- Validate data integrity post-migration

---

## 4. Data Acquisition Layer

### 4.1 DOM Streaming Implementation

#### 4.1.1 WebSocket Connection Management
**Objective**: Establish persistent WebSocket connections to providers supporting Level 2/3 market data.

**Requirements**:
- Connection lifecycle management (connect, disconnect, reconnect)
- Automatic reconnection with exponential backoff
- Heartbeat/keepalive mechanism
- Error detection and recovery
- Connection status monitoring

#### 4.1.2 Subscription Management
**Objective**: Subscribe to order book updates for user-selected symbols.

**Requirements**:
- Dynamic subscription addition/removal
- Subscription state tracking
- Handling subscription confirmations and errors
- Resource cleanup on unsubscribe

#### 4.1.3 Message Parsing and Normalization
**Objective**: Convert provider-specific DOM messages into standardized internal format.

**Requirements**:
- Parse bid/ask arrays from provider formats
- Normalize price levels and volumes
- Handle incremental vs full snapshot updates
- Timestamp normalization to UTC milliseconds
- Validation of data integrity

#### 4.1.4 Provider Integration Points
Each provider supporting DOM must implement:
- `_stream_market_depth_impl()` method
- DOM message type handling
- Provider-specific data transformations
- Rate limit compliance
- Error reporting to health monitoring

### 4.2 Historical DOM Data (If Available)

#### 4.2.1 Historical Snapshots
For providers offering historical order book data:
- Define retrieval interface
- Implement batched downloading for large ranges
- Store with appropriate temporal granularity
- Handle missing data gracefully

---

## 5. Processing and Analysis Layer

### 5.1 DOM Aggregation Service

#### 5.1.1 Service Activation
**Objective**: Transform the placeholder DOMAggreg atorService into a fully functional background service.

**Requirements**:
- Initialize service on application startup
- Process DOM snapshots from queue/stream
- Calculate derived metrics (mid price, spread, imbalance)
- Persist processed snapshots to database
- Emit events for UI updates

#### 5.1.2 Metrics Calculation
The service must calculate:
- **Mid Price**: Volume-weighted average of best bid/ask
- **Spread**: Difference between best ask and best bid
- **Order Book Imbalance**: Ratio of bid volume to ask volume
- **Liquidity Depth**: Total volume within N price levels
- **Weighted Mid Price**: Volume-weighted mid across top levels

#### 5.1.3 Real-Time Processing Pipeline
**Data Flow**:
1. DOM snapshot arrives from provider
2. Snapshot enters processing queue
3. Service validates and normalizes data
4. Metrics calculated from bid/ask arrays
5. Results persisted to database
6. Event emitted to subscribers (UI components)

#### 5.1.4 Performance Considerations
- Asynchronous processing to avoid blocking
- Configurable processing interval
- Memory-efficient handling of snapshot history
- Garbage collection of old snapshots

### 5.2 Volume Profile Integration

#### 5.2.1 Real-Time Calculation Trigger
**Objective**: Automatically recalculate volume profile when new candles arrive.

**Requirements**:
- Subscribe to candle update events
- Determine calculation window dynamically
- Trigger profile calculation for affected timeframes
- Update cached results
- Notify UI of changes

#### 5.2.2 Multi-Timeframe Support
Volume profile must calculate across:
- Intraday windows (last N hours)
- Daily windows (current session, previous session)
- Weekly windows (current week)
- Custom user-defined windows

### 5.3 VSA Signal Generation

#### 5.3.1 Signal Pipeline Integration
**Objective**: Connect VSA analyzer to the signal processing system.

**Requirements**:
- Process each completed candle through VSA analysis
- Generate VSA signals with classification and strength
- Store signals in signals table
- Associate signals with candles for charting
- Filter signals by minimum strength threshold

#### 5.3.2 Signal Types Coverage
Ensure detection and categorization of:
- Accumulation patterns
- Distribution patterns
- Buying climax
- Selling climax
- No demand
- No supply
- Upthrust
- Spring

### 5.4 Order Flow Signal Generation

#### 5.4.1 Integration with DOM Data
**Objective**: Utilize real-time DOM data for order flow analysis.

**Requirements**:
- Correlate DOM snapshots with price action
- Detect absorption patterns (large volume, small price movement)
- Identify exhaustion (declining volume in trend)
- Track large order detection
- Calculate spread anomaly z-scores

#### 5.4.2 Signal Output
Generated signals must include:
- Signal type classification
- Directional bias (bullish/bearish)
- Strength metric (0-1)
- Confidence metric (0-1)
- Entry/target/stop price suggestions
- Metadata for signal reasoning

---

## 6. User Interface Components

### 6.1 Chart Tab Integration

#### 6.1.1 Volume Indicators on Main Chart
**Objective**: Display volume analysis directly on price charts.

**Location**: Within existing chart_tab_ui subplot structure

**Visual Elements**:
- **Volume Bars**: Traditional volume histogram below price
  - Color-coded by candle direction (up/down)
  - Optional separation of buy/sell volume
  - Overlay of volume moving average

- **Volume Profile Overlay**: Horizontal volume distribution
  - POC line marking highest volume price
  - Value Area highlighting (70% volume zone)
  - HVN markers on high volume nodes
  - LVN markers on low volume nodes

- **VSA Signal Markers**: Visual indicators on candles
  - Icon/symbol per signal type
  - Color coding by bullish/bearish
  - Hover tooltip with signal details
  - Optional signal strength indication (size/opacity)

#### 6.1.2 Subpanel Configuration
**Objective**: Dedicate subplot 1 to volume analysis.

**Requirements**:
- Toggle visibility via settings
- Configurable subplot height
- Multiple visualization modes:
  - Standard volume histogram
  - Volume with moving average
  - Buy/sell volume split
  - Cumulative volume delta
  - Volume profile horizontal distribution

#### 6.1.3 Visualization Controls
User-accessible settings for:
- Volume display style selection
- Color scheme customization
- Volume Profile window length
- HVN/LVN prominence thresholds
- VSA signal visibility toggles
- Signal strength filter

### 6.2 Order Book Widget (Floating Panel)

#### 6.2.1 Widget Architecture
**Design Pattern**: Floating dockable panel, activatable from settings/view menu.

**Layout Structure**:
- Header: Symbol, last update timestamp, connection status
- Bid Section: Table of bid levels (price, volume, cumulative)
- Mid Section: Spread display, mid price, imbalance indicator
- Ask Section: Table of ask levels (price, volume, cumulative)
- Footer: Controls and settings

#### 6.2.2 Data Display Format
**Bid/Ask Tables**:
- Columns: Price | Volume | Cumulative Volume | % of Total
- Rows: Configurable depth (5, 10, 20, 50 levels)
- Formatting: Appropriate decimal places for symbol
- Highlighting: Best bid/ask emphasized
- Color coding: Volume intensity heatmap

**Dynamic Updates**:
- Real-time refresh as snapshots arrive
- Smooth transitions (not jarring jumps)
- Flash indicators for level changes
- Cumulative volume recalculation

#### 6.2.3 Visual Enhancements
- **Volume Bars**: Horizontal bars representing volume at each level
- **Imbalance Indicator**: Visual representation of bid/ask volume ratio
- **Price Ladder**: Optional price axis for quick reference
- **Spread Visualization**: Graphical representation of spread width

#### 6.2.4 Interaction Features
- Click price level to place order (if trading enabled)
- Hover for detailed level information
- Drag to reorder columns
- Right-click context menu for actions
- Export snapshot to CSV/JSON

#### 6.2.5 Widget Controls
Settings accessible within widget:
- Depth level selection (5/10/20/50)
- Display mode (compact/detailed)
- Color scheme
- Update rate throttling
- Auto-hide when symbol changes
- Pin to specific symbol

### 6.3 DOM Heatmap Visualization

#### 6.3.1 Heatmap Concept
**Objective**: Visualize order book depth evolution over time.

**Visual Representation**:
- X-axis: Time
- Y-axis: Price levels
- Color intensity: Volume magnitude at price/time
- Overlay: Current price line

#### 6.3.2 Implementation Approach
**Widget Type**: Embeddable plot or standalone window

**Data Source**: Historical DOM snapshots from database

**Rendering**:
- Use appropriate plotting library (matplotlib, pyqtgraph, plotly)
- Implement efficient rendering for large datasets
- Support zooming and panning
- Time range selector

#### 6.3.3 Interpretation Aids
- POC tracking line over time
- Imbalance zones highlighted
- Support/resistance level detection overlays
- Liquidity cluster identification

#### 6.3.4 Configuration Options
- Time window selection (1h, 4h, 1d, custom)
- Price range (auto, manual)
- Color scheme (heat, cool, custom)
- Sampling resolution (balance detail vs performance)

### 6.4 Settings Integration

#### 6.4.1 Volume Settings Section
**Location**: Within existing settings dialog

**Parameters Exposed**:
- Volume Profile calculation parameters
  - Window size (number of bars)
  - Value area percentage (default 70%)
  - HVN/LVN prominence thresholds
  
- VSA Analysis parameters
  - Volume MA period
  - Spread MA period
  - Volume thresholds (high, ultra, low)
  - Spread thresholds (narrow, wide)
  
- Order Flow parameters
  - Rolling window size
  - Imbalance threshold
  - Z-score threshold
  - Large order percentile

#### 6.4.2 DOM Settings Section
**Parameters Exposed**:
- WebSocket connection settings
  - Enable/disable DOM streaming
  - Auto-connect on startup
  - Reconnection strategy
  
- Display preferences
  - Default depth level
  - Update rate (real-time, throttled)
  - Floating widget position/size memory
  
- Data retention
  - Snapshot storage duration
  - Cleanup policy

#### 6.4.3 Visualization Settings
**Parameters Exposed**:
- Chart overlays
  - Volume Profile visibility
  - VSA signals visibility
  - Order flow indicators visibility
  
- Color schemes
  - Volume bars colors
  - VSA signal icons/colors
  - DOM heatmap palette
  
- Performance settings
  - Rendering throttling
  - Maximum visible data points

---

## 7. Integration Points

### 7.1 Workflow Integration

#### 7.1.1 Application Startup Sequence
**Integration Point**: Main application initialization

**Requirements**:
- Initialize DOM aggregation service after database connection
- Start WebSocket connections if enabled in settings
- Register event listeners for data flow
- Restore UI state (floating panels, settings)

#### 7.1.2 Symbol Change Workflow
**Integration Point**: When user selects different symbol

**Requirements**:
- Unsubscribe from previous symbol's DOM stream
- Subscribe to new symbol's DOM stream
- Load historical DOM data for new symbol
- Update order book widget
- Recalculate volume profile for new symbol
- Clear and reload VSA signals

#### 7.1.3 Timeframe Change Workflow
**Integration Point**: When user selects different timeframe

**Requirements**:
- Trigger volume profile recalculation with appropriate window
- Reload VSA signals for timeframe
- Adjust order flow analysis window
- Update volume indicators on chart

#### 7.1.4 Data Update Workflow
**Integration Point**: When new market data arrives

**Requirements**:
- Route tick data to volume analysis
- Route DOM snapshots to aggregation service
- Trigger UI updates via event system
- Update cached calculations incrementally
- Maintain UI responsiveness (non-blocking updates)

### 7.2 Component Interconnections

#### 7.2.1 Provider → Processing Layer
**Data Flow**:
- Provider receives market data (candles, ticks, DOM)
- Provider normalizes data to internal format
- Provider pushes data to appropriate service queue
- Service processes and persists data
- Service emits events for downstream consumers

#### 7.2.2 Processing Layer → UI Layer
**Data Flow**:
- Service calculates derived metrics/signals
- Service emits update events via event bus
- UI components subscribe to relevant events
- UI components refresh displays upon event receipt
- UI maintains smooth user experience (throttling, batching)

#### 7.2.3 UI → Processing Layer (User Actions)
**Data Flow**:
- User adjusts settings or parameters
- UI validates inputs
- UI propagates changes to service layer
- Services recalculate affected metrics
- Services emit update events back to UI

### 7.3 Event Bus Integration

#### 7.3.1 Event Types Definition
Define and implement events for:
- `MarketDataUpdate`: New candle/tick available
- `DOMSnapshotUpdate`: New order book snapshot
- `VolumeProfileCalculated`: Volume profile metrics ready
- `VSASignalDetected`: New VSA signal generated
- `OrderFlowSignalDetected`: New order flow signal generated
- `SymbolChanged`: User switched symbol
- `TimeframeChanged`: User switched timeframe

#### 7.3.2 Publisher-Subscriber Patterns
**Best Practices**:
- Services publish events after data processing
- UI components subscribe during initialization
- Unsubscribe on component destruction
- Include relevant context in event payload
- Avoid cyclic dependencies

---

## 8. Configuration and Settings

### 8.1 Configuration File Structure

#### 8.1.1 Provider Configuration
**Section**: `providers.ctrader` (or other provider)

**New Parameters**:
- DOM streaming enabled flag
- WebSocket endpoint URL
- Subscription depth level
- Snapshot interval
- Reconnection policy

#### 8.1.2 Volume Analysis Configuration
**Section**: `features.volume_analysis`

**Parameters**:
- Volume Profile settings (window, thresholds)
- VSA settings (MA periods, thresholds)
- Order Flow settings (windows, thresholds)

#### 8.1.3 UI Configuration
**Section**: `ui.market_depth`

**Parameters**:
- Default widget visibility
- Widget dimensions and position
- Display preferences
- Color schemes

### 8.2 User Preferences Persistence

#### 8.2.1 Widget State Saving
Persist user adjustments:
- Floating panel positions and sizes
- Column widths in order book
- Enabled/disabled indicators
- Custom color schemes
- Preferred timeframes and symbols

#### 8.2.2 Settings Reset Mechanism
Provide functionality to:
- Reset individual sections to defaults
- Reset all settings globally
- Import/export settings profiles

---

## 9. Version Control Strategy

### 9.1 Commit Granularity

#### 9.1.1 Task-Level Commits
**Principle**: Each completed task represents a commit.

**Task Definition**: A task is a self-contained unit that:
- Adds a specific feature or component
- Fixes a specific issue
- Refactors a specific module
- Can be understood independently

**Examples**:
- "Add market_depth table schema via Alembic migration"
- "Implement DOM WebSocket subscription in cTraderProvider"
- "Create OrderBookWidget with bid/ask display"
- "Integrate Volume Profile calculation with event bus"

#### 9.1.2 Subtask-Level Commits
**Principle**: Complex tasks may have meaningful intermediate states.

**Subtask Definition**: A subtask:
- Completes a portion of a larger task
- Leaves codebase in stable, testable state
- Represents logical progression

**Examples**:
- "Add DOM message parsing (subtask 1/3: connection)"
- "Implement bid table rendering (subtask 2/4: order book UI)"
- "Wire Volume Profile to chart overlay (subtask 3/5: visualization)"

### 9.2 Commit Message Format

#### 9.2.1 Structure
```
<Type>: <Brief Description>

<Detailed functional description of what was accomplished>
<Why this change was necessary>
<How it integrates with existing components>

<Optional: Related task numbers or references>
```

#### 9.2.2 Type Prefixes
- **feat**: New feature addition
- **fix**: Bug fix or correction
- **refactor**: Code restructuring without feature change
- **integrate**: Connecting components or systems
- **ui**: User interface changes
- **db**: Database schema or migration changes
- **config**: Configuration or settings changes
- **docs**: Documentation updates

#### 9.2.3 Descriptive Quality
Commit messages must answer:
- What functionality was added/changed?
- What problem does it solve?
- How does it work with existing code?
- What are the integration points?

**Example**:
```
feat: Add DOM aggregation service for order book processing

Implemented fully functional DOMAggreg atorService that processes 
real-time order book snapshots from WebSocket providers. Service 
calculates mid price, spread, and imbalance metrics, persisting 
results to market_depth table.

Integrated with existing provider health monitoring and event bus 
for downstream UI updates. Service starts automatically on app 
initialization and handles reconnections gracefully.

Connects to: cTraderProvider DOM streaming, market_depth table, 
event bus (DOMSnapshotUpdate events)
```

### 9.3 Integration Testing Between Commits

#### 9.3.1 Pre-Commit Verification
Before committing, verify:
- Code runs without errors
- New functionality works as intended
- Existing functionality not broken
- No compilation/import errors
- Configuration files valid

#### 9.3.2 Rollback Safety
Each commit should:
- Be revertible without breaking system
- Contain complete unit of work
- Not leave half-implemented features

---

## 10. Dependency Management

### 10.1 Library Additions

#### 10.1.1 Python Dependencies
All new libraries must be added to `pyproject.toml` or equivalent:
- Specify minimum version requirements
- Document purpose of dependency
- Group by functionality (e.g., visualization, data processing)

**Potential New Dependencies**:
- WebSocket client libraries (if not already present)
- Additional plotting/charting libraries for heatmap
- JSON schema validation libraries (for DOM data)

#### 10.1.2 Dependency Justification
For each new dependency, document:
- What functionality it provides
- Why existing libraries insufficient
- Version compatibility considerations
- License compatibility

### 10.2 Cleanup of Unused Code

#### 10.2.1 Orphan Detection
Before finalization, audit for:
- Unused import statements
- Placeholder methods never called
- Dead code paths
- Unreferenced configuration parameters
- Empty or stub files

#### 10.2.2 Deprecation Strategy
If removing existing functionality:
- Document what is being removed and why
- Provide migration path if applicable
- Warn users in advance if breaking change

---

## 11. Testing Considerations

### 11.1 Functional Testing

#### 11.1.1 Data Flow Validation
Verify end-to-end data flow:
- Provider → Database → Processing → UI
- Events propagate correctly
- No data loss or corruption
- Timestamps remain consistent

#### 11.1.2 UI Responsiveness
Test user interface under load:
- High-frequency DOM updates
- Large historical data loads
- Rapid symbol switching
- Multiple indicators enabled simultaneously

### 11.2 Edge Case Handling

#### 11.2.1 Missing Data Scenarios
System must handle gracefully:
- No DOM data available for symbol
- Historical data gaps
- Provider disconnections
- Malformed messages

#### 11.2.2 Configuration Edge Cases
Test with:
- Invalid parameter values
- Extreme threshold settings
- Disabled features
- First-time user (no saved preferences)

---

## 12. Performance Optimization

### 12.1 Database Query Optimization

#### 12.1.1 Index Strategy
Ensure appropriate indexes on:
- (symbol, ts_utc) for time-range queries
- (symbol, timeframe, ts_utc) for candle lookups
- Frequently filtered columns

#### 12.1.2 Query Efficiency
- Limit result sets to necessary range
- Use pagination for large historical queries
- Cache frequently accessed data
- Avoid N+1 query patterns

### 12.2 UI Rendering Optimization

#### 12.2.1 Update Throttling
- Batch rapid updates before rendering
- Implement frame-rate limiting (e.g., 30 FPS max)
- Skip intermediate states when updates faster than render

#### 12.2.2 Data Downsampling
For visualization:
- Reduce data points when zoomed out
- Aggregate bars when rendering large ranges
- Use level-of-detail rendering techniques

---

## 13. Documentation Requirements

### 13.1 Code Documentation

#### 13.1.1 Docstring Standards
All new functions/classes must have:
- Purpose description
- Parameter descriptions with types
- Return value description
- Usage examples where appropriate
- Notes on integration points

#### 13.1.2 Inline Comments
Complex logic should include:
- Explanation of algorithm choices
- Clarification of non-obvious code
- References to external documentation
- Warnings about edge cases

### 13.2 User-Facing Documentation

#### 13.2.1 Feature Documentation
Create user guides for:
- How to enable DOM streaming
- Interpreting order book widget
- Understanding Volume Profile indicators
- Using VSA signals for trading decisions
- Configuring visualization settings

#### 13.2.2 Settings Documentation
Document each configuration parameter:
- What it controls
- Valid value range
- Default value and reasoning
- Impact on performance or behavior

---

## 14. Future Extensibility

### 14.1 Modular Design

#### 14.1.1 Component Interfaces
Design components with clear interfaces:
- Abstract base classes for providers
- Pluggable analysis modules
- Swappable visualization backends

#### 14.1.2 Extension Points
Identify where future enhancements attach:
- Additional VSA signal types
- Custom volume indicators
- Alternative DOM visualizations
- Machine learning integration for order flow

### 14.2 Scalability Considerations

#### 14.2.1 Multi-Symbol Support
Architecture should support:
- Multiple simultaneous DOM streams
- Parallel volume analysis across symbols
- Shared vs isolated processing resources

#### 14.2.2 Data Retention Policies
Design for growth:
- Configurable retention periods
- Automated archival of old data
- Data compression strategies
- Efficient purging mechanisms

---

## 15. Implementation Priority

### 15.1 Phase 1: Core Infrastructure
**Focus**: Establish foundational data flow

1. Database schema for market_depth
2. DOM streaming in provider(s)
3. DOM aggregation service activation
4. Basic persistence and retrieval

**Success Criteria**: DOM data flowing from provider to database

### 15.2 Phase 2: Processing Integration
**Focus**: Connect analysis engines

1. Volume Profile integration with data updates
2. VSA signal generation and storage
3. Order Flow signal generation
4. Event bus wiring for all components

**Success Criteria**: Signals being generated and stored

### 15.3 Phase 3: Visualization
**Focus**: User-facing displays

1. Volume indicators on main chart
2. Order Book floating widget
3. VSA signal overlays
4. Volume Profile chart overlay

**Success Criteria**: Users can view all analysis results

### 15.4 Phase 4: Advanced Features
**Focus**: Enhanced capabilities

1. DOM heatmap visualization
2. Advanced order flow patterns
3. Multi-timeframe volume analysis
4. Configuration and customization UI

**Success Criteria**: Full feature set operational

---

## 16. Acceptance Criteria

### 16.1 Functional Completeness
The implementation is complete when:
- All listed features implemented and functional
- All components integrated and communicating
- No placeholder code remaining
- All settings exposed in GUI
- Documentation complete

### 16.2 Quality Standards Met
The implementation meets quality when:
- No orphaned files or methods
- All dependencies declared properly
- Database migrations successful
- Commits follow specified strategy
- Code follows project conventions

### 16.3 User Experience
The implementation is user-ready when:
- UI responsive under typical load
- Settings intuitive and discoverable
- Visualizations clear and informative
- Error messages helpful
- Configuration persists across sessions

---

## 17. Notes for Implementation Agent

### 17.1 Decision Authority
The implementing agent has full authority over:
- Specific algorithms and data structures
- Code organization and file structure
- Implementation details and optimizations
- Library and framework selection (within project constraints)
- Refactoring approaches

### 17.2 Constraints to Respect
The implementing agent must adhere to:
- Existing project architecture patterns
- Database management via SQLAlchemy/Alembic
- Configuration management approach
- Event bus patterns
- UI framework and conventions

### 17.3 When to Seek Clarification
Request clarification if:
- Specification conflicts with existing code
- Requirements ambiguous or contradictory
- Technical limitation prevents direct implementation
- Breaking change necessary to existing API

---

## Document Status

**Current Phase**: Specification - Awaiting Additional Requirements

**Next Steps**:
1. Review and validate specifications
2. Identify any additional requirements
3. Approve for implementation
4. Begin Phase 1 implementation

**Change Log**:
- 2025-10-07: Initial draft created
- [Future additions will be appended here]

---

## Appendices

### Appendix A: Glossary

- **DOM**: Depth of Market - order book showing bid/ask prices and volumes
- **POC**: Point of Control - price level with highest volume
- **VA**: Value Area - price range containing 70% of total volume
- **HVN**: High Volume Node - local maximum in volume distribution
- **LVN**: Low Volume Node - local minimum in volume distribution
- **VSA**: Volume Spread Analysis - technique analyzing volume relative to price range
- **Order Flow**: Analysis of buying/selling pressure from market orders

### Appendix B: Reference Implementation Locations

**Existing Code to Integrate**:
- Volume Profile: `src/forex_diffusion/features/volume_profile.py`
- VSA Analysis: `src/forex_diffusion/features/vsa.py`
- Order Flow: `src/forex_diffusion/analysis/order_flow_analyzer.py`
- DOM Service: `src/forex_diffusion/services/dom_aggregator.py` (placeholder)
- Provider Base: `src/forex_diffusion/providers/base.py`
- cTrader Provider: `src/forex_diffusion/providers/ctrader_provider.py`
- Chart UI: `src/forex_diffusion/ui/chart_tab_ui.py`

### Appendix C: Related Specifications

[This section reserved for cross-references to other specification documents]

---

**End of Document**
