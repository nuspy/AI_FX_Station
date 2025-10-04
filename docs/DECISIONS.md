# Architectural Decision Records (ADR)

## Overview

This document records important architectural decisions made during the multi-provider integration project.

---

## ADR-001: Provider Abstraction Pattern

**Date**: 2025-01-05
**Status**: Accepted
**Context**: Need to support multiple market data providers (Tiingo, cTrader, future providers)

### Decision

Implement Strategy Pattern with `BaseProvider` abstract class:
- Each provider implements `BaseProvider` interface
- Providers declare supported capabilities via `ProviderCapability` enum
- `ProviderManager` acts as factory and lifecycle manager

### Rationale

**Pros**:
- Clean separation of concerns
- Easy to add new providers without modifying existing code
- Capability-based routing allows intelligent provider selection
- Testable in isolation (mock providers)

**Cons**:
- Additional abstraction layer adds complexity
- Need to maintain interface compatibility

**Alternatives Considered**:
1. **Direct provider implementations**: Too tightly coupled, hard to extend
2. **Plugin system with dynamic loading**: Overkill for current needs
3. **Adapter pattern only**: Lacks capability discovery

### Consequences

- All new providers must implement `BaseProvider`
- Provider-specific features require new capabilities
- Interface changes require updating all providers

---

## ADR-002: Credential Storage

**Date**: 2025-01-05
**Status**: Accepted
**Context**: Need secure storage for API keys, OAuth tokens, secrets

### Decision

Use OS-level keyring with Fernet encryption:
- Store credentials in system keyring (Windows Credential Manager, macOS Keychain, etc.)
- Additional Fernet encryption layer
- No plaintext credentials in files or environment variables (except override)

### Rationale

**Pros**:
- OS-level security leverages platform best practices
- Fernet encryption adds defense in depth
- Survives application restart without re-authentication
- Cross-platform support via `keyring` library

**Cons**:
- Requires `keyring` system support
- Complexity in setup (OS keyring configuration)
- Cannot easily share credentials between machines

**Alternatives Considered**:
1. **Plaintext files**: Insecure, rejected
2. **Encrypted files only**: No OS integration, manual key management
3. **Cloud secret management (AWS Secrets Manager)**: Overkill, adds external dependency

### Consequences

- Users must have functioning OS keyring
- Migration to new machine requires re-authentication
- Encrypted credentials tied to specific installation

---

## ADR-003: Twisted to asyncio Bridge

**Date**: 2025-01-05
**Status**: Accepted
**Context**: cTrader Open API uses Twisted, rest of codebase uses asyncio

### Decision

Bridge Twisted callbacks to asyncio via `asyncio.Queue`:
- Twisted callbacks push messages to thread-safe queue
- asyncio consumers pull from queue via `AsyncIterator`
- Keep Twisted isolated to provider implementation

### Rationale

**Pros**:
- Maintains asyncio consistency in codebase
- Twisted isolated to cTrader provider only
- Works with existing asyncio event loop

**Cons**:
- Queue overhead (memory, latency)
- Complex debugging (two async frameworks)
- Potential deadlocks if not careful

**Alternatives Considered**:
1. **Full migration to Twisted**: Would require rewriting entire codebase
2. **Run separate Twisted reactor thread**: Complicated event loop management
3. **Use cTrader REST only**: Loses real-time WebSocket capabilities

### Consequences

- Queue size must be tuned (default: 10,000 messages)
- Need careful thread/event loop coordination
- WebSocket disconnects require queue drain

---

## ADR-004: Database Schema Extension

**Date**: 2025-01-05
**Status**: Accepted
**Context**: Need to store multi-provider data without breaking existing schema

### Decision

Extend `market_data_candles` with nullable columns + create new tables:
- Add `tick_volume`, `real_volume`, `provider_source` to existing table
- Create separate tables for DOM, sentiment, news, calendar
- Maintain backward compatibility

### Rationale

**Pros**:
- Zero breaking changes (all new columns nullable)
- Existing queries still work
- Alembic migration reversible
- Provider-specific data in dedicated tables

**Cons**:
- Wider table (more columns)
- NULL handling required
- Query complexity increases

**Alternatives Considered**:
1. **Separate table per provider**: Data duplication, complex queries
2. **JSON blob for provider-specific data**: Loses type safety, hard to query
3. **Vertical partitioning**: Over-engineering for current scale

### Consequences

- Must handle NULL values in tick_volume/real_volume
- Queries should filter by provider_source for accuracy
- Migration 0007 required for multi-provider support

---

## ADR-005: Configuration Format (YAML vs TOML)

**Date**: 2025-01-05
**Status**: Accepted
**Context**: Existing codebase uses YAML, new docs suggested TOML

### Decision

Keep YAML for configuration:
- Existing system uses YAML with Pydantic validation
- Extend YAML with new provider sections
- Environment variable interpolation already implemented

### Rationale

**Pros**:
- No migration needed
- Pydantic models already defined
- Team familiarity with YAML
- Better comments and readability

**Cons**:
- YAML indentation sensitivity
- No strong typing (vs TOML)

**Alternatives Considered**:
1. **TOML**: Would require Pydantic model updates, migration effort
2. **JSON**: Less human-friendly, no comments
3. **Python files**: Security risk (code execution)

### Consequences

- Continue using YAML validation
- New providers add sections to `providers:` key
- Refresh rates in `refresh_rates:` section

---

## ADR-006: OAuth Flow Implementation

**Date**: 2025-01-05
**Status**: Accepted
**Context**: cTrader requires OAuth 2.0 authorization

### Decision

Implement Authorization Code Flow with localhost callback:
- Temporary HTTP server on localhost:5000
- Browser-based user authorization
- State parameter for CSRF protection
- Token storage in keyring

### Rationale

**Pros**:
- Standard OAuth 2.0 flow
- No external redirect required
- User-friendly (opens browser automatically)
- Secure with state parameter

**Cons**:
- Requires localhost port 5000 availability
- Firewall may block
- Server mode CLI requires manual URL visit

**Alternatives Considered**:
1. **Device flow**: Not supported by cTrader
2. **Client credentials flow**: Doesn't provide user-specific access
3. **External redirect**: Requires hosting callback server

### Consequences

- Port 5000 must be available
- Firewall exceptions may be needed
- Manual flow for headless/server deployments

---

## ADR-007: Failover Strategy

**Date**: 2025-01-05
**Status**: Accepted
**Context**: Need resilience when primary provider fails

### Decision

Automatic failover with health monitoring:
- Configure primary + secondary provider
- Health checks every 30 seconds
- Auto-failover on error rate > 10% or disconnect
- Swap primary ↔ secondary on failover

### Rationale

**Pros**:
- Automatic recovery
- Minimal downtime
- Health-based decision making
- Configurable thresholds

**Cons**:
- Complexity in state management
- Potential data inconsistency during swap
- Need both providers configured

**Alternatives Considered**:
1. **Manual failover only**: Requires human intervention
2. **Load balancing**: Complicates data consistency
3. **Circuit breaker pattern**: Adds complexity without clear benefit

### Consequences

- Both providers must support same capabilities
- Health monitoring adds overhead
- Failover events should be logged/alerted

---

## ADR-008: Volume Data Handling

**Date**: 2025-01-05
**Status**: Accepted
**Context**: Different providers report different volume types

### Decision

Store both tick_volume and real_volume separately:
- `tick_volume`: Number of price changes (available from most providers)
- `real_volume`: Actual traded volume in currency (cTrader only)
- Keep legacy `volume` column for backward compatibility

### Rationale

**Pros**:
- Preserves all available data
- Allows volume type selection in analysis
- Backward compatible

**Cons**:
- Three volume columns (confusing)
- NULL handling required
- Storage overhead

**Alternatives Considered**:
1. **Single volume column**: Loses information about volume type
2. **Metadata JSON**: Hard to query, loses type safety
3. **Provider-specific tables**: Increases query complexity

### Consequences

- Queries must specify which volume to use
- Analytics should handle NULL volumes gracefully
- Documentation must explain volume types

---

## ADR-009: Real-Time Data Caching

**Date**: 2025-01-05
**Status**: Accepted
**Context**: Some data (news, DOM) changes frequently but doesn't need persistence

### Decision

Two-tier caching strategy:
- **Persistent (Database)**: OHLCV, tick snapshots, calendar events
- **RAM Cache (LRU)**: News feed, DOM updates, sentiment ticks

### Rationale

**Pros**:
- Reduces database writes
- Fast access to real-time data
- Automatic eviction of stale data
- Configurable cache sizes

**Cons**:
- Data loss on restart
- Memory usage
- Cache invalidation complexity

**Alternatives Considered**:
1. **All persistent**: DB write overhead, disk space
2. **All cached**: Data loss risk
3. **Redis/Memcached**: External dependency, overkill

### Consequences

- Cache sizes must be tuned per use case
- Real-time data may not survive restart
- Need monitoring for cache hit rates

---

## ADR-010: GUI Framework (FinPlot)

**Date**: 2025-01-05
**Status**: Corrected
**Context**: Initial docs incorrectly referenced Qt, actual GUI uses FinPlot

### Decision

Use FinPlot for financial charting with custom overlays:
- News/Calendar as text overlays or separate panes
- Sentiment badge as chart annotation
- Volume bars as subplot
- DOM ladder as side panel (if supported)

### Rationale

**Pros**:
- Already in use and familiar to team
- Optimized for financial data
- Lightweight and fast
- Good customization options

**Cons**:
- Less flexible than full Qt
- Limited widget library
- May need custom rendering for complex features

**Alternatives Considered**:
1. **Qt (PyQt/PySide)**: Heavy, not already in use
2. **Plotly/Dash**: Web-based, different architecture
3. **Matplotlib**: Too slow for real-time

### Consequences

- GUI features limited by FinPlot capabilities
- Custom widgets may require low-level drawing
- Performance good for real-time updates

---

## Future Decisions

### Pending

1. **Message Queue for Distributed Architecture**: Redis pub/sub vs RabbitMQ vs Kafka
2. **Cloud Deployment Strategy**: Serverless (Lambda) vs Containers vs VMs
3. **Order Execution Integration**: Direct broker integration vs separate execution layer

### Deferred

1. **Machine Learning Integration**: Anomaly detection, sentiment analysis
2. **Multi-Instance Coordination**: Shared state management
3. **Advanced Analytics**: Custom indicators, backtesting framework

---

**Last Updated**: 2025-01-05
**Maintained By**: Development Team
