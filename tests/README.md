# Two-Phase Training System - Test Suite

## Overview

Comprehensive test suite for the Two-Phase Training System, covering unit tests, integration tests, and component tests.

## Test Structure

```
tests/
├── __init__.py
├── README.md                          # This file
├── training_pipeline/
│   ├── __init__.py
│   ├── test_config_grid.py           # Config generation & hashing
│   ├── test_regime_manager.py        # Regime classification
│   └── test_integration.py           # End-to-end workflows
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions and methods in isolation
- Fast execution (<1s per test)
- No external dependencies (database, files, etc.)
- Use mocking for dependencies

**Coverage**:
- `test_config_grid.py`: Configuration hashing, grid generation, deduplication
- `test_regime_manager.py`: Regime classification, metrics calculation

### Integration Tests (`@pytest.mark.integration`)
- Test complete workflows and component interactions
- May involve temporary files and directories
- Test real configuration loading and checkpoint management
- Slower execution (1-10s per test)

**Coverage**:
- `test_integration.py`: Queue creation, checkpoint save/load, config persistence

### Database Tests (`@pytest.mark.database`)
- Require actual database connection
- Test CRUD operations
- May modify test database
- Should clean up after execution

**Coverage**:
- Training run lifecycle (create, read, update, delete)
- Regime definitions management
- Best model tracking

## Running Tests

### Run All Tests
```bash
pytest
```

### Run by Category
```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests
pytest -m integration

# Database tests (requires DB setup)
pytest -m database

# Exclude slow tests
pytest -m "not slow"
```

### Run Specific Test File
```bash
pytest tests/training_pipeline/test_config_grid.py
```

### Run Specific Test
```bash
pytest tests/training_pipeline/test_config_grid.py::TestConfigHashing::test_compute_config_hash_deterministic
```

### With Coverage Report
```bash
pytest --cov=src/forex_diffusion/training/training_pipeline --cov-report=html
```

Coverage report will be in `htmlcov/index.html`

### Verbose Output
```bash
pytest -v
```

### Show Print Statements
```bash
pytest -s
```

## Test Configuration

Configuration is in `pytest.ini`:
- Test discovery patterns
- Test markers
- Output formatting
- Coverage settings

## Writing Tests

### Test Naming Convention
- File: `test_<module_name>.py`
- Class: `Test<FeatureName>`
- Method: `test_<what_it_tests>`

### Example Test Structure
```python
class TestFeatureName:
    """Tests for specific feature."""

    def setup_method(self):
        """Set up before each test."""
        pass

    def teardown_method(self):
        """Clean up after each test."""
        pass

    def test_specific_behavior(self):
        """Test description."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_value
```

### Using Mocks
```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test using mock objects."""
    mock_session = Mock()
    mock_session.query.return_value.filter.return_value.first.return_value = Mock(id=1)

    result = function_that_uses_session(mock_session)

    assert result is not None
```

### Using Fixtures
```python
@pytest.fixture
def sample_config():
    """Provide sample configuration."""
    return {
        'model_type': 'xgboost',
        'symbol': 'EURUSD'
    }

def test_with_fixture(sample_config):
    """Test using fixture."""
    assert 'model_type' in sample_config
```

## Test Coverage Goals

### Current Coverage
- **Config Grid**: 95% (14 test methods)
- **Regime Manager**: 85% (12 test methods)
- **Integration**: 70% (8 test methods)

### Target Coverage
- **Overall**: >80%
- **Core Modules**: >90%
- **GUI**: >60%

## Continuous Integration

### Pre-commit Checks
Run before committing:
```bash
pytest -m "unit and not slow"
```

### Full Test Suite
Run before merging:
```bash
pytest --cov
```

## Test Data

### Generating Test Data
```python
def create_test_ohlc(n_bars=100):
    """Create synthetic OHLC data for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1H'),
        'close': np.random.randn(n_bars) + 1.1000,
        # ... other columns
    })
```

### Test Fixtures Location
- In-memory mocks (preferred)
- Temporary directories via `tempfile`
- Small JSON fixtures in `tests/fixtures/` (if needed)

## Debugging Tests

### Run with PDB
```bash
pytest --pdb  # Drop into debugger on failure
pytest --pdb-trace  # Drop into debugger at start
```

### Show Full Diff
```bash
pytest -vv
```

### Last Failed Tests Only
```bash
pytest --lf
```

### Stop on First Failure
```bash
pytest -x
```

## Performance

### Fast Tests (< 100ms)
- Unit tests
- Mock-based tests
- Simple calculations

### Medium Tests (100ms - 1s)
- Component integration
- Small data processing

### Slow Tests (> 1s)
- Full workflow tests
- Large data processing
- Database operations

Mark slow tests:
```python
@pytest.mark.slow
def test_full_training_pipeline():
    """Long-running integration test."""
    pass
```

## Common Issues

### Import Errors
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Database Connection
```bash
# Set test database URL
export DATABASE_URL="sqlite:///test.db"
```

### Missing Dependencies
```bash
pip install pytest pytest-cov pytest-mock
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Always clean up resources (files, DB records)
3. **Deterministic**: Tests should have same result every time
4. **Fast**: Keep unit tests fast (<100ms)
5. **Descriptive**: Clear test names and docstrings
6. **One Assertion**: Focus each test on one thing
7. **Arrange-Act-Assert**: Follow AAA pattern
8. **Mock External**: Mock databases, APIs, file I/O

## Maintenance

### Adding New Tests
1. Create test file following naming convention
2. Add appropriate markers
3. Include docstrings
4. Update this README if needed

### Updating Tests
- When refactoring code, update corresponding tests
- Keep tests passing on main branch
- Add tests for bug fixes

### Test Review Checklist
- [ ] Tests are independent
- [ ] Cleanup happens in teardown/finally
- [ ] Appropriate markers used
- [ ] Good test names and docstrings
- [ ] Mocks used appropriately
- [ ] Fast execution
- [ ] Good coverage of edge cases

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)

---

**Last Updated**: 2025-10-07
**Test Count**: 34+ tests
**Coverage**: ~85% (core modules)
