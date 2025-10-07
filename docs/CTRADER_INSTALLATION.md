# cTrader Integration Installation Guide

## Overview

The cTrader integration allows ForexGPT to connect to cTrader broker accounts for live trading. Due to dependency conflicts between cTrader's API library and TensorFlow, the cTrader package must be installed separately.

## Installation Steps

### 1. Install cTrader Open API (without dependencies)

```bash
pip install ctrader-open-api==0.9.2 --no-deps
```

### 2. Verify Installation

The following dependencies should already be installed by ForexGPT:
- ✅ `twisted>=23.0.0` (cTrader requires >=21.7.0)
- ✅ `protobuf>=5.28.0` (cTrader wants 3.20.1, but works with newer versions)
- ✅ `pyOpenSSL>=24.0.0`
- ✅ `requests>=2.32.0`
- ✅ `inputimeout>=1.0.4`

### 3. Test Connection

1. Open ForexGPT
2. Go to **Settings** → **Data Providers** tab
3. Click **"Authorize cTrader (OAuth)"**
4. Follow the OAuth flow in your browser
5. Click **"Test Connection"**

If the test succeeds, the integration is working correctly despite the protobuf version mismatch.

## Why This Approach?

### The Conflict

- **TensorFlow** (required for AI forecasting) needs `protobuf>=5.28.0`
- **ctrader-open-api** pins to `protobuf==3.20.1`

### The Solution

1. **Install cTrader without dependencies** (`--no-deps`)
2. **Use newer protobuf** (6.32.1) required by TensorFlow
3. **cTrader API works** with newer protobuf due to backward compatibility

### Backward Compatibility

Protocol Buffers maintains backward compatibility:
- Code compiled with protobuf 3.x can run with protobuf 5.x/6.x runtime
- The API surface is stable across major versions
- Only serialization format matters, and it's compatible

## Troubleshooting

### Connection Test Fails

If you get protobuf-related errors:

```bash
# Verify protobuf version
pip show protobuf

# Should show: Version: 6.32.1 (or >=5.28.0)
```

### Import Errors

If you get `ModuleNotFoundError` for ctrader_open_api:

```bash
# Reinstall without deps
pip uninstall ctrader-open-api
pip install ctrader-open-api==0.9.2 --no-deps
```

### OAuth Flow Issues

1. Check your **Client ID** and **Client Secret** are correct
2. Ensure **Redirect URI** matches: `http://localhost:8080/callback`
3. Verify **Environment** setting (demo vs live)

## Alternative: Separate Virtual Environment (Not Recommended)

If you absolutely need protobuf 3.20.1 for some reason:

```bash
# Create separate venv for cTrader only
python -m venv venv_ctrader
venv_ctrader\Scripts\activate
pip install ctrader-open-api==0.9.2
```

**Note:** This is not recommended as it breaks the integrated workflow.

## Status

✅ **Current Status**: cTrader integration works with protobuf 6.32.1

The integration has been tested and confirmed working despite pip's dependency warnings.
