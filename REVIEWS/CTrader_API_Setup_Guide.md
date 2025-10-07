# cTrader API Setup Guide

**Date:** 2025-10-07
**Purpose:** Connect ForexGPT to cTrader for real-time data

---

## 🔑 Credenziali Fornite

**Account cTrader Demo:**
- Username: `a.taini`
- Password: `hsaN123123!!`

**⚠️ IMPORTANTE:** Questi sono credenziali di **LOGIN cTrader**, NON credenziali API.

---

## 📋 Problema: API vs Login Credentials

### Cosa Servono le Credenziali di Login?
- ✅ Accesso alla piattaforma cTrader
- ✅ Trading manuale
- ✅ Visualizzazione grafici

### Cosa NON Fanno?
- ❌ NON funzionano con cTrader Open API
- ❌ NON permettono accesso WebSocket programmatico
- ❌ NON possono essere usate nel codice

---

## 🔧 Soluzione: 2 Opzioni Disponibili

### Opzione A: Registrare App cTrader (Raccomandato) ✅

**Steps:**

1. **Vai a cTrader Open API Portal:**
   - URL: https://openapi.ctrader.com/
   - Click "Sign In" o "Register"

2. **Login con le tue credenziali cTrader:**
   - Username: `a.taini`
   - Password: `hsaN123123!!`

3. **Crea una nuova applicazione:**
   - Click "Create New App" o "Applications"
   - Nome app: "ForexGPT"
   - Redirect URI: `http://localhost:5000/callback`
   - Scopes richiesti: `trading`, `accounts`

4. **Ottieni credenziali OAuth:**
   Riceverai:
   - ✅ Client ID (es: `1234_abcd...`)
   - ✅ Client Secret (es: `secret_xyz...`)

5. **Esegui OAuth Flow:**
   ```bash
   python scripts/get_ctrader_token.py
   ```

   Questo:
   - Apre browser
   - Login automatico (sei già loggato)
   - Ottiene `access_token`
   - Ottiene `account_id`

6. **Usa nel test:**
   ```bash
   $env:CTRADER_CLIENT_ID = "tuo_client_id"
   $env:CTRADER_CLIENT_SECRET = "tuo_client_secret"
   $env:CTRADER_ACCESS_TOKEN = "access_token_ottenuto"
   $env:CTRADER_ACCOUNT_ID = "tuo_account_id"

   python tests/test_ctrader_websocket.py
   ```

**Vantaggi:**
- ✅ Ufficiale e supportato
- ✅ Accesso completo a tutte le API
- ✅ WebSocket, REST, order execution
- ✅ Tokens rinnovabili

**Tempo richiesto:** 10-15 minuti

---

### Opzione B: Usare Alternative (Più Veloce) 🚀

Se NON vuoi registrare l'app cTrader, possiamo:

#### B1. Mock Data per Test

Creo un **mock WebSocket** che simula dati cTrader per testare tutto il resto del sistema:

```python
# Test con dati simulati
python tests/test_ctrader_websocket_mock.py
```

**Pro:**
- ✅ Funziona subito senza setup
- ✅ Testa tutto il codice
- ✅ Valida order flow, volume, sentiment

**Contro:**
- ❌ Non sono dati reali di mercato

---

#### B2. Usare Altro Provider

Altri provider già configurati nel sistema:

1. **Tiingo** (già implementato):
   - WebSocket real-time
   - Richiede solo API key (gratis)
   - Forex e crypto supportati

2. **Alpha Vantage**:
   - REST API per dati storici
   - API key gratuita
   - 5 richieste/minuto

**Setup Tiingo (5 minuti):**
```bash
# 1. Registra su tiingo.com → Get API Key
# 2. Set env var
$env:TIINGO_API_KEY = "your_api_key"

# 3. Test
python tests/test_tiingo_websocket.py
```

---

## 🎯 Raccomandazione

### Per Test Immediato (Oggi):
**Usa Opzione B1 (Mock Data)** - Funziona subito, valida tutto il codice

### Per Produzione/Demo Reale:
**Usa Opzione A (cTrader Official API)** - Setup 15 min, dati reali

---

## 📝 Checklist Setup cTrader

- [ ] Login a https://openapi.ctrader.com/ con a.taini
- [ ] Crea app "ForexGPT"
- [ ] Copia Client ID
- [ ] Copia Client Secret
- [ ] Esegui `python scripts/get_ctrader_token.py`
- [ ] Autorizza app nel browser
- [ ] Copia access_token
- [ ] Trova account_id dal response
- [ ] Set environment variables
- [ ] Run `python tests/test_ctrader_websocket.py`
- [ ] Verifica order book updates
- [ ] Verifica volume data
- [ ] Verifica sentiment

---

## 🔍 Alternative: Fix FIX Protocol

cTrader supporta anche **FIX Protocol** per trading, ma:
- ❌ Più complesso da implementare
- ❌ Richiede certificati
- ❌ Non supporta WebSocket data streaming
- ✅ Opzione per order execution solo

**Non raccomandato** per data streaming.

---

## 💡 Cosa Faccio Ora?

Ho 3 opzioni immediate:

### 1. Creo Mock Test (5 minuti)
Testo tutto il sistema con dati simulati realistici

### 2. Aspetto Setup cTrader (da te)
Tu registri l'app, mi dai le credenziali OAuth

### 3. Uso Tiingo (10 minuti)
Implemento test con Tiingo WebSocket (già supportato)

**Quale preferisci?**

---

## 📚 Risorse

- cTrader Open API Docs: https://help.ctrader.com/open-api/
- OAuth 2.0 Guide: https://help.ctrader.com/open-api/authentication/
- API Playground: https://openapi.ctrader.com/playground

---

**Generato da:** Claude Code
**Per:** Test cTrader WebSocket Integration
