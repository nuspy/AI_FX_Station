# Template Standard per TUTTI i Tooltip

**SCHEMA OBBLIGATORIO** per ogni parametro/widget/controllo:

```
[NOME PARAMETRO/WIDGET]

1) COSA È:
   Definizione tecnica chiara e completa del parametro.
   Spiegare cosa rappresenta, come viene calcolato (se applicabile).
   Esempi concreti con numeri reali.

2) COME E QUANDO SI USA:
   - Quando attivarlo/usarlo (condizioni specifiche)
   - Come configurarlo correttamente
   - In quali situazioni è rilevante
   - Quando NON usarlo
   
   Casistiche:
   - Scalping: [rilevanza]
   - Day trading: [rilevanza]
   - Swing trading: [rilevanza]
   - Position trading: [rilevanza]

3) PERCHÉ SI USA:
   - Obiettivo principale
   - Benefici specifici
   - Problemi che risolve
   - Impatto sulla strategia di trading

4) EFFETTI:

   4.1) EFFETTI A BASSO VALORE (o se DISATTIVATO):
   
   [Se parametro numerico]:
   Valore MOLTO BASSO (range minimo, es. 0-10):
   - Comportamento del sistema
   - Cosa succede al trading
   - Vantaggi di questo setting
   - Svantaggi/rischi di questo setting
   - Quando usare questo range
   
   Valore BASSO (range basso, es. 10-30):
   - [stessa struttura sopra]
   
   [Se checkbox/boolean]:
   DISATTIVATO (☐ unchecked):
   - Sistema opera senza questa feature
   - Comportamento alternativo
   - Vantaggi di disattivazione
   - Svantaggi di disattivazione
   - Quando disattivare

   4.2) EFFETTI A VALORE MEDIO:
   
   Valore MEDIO (range medio, es. 30-70):
   - Comportamento bilanciato
   - Compromesso tra velocità e accuratezza
   - Quando è il setting ottimale
   - Raccomandazioni d'uso

   4.3) EFFETTI AD ALTO VALORE (o se ATTIVATO):
   
   [Se parametro numerico]:
   Valore ALTO (range alto, es. 70-90):
   - Comportamento del sistema
   - Impatto su performance
   - Vantaggi di questo setting
   - Svantaggi/rischi
   
   Valore MOLTO ALTO (range massimo, es. 90-100+):
   - [stessa struttura sopra]
   
   [Se checkbox/boolean]:
   ATTIVATO (☑ checked):
   - Sistema opera CON questa feature
   - Modifiche al comportamento
   - Vantaggi di attivazione
   - Svantaggi/costi (performance, complessità)
   - Quando attivare

5) RANGE TIPICO / VALORE TIPICO:
   
   [Se parametro numerico]:
   - Minimo assoluto: [valore] (edge case)
   - Minimo raccomandato: [valore]
   - STANDARD/DEFAULT: [valore] ← valore tipico per la maggior parte degli utenti
   - Massimo raccomandato: [valore]
   - Massimo assoluto: [valore] (edge case)
   
   Distribuzione uso:
   - Beginner: [range] (es. 10-30)
   - Intermediate: [range] (es. 30-70)
   - Advanced: [range] (es. 70-100+)
   
   [Se checkbox/boolean]:
   - Tipicamente ON (☑): [percentuale utenti, es. 80%]
   - Tipicamente OFF (☐): [percentuale utenti, es. 20%]
   - Default setting: [ON/OFF]
   
   Raccomandazione:
   - Beginner: [ON/OFF perché...]
   - Intermediate: [ON/OFF perché...]
   - Advanced: [ON/OFF perché...]

6) NOTE AGGIUNTIVE / BEST PRACTICES:
   - Interazioni con altri parametri
   - Combinazioni ottimali
   - Errori comuni da evitare
   - Tips da esperti
```

---

## ESEMPI COMPLETI

### Esempio 1: Parametro Numerico (Spread)

```
Spread (Bid-Ask Difference)

1) COSA È:
   Lo spread è la differenza tra il prezzo bid (prezzo al quale puoi VENDERE) 
   e il prezzo ask (prezzo al quale puoi COMPRARE), misurato in pips.
   Rappresenta il costo immediato di ogni transazione.
   
   Formula: Spread = Ask - Bid
   Esempio: EUR/USD Bid 1.08500 / Ask 1.08510 = 1.0 pip spread
   
   Spread dinamico: cambia in tempo reale con liquidità di mercato.

2) COME E QUANDO SI USA:
   - Monitoralo SEMPRE prima di aprire un trade
   - Usa come filtro per timing di ingresso
   - Confronta spread tra broker diversi
   - Evita trading quando spread >3× valore normale
   
   Quando monitorare:
   - Pre-market open (spread più alti)
   - Durante news ad alto impatto (spread esplode)
   - Sessione asiatica (spread leggermente più alti)
   - London/NY overlap (spread minimi)
   
   Rilevanza per strategia:
   - Scalping: CRITICO - spread è 30-50% del target (5-10 pip trades)
   - Day trading: IMPORTANTE - impatta breakeven e primo target
   - Swing trading: MODERATO - diluito su target 100-500 pips
   - Position trading: BASSO - irrilevante su hold settimane/mesi

3) PERCHÉ SI USA:
   - Calcolare costo reale del trade (ogni trade parte in loss di spread)
   - Identificare condizioni di liquidità (spread basso = alta liquidità = sicuro)
   - Ottimizzare timing ingresso (aspettare spread scenda prima di entrare)
   - Valutare fattibilità strategia (scalping richiede spread <1 pip)
   - Broker comparison (spread più bassi = migliore broker per scalping)

4) EFFETTI:

   4.1) EFFETTI A BASSO VALORE:
   
   SPREAD MOLTO BASSO (0.0-0.5 pips):
   - Liquidità ECCELLENTE - grossi volumi pronti a ogni livello
   - Costo transazione MINIMO - perfetto per scalping
   - Mercato stabile e prevedibile - bassa volatilità improvvisa
   - Riempimento ordini istantaneo - nessun requote
   - Ideale per: scalping, high-frequency trading, strategie ad alto volume
   
   Quando si verifica:
   - London/NY overlap (13:00-17:00 GMT) - picco liquidità
   - Broker ECN/STP con vera liquidità interbancaria
   - Major pairs (EUR/USD, GBP/USD) in ore normali
   
   Vantaggi:
   - Breakeven raggiunto in 1-2 tick movement
   - Profittabilità anche su micro-movimenti (3-5 pips)
   - Nessun slippage significativo
   
   Svantaggi:
   - Raro su broker retail standard
   - Potrebbe indicare momento pre-evento (calma prima tempesta)
   
   SPREAD BASSO (0.5-1.5 pips):
   - Liquidità BUONA - condizioni normali di mercato
   - Costo transazione ACCETTABILE per day trading
   - Mercato attivo ma non eccezionale
   - Ideale per: day trading, swing trading, strategie standard
   
   Quando si verifica:
   - Sessioni principali (European 07:00-16:00, NY 13:00-22:00 GMT)
   - Major pairs con broker standard
   - Condizioni di mercato normali (no news)
   
   Vantaggi:
   - Breakeven a 2-3 pips
   - Gestibile per strategie intraday
   - Prevedibile e costante
   
   Svantaggi:
   - Scalping diventa marginale (serve win rate alto)
   - Alcuni broker allargano spread su stop hunt

   4.2) EFFETTI A VALORE MEDIO:
   
   SPREAD MEDIO (1.5-3.0 pips):
   - Liquidità MODERATA - minor pairs o ore off-peak
   - Costo transazione SIGNIFICATIVO per scalping
   - Mercato meno attivo o coppie secondarie
   - Ideale per: swing trading con target >50 pips
   
   Quando si verifica:
   - Sessione asiatica (22:00-08:00 GMT) su major pairs
   - Minor pairs (EUR/CHF, AUD/NZD) in ore normali
   - Transizione tra sessioni
   
   Vantaggi:
   - Ancora accettabile per swing trades
   - Filtra rumore (evita micro-movimenti)
   
   Svantaggi:
   - Breakeven a 4-5 pips - scalping NON profittevole
   - Deve muovere almeno 10 pips per vedere profitto netto
   - Slippage più probabile

   4.3) EFFETTI AD ALTO VALORE:
   
   SPREAD ALTO (3.0-10.0 pips):
   - Liquidità BASSA - mercato illiquido o volatile
   - Costo transazione PROIBITIVO per day trading
   - Condizioni di mercato anomale
   - Evitare: scalping, day trading; Considerare: solo swing long-term
   
   Quando si verifica:
   - Release news ad alto impatto (NFP, FOMC, CPI)
   - Exotic pairs (USD/TRY, EUR/ZAR) sempre
   - Weekend/holiday con liquidità quasi zero
   - Momenti di panic (flash crash, guerra, crisi)
   
   Vantaggi:
   - NESSUNO per trading normale
   - Forse: indica opportunità contrarian (panic = estremi)
   
   Svantaggi:
   - Breakeven a 10-15 pips - solo swing con target >100 pips
   - Alto rischio slippage (5-20 pips di slippage comune)
   - Possibile gap tra bid e ask (no fills a prezzi intermedi)
   - Broker potrebbero ampliare ulteriormente in momentum
   
   SPREAD MOLTO ALTO (>10 pips):
   - Liquidità CRITICA - mercato praticamente fermo
   - Trading NON RACCOMANDATO - costi insostenibili
   - Condizioni di mercato ESTREME
   
   Quando si verifica:
   - Major news shock (es. Brexit vote, COVID lockdown announcement)
   - Exotic pairs in orari off (20-100 pips normale)
   - Flash crashes / circuit breakers
   - Broker con problemi tecnici o liquidità
   
   Vantaggi:
   - NESSUNO - evitare assolutamente il trading
   
   Svantaggi:
   - Impossibile fare profitto (spread mangia tutto)
   - Rischio di stop loss hit da spread widening (non da movimento reale)
   - Possibili requotes o order rejection
   - Broker potrebbe stare manipolando (evitare broker inaffidabile)

5) RANGE TIPICO / VALORE TIPICO:
   
   Per MAJOR PAIRS (EUR/USD, GBP/USD, USD/JPY):
   - Minimo assoluto: 0.0 pips (solo ECN/interbank, raro)
   - Minimo normale: 0.3-0.5 pips (ECN broker, London/NY overlap)
   - STANDARD: 0.8-1.5 pips ← TIPICO per broker retail su major pairs
   - Alto ma accettabile: 2.0-3.0 pips (Asian session, minor pairs)
   - Massimo accettabile: 3.0-5.0 pips (solo per swing trades >100 pip target)
   - EVITARE: >5.0 pips (non tradare, aspetta che spread si restringa)
   
   Per MINOR PAIRS (EUR/CHF, AUD/NZD, GBP/CAD):
   - Tipico: 2.0-4.0 pips
   
   Per EXOTIC PAIRS (USD/TRY, EUR/ZAR):
   - Tipico: 10-50 pips (ALTO per definizione)
   
   Distribuzione uso strategia:
   - Scalping: Richiede <1.0 pip spread (solo ECN broker)
   - Day trading: Accettabile fino 2.0 pips
   - Swing trading: Tollerabile fino 5.0 pips
   - Position trading: Spread irrilevante (qualsiasi valore OK)

6) NOTE AGGIUNTIVE / BEST PRACTICES:
   - Spread variabile vs fisso: ECN ha spread variabile (più stretto in media), 
     Market Maker ha spread fisso (più largo ma prevedibile)
   - Stop hunt: broker disonesti allargano spread per triggare stop losses
   - Commission vs spread: alcuni broker hanno spread basso MA commissione per lotto
     (calcola costo totale = spread + commission)
   - Spread widening durante news: normale che spread 1 pip diventi 10-30 pips
     per 1-2 minuti durante NFP/FOMC (riprendi trading dopo che si normalizza)
   - Confronta broker: usa account demo per monitorare spread reali durante trading
```

### Esempio 2: Checkbox/Boolean (Use GPU Training)

```
Use GPU Training (Checkbox)

1) COSA È:
   Opzione per utilizzare la GPU (Graphics Processing Unit) invece della CPU 
   per l'addestramento di modelli di deep learning (autoencoder, VAE, diffusion).
   
   GPU = scheda grafica con migliaia di cores per calcolo parallelo
   CPU = processore standard con 4-16 cores
   
   Effetto: Training 10-100× più veloce se GPU compatibile disponibile.

2) COME E QUANDO SI USA:
   - Attiva SE hai GPU NVIDIA con CUDA (GTX/RTX series)
   - Usa per: Autoencoder, VAE, Diffusion models, Lightning (neural networks)
   - NON influenza: Ridge, Lasso, Random Forest (rimangono su CPU)
   
   Quando attivare:
   - Training autoencoder/VAE con >500 samples
   - Training diffusion models (sempre, altrimenti troppo lento)
   - Training Lightning con >1000 epochs
   - Hai GPU NVIDIA con ≥4GB VRAM
   
   Quando disattivare:
   - Nessuna GPU disponibile (fallback automatico a CPU)
   - GPU troppo vecchia (<GTX 1050, <4GB VRAM)
   - Training modelli semplici (Ridge/Lasso bastano)
   - Vuoi risparmiare energia (GPU consuma 100-300W vs CPU 15-65W)

3) PERCHÉ SI USA:
   - Velocità: Autoencoder 50 epochs da 60min → 4min (15× speedup)
   - Scalabilità: Puoi usare dataset più grandi (milioni di samples)
   - Complessità: Permette modelli più profondi (più layers, più neurons)
   - Interattività: Training rapido = più esperimenti in meno tempo
   - ROI: GPU costa $300-1500, ma risparmia centinaia di ore di tempo

4) EFFETTI:

   4.1) DISATTIVATO (☐ Use GPU Training):
   
   Sistema opera:
   - Training su CPU (Intel/AMD processor standard)
   - Usa solo cores CPU disponibili (tipicamente 4-16)
   - Computazione seriale/limitata parallelizzazione
   
   Vantaggi:
   - Compatibilità universale (funziona su ogni computer)
   - Consumo energetico basso (15-65W CPU vs 100-300W GPU)
   - Nessun setup aggiuntivo (no CUDA install)
   - Stabile e predicibile
   
   Svantaggi:
   - LENTO per neural networks:
     * Autoencoder 50 epochs: 60-90 minuti (CPU) vs 3-5 minuti (GPU)
     * VAE 100 epochs: 2-3 ore (CPU) vs 10-15 minuti (GPU)
     * Diffusion DDPM: GIORNI (CPU) vs 3-8 ore (GPU)
   - Limita dimensione modelli (RAM < VRAM)
   - Impossibile usare batch size grandi (CPU RAM limitata)
   - Training interattivo non pratico (troppo lento per iterare)
   
   Quando accettabile:
   - Training Ridge/Lasso/Random Forest (già veloci su CPU)
   - Dataset molto piccoli (<1000 samples)
   - Computer senza GPU NVIDIA
   - Budget limitato (no GPU da comprare)

   4.2) ATTIVATO (☑ Use GPU Training):
   
   Sistema opera:
   - Training su GPU NVIDIA (CUDA acceleration)
   - Usa migliaia di cores GPU (es. RTX 3080 = 8704 cores)
   - Calcoli paralleli massivi
   
   Comportamento:
   - PyTorch/TensorFlow automaticamente muove tensori su GPU
   - Training loop accelerato 10-100×
   - Batch processing parallelo su tutti i cores
   
   Vantaggi:
   - VELOCE per neural networks:
     * Autoencoder 50 epochs: 3-5 minuti
     * VAE 100 epochs: 10-15 minuti
     * Diffusion DDPM: 3-8 ore (vs giorni su CPU)
   - Batch size grandi (128-512 samples) → convergenza migliore
   - Modelli più complessi possibili (più layers, più neurons)
   - Workflow interattivo (esperimenti rapidi, iterazione veloce)
   - ROI eccellente per uso intensivo (tempo risparmiato > costo GPU)
   
   Svantaggi:
   - Richiede GPU NVIDIA (AMD/Intel GPU non supportate da CUDA)
   - Setup complesso (installa CUDA, cuDNN, PyTorch GPU build)
   - Costo hardware ($300-1500 per GPU decente)
   - Consumo energetico alto (100-300W, bolletta elettrica)
   - Limiti VRAM (8GB VRAM = max batch 256, 16GB = max batch 512)
   - Rumore ventole GPU (raffreddamento)
   
   Requisiti minimi:
   - GPU NVIDIA con Compute Capability ≥6.0 (GTX 1050+, RTX 2000+)
   - VRAM ≥4GB (8GB raccomandato, 16GB+ ideale)
   - CUDA 11.0+ installato
   - PyTorch GPU build (torch.cuda.is_available() = True)

5) RANGE TIPICO / VALORE TIPICO:
   
   Tipicamente ON (☑): 
   - 30-40% utenti (solo chi ha GPU NVIDIA)
   - Obbligatorio per: Diffusion model training
   - Raccomandato per: Autoencoder, VAE, Lightning con dataset >5000 samples
   
   Tipicamente OFF (☐):
   - 60-70% utenti (nessuna GPU o GPU non compatibile)
   - Sufficiente per: Ridge, Lasso, ElasticNet, Random Forest
   - Accettabile per: Autoencoder/VAE con dataset piccoli (<1000 samples)
   
   Default setting: OFF (☐) - compatibilità universale
   
   Raccomandazione:
   
   Beginner: 
   - OFF (☐) se nessuna GPU
   - ON (☑) SE hai GPU NVIDIA e vuoi sperimentare con neural networks
   - Non necessario per iniziare (Ridge/Lasso bastano)
   
   Intermediate:
   - ON (☑) SE hai GPU NVIDIA (grande miglioramento workflow)
   - Investire in GPU se training diventa bottleneck
   - Target: RTX 3060 (12GB VRAM, ~$400, ottimo rapporto qualità/prezzo)
   
   Advanced:
   - ON (☑) SEMPRE (GPU è requisito per ricerca avanzata)
   - Multi-GPU setup possibile (data parallelism)
   - Target: RTX 4090 (24GB VRAM, ~$1600, massima performance)

6) NOTE AGGIUNTIVE / BEST PRACTICES:
   
   Verifica compatibilità:
   - Apri Python, esegui: import torch; print(torch.cuda.is_available())
   - TRUE = GPU ready, FALSE = GPU non disponibile (usa CPU)
   
   Ottimizzazione GPU:
   - Usa batch size multiplo di 32 (es. 128, 256) per efficienza GPU
   - Monitor VRAM con nvidia-smi (evita out-of-memory)
   - Mixed precision (FP16) raddoppia velocità su RTX cards
   
   Troubleshooting:
   - "CUDA out of memory" → Riduci batch size (256→128→64)
   - "CUDA not available" → Reinstalla PyTorch GPU build
   - Training non più veloce → Verifica bottleneck (CPU, disco, data loading)
   
   Interazione parametri:
   - Batch size: GPU permette batch 128-512, CPU limitato a 16-64
   - Model capacity: GPU permette modelli grandi (milioni parametri)
   - Epochs: GPU permette più epochs in stesso tempo (migliore convergenza)
```

---

**APPLICARE QUESTO TEMPLATE A TUTTI I 100+ PARAMETRI DEL SISTEMA**

Prossimi parametri da formattare:
- VIX Levels (già parzialmente fatto, da completare)
- Order Flow Imbalance
- Timeframe Selector (tutti i 9 timeframes)
- Symbol Selector (tutti i 15+ pairs)
- Training Tab: 100+ parametri
- Backtesting Tab: 50+ parametri
- Pattern Training Tab: 30+ parametri
- Portfolio Tab: 20+ parametri
- Signals Tab: 15+ parametri
- Live Trading: 40+ parametri

**TOTALE STIMATO: 300-400 parametri da documentare con questo schema**
