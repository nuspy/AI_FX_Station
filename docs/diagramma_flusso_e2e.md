# Spiegazione del Diagramma di Flusso End-to-End di ForexGPT

**Autore:** Gemini
**Data:** 20 ottobre 2025

---

Questo documento serve come guida e spiegazione per il diagramma di flusso dettagliato che si trova nel file `diagramma_flusso_e2e.drawio`. Il diagramma illustra l'intera architettura del sistema ForexGPT, dalla preparazione dei modelli all'esecuzione finale delle operazioni di trading.

Il flusso è diviso in due fasi concettuali principali:

1.  **Fase 1: Processi Offline** - Tutte le operazioni di addestramento, analisi e ottimizzazione che avvengono prima dell'inizio del trading live.
2.  **Fase 2: Processi Real-Time** - Il ciclo operativo che viene eseguito in tempo reale per analizzare il mercato e prendere decisioni di trading.

---

### Fase 1: Processi Offline (Addestramento e Ottimizzazione)

L'obiettivo di questa fase è preparare e calibrare tutti i componenti "intelligenti" del sistema. Questa fase non viene eseguita durante il trading live, ma periodicamente per mantenere i modelli aggiornati.

#### 1.1 Fonti Dati Offline

*   **Dati Storici:** La materia prima per tutto l'addestramento. Include serie storiche di prezzi (OHLCV), notizie passate, dati di sentiment storici, ecc.
*   **Storico Trade:** Un database contenente i risultati di tutte le operazioni passate (sia in backtest che in live), inclusi profitti/perdite (PnL), motivo di chiusura, e i parametri usati. Questo è fondamentale per il ciclo di feedback.

#### 1.2 Moduli di Addestramento e Ottimizzazione

Questi processi vengono eseguiti in parallelo:

*   **Ottimizzazione Parametri Pattern:** Il motore di riconoscimento dei pattern analizza i dati storici per trovare i parametri ottimali (es. soglie, lunghezze) per ogni tipo di pattern, salvandoli come artefatto.
*   **Addestramento Modelli AI:** Ogni modello di intelligenza artificiale viene addestrato sui dati storici per imparare a fare previsioni:
    *   **Diffusion Base (VAE):** Addestramento del modello generativo di base.
    *   **SSSD:** Addestramento del modello che combina State-Space Models e Diffusione, specializzato in previsioni probabilistiche multi-timeframe.
    *   **LDM4TS:** Addestramento del modello latente a diffusione, che usa un approccio basato su "vision" per analizzare le serie storiche.
*   **Ottimizzazione Adattiva Parametri (`AdaptiveParameterSystem`):** Questo è un componente di meta-apprendimento. Analizza lo **Storico Trade** per capire quali configurazioni del sistema portano a profitti o perdite, e ottimizza di conseguenza i parametri di alto livello (es. la soglia di qualità minima per un segnale, i moltiplicatori per il risk management).

#### 1.3 Artefatti Generati

L'output della fase offline è un insieme di "artefatti" pronti per essere caricati dal motore di trading live:
*   **Parametri Pattern Ottimizzati:** Le configurazioni più efficaci per il riconoscimento dei pattern.
*   **Checkpoint dei Modelli AI:** I file dei modelli addestrati (`.pt`, `.ckpt`) che contengono l'"intelligenza" appresa.
*   **Parametri di Rischio e Qualità Ottimizzati:** Le soglie e i moltiplicatori aggiornati dal sistema adattivo, pronti per essere usati dal `SignalQualityScorer` e dal `PositionSizer`.

---

### Fase 2: Processi Real-Time (Inferenza e Trading)

Questa è la fase operativa del sistema, un ciclo continuo che si ripete a intervalli definiti (es. ogni minuto).

#### 2.1 Fonti Dati Real-Time

*   **Dati di Mercato Real-Time:** Il flusso live di quotazioni, order book e notizie dal broker.
*   **Servizi Dedicati:** In parallelo, servizi come `SentimentAggregatorService` e `VIXService` processano i dati grezzi per estrarre informazioni di alto livello (es. sentiment contrarian, livello di volatilità VIX).

#### 2.2 Generazione dei Segnali

Tutti i seguenti componenti lavorano in parallelo per generare potenziali segnali di trading:
*   **Pattern Detection Engine:** Cerca pattern sui dati live usando i parametri ottimizzati.
*   **AI Inference Services (SSSD, LDM4TS):** Caricano i loro checkpoint e usano i dati live per generare previsioni probabilistiche.
*   **Analyzers (Order Flow, Eventi):** Analizzano l'order book e le notizie per generare segnali specifici.

#### 2.3 Il Cervello Decisionale: `UnifiedSignalFusion`

Questo è il cuore del sistema. Tutti i segnali grezzi generati al passo precedente convergono qui.
1.  **Scoring:** Il `SignalQualityScorer` interno valuta ogni segnale sulla base di molteplici dimensioni (affidabilità della fonte, conferma multi-timeframe, allineamento col sentiment, rischio di correlazione, ecc.).
2.  **Ranking:** I segnali vengono classificati in base al loro punteggio di qualità composito.
3.  **Output:** Viene prodotto un singolo **Miglior Segnale Classificato (`FusedSignal`)** che rappresenta la migliore opportunità di trading disponibile in quel momento.

#### 2.4 Gestione del Portafoglio: `IntelligentPortfolioManager`

Il miglior segnale viene passato a questo gestore, che esegue una fusione finale:
*   Prende il segnale qualitativo dall'AI (`FusedSignal`).
*   In parallelo, interroga il `Riskfolio Optimizer`, un modello analitico che calcola l'allocazione ottimale del portafoglio secondo la teoria di Markowitz.
*   Combina i due output (es. usando il segnale AI per "inclinare" il portafoglio calcolato da Riskfolio) per decidere i **Pesi Target Finali del Portafoglio**.

#### 2.5 Esecuzione e Risk Management: `AutomatedTradingEngine`

1.  **Sizing e Filtering:** Prima di eseguire, il motore applica un ultimo strato di filtri di "buon senso":
    *   **Filtro Sentiment e VIX:** La size della posizione viene aumentata o ridotta in base al sentiment e alla volatilità di mercato.
    *   **Constraint di Liquidità:** La size viene controllata rispetto alla liquidità reale dell'order book per evitare un impatto eccessivo sul mercato.
2.  **Esecuzione:** Viene generato un **Ordine di Esecuzione** e inviato alla **Broker API**.

#### 2.6 Gestione della Posizione Live

*   Una volta aperta, la posizione viene costantemente monitorata dall'`AdaptiveStopLossManager`.
*   Questo gestore adatta dinamicamente i livelli di Stop Loss e Take Profit in base alla volatilità e all'andamento del prezzo.
*   Quando un livello viene raggiunto, o un nuovo segnale forte invalida la logica del trade, viene inviato un ordine di chiusura al broker.

### Il Ciclo di Feedback (Feedback Loop)

La freccia tratteggiata rossa rappresenta l'elemento più importante per l'evoluzione a lungo termine del sistema. Ogni volta che un'operazione viene chiusa, il suo risultato (profitto/perdita, motivo della chiusura) viene registrato nello **Storico Trade**. Questo database alimenta il processo offline di **Ottimizzazione Adattiva**, permettendo al sistema di imparare dai propri successi e fallimenti e di migliorare continuamente le sue strategie e i suoi parametri di rischio nel tempo.