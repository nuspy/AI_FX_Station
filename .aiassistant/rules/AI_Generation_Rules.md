---
apply: always
---

Obiettivo
Massimizza il numero di task completati per iterazione. Minimizza tutto ciò che non è codice eseguibile o diff utile.

Regole dure di output

Niente preamboli, niente stampa dell´analisi, niente scuse, niente spiegazioni lunghe.

Non chiedere conferme: assumi in modo conservativo; se qualcosa è davvero ambiguo, aggiungi un TODO breve e procedi.

Completa quanti più task possibile finché non raggiungi il limite di contesto.

Priorità assoluta al codice completo dei file. Commenti sintetici e in inglese SOLO inline nel codice dove servono.

Aggiorna sempre README.md (in Italiano) ma in forma ultra-breve (max 8–12 righe).

Aggiorna sempre pyproject.toml quando aggiungi librerie, vincoli Python 3.12, preferisci >=x.y,<x+1.0.

Zero testo ornamentale. Nessun blocco “spiegazione” al di fuori del mini-riepilogo finale.

**Politica di continuità**  
- Non attendere “continua”. Prosegui internamente finché possibile.  
- Se lo spazio non basta, chiudi con una singola riga:  
  `CONTINUA_NECESSARIA: <breve contesto e lista dei prossimi file/step in ordine>`

**Standard & Vincoli**  
- Target: **Python 3.12**.  
- Commenti nel codice: **sintetici, in inglese**.  
- `README.md`: in **Italiano**, conciso.  
- Nessun testo fuori dallo schema indicato, eccetto il riepilogo finale e (se presenti) `[SHELL]` e `CONTINUA_NECESSARIA`.
