# ISTA Strata — debug del PC, separato dal launcher normale

BAT Desktop: `1e - ISTA Strata - Debug 98K.bat`.

1. Se Windows e ancora bloccato, salva il possibile e riavvialo normalmente.
2. Chiudi Codex, senza avviare anche il vecchio BAT/server.
3. Avvia il BAT **Debug**. Registra15s di baseline, poi avvia lo stesso guardian
   vision98K con gli stessi gate/protezioni del BAT1d. Non e un altro modello.
4. Attendi PRONTO e apri una nuova sessione `pi` in WSL2, nel solito progetto.
5. Parti con richieste brevi; annota l'ora se Windows diventa poco reattivo.
   INVIO nella finestra server arresta il modello; chiudere la finestra termina
   il Job posseduto. Il monitor registra ancora circa20s, poi si chiude da solo.

Il Pi globale/provider/modello non viene modificato. RAMfree8GiB, commitfree4GiB,
Job60GiB, GPU80C, nessuna soglia di VRAM libera: protezioni identiche. Il debug
non rimuove gate, non cambia contesto/quant/cache/CPU, non modifica Windows,
pagefile, driver o altre app. Se il monitor muore durante il lavoro, il launcher
chiede l'arresto del proprio guardian; non termina processi altrui.

## Cosa registra

Timeline JSONL: RAM fisica totale/usata/disponibile, commit usato/limite,
pagefile occupato/picco, rate di paging, CPU, I/O e coda/latency disco C:, VRAM/
utilizzo/temperatura delle due GPU. Campioni base5s circa, GPU10s, processi15s.
Top processi per memoria, gruppi Strata/encoder/WSL/Codex/Git e memoria del monitor
stesso; fase del guardian per correlare caricamento/vision/PRONTO/arresto.

Nomi/PID e contatori soltanto: nessuna commandline, chiave API, prompt o immagine
copiata dal monitor. I log originali del server rimangono nella propria cartella,
con la politica preesistente: questo monitor non li trascrive nel suo JSONL.

Il monitor e nativo Windows, non lancia PowerShell/CIM a ogni campione; unica
query esterna periodica e nvidia-smi con timeout. E a priorita ridotta e fuori
dal Job del modello, per poter vedere il rilascio di RAM dopo l'arresto.
Durata finita4h+baseline/coda (max14520s), log campioni max64MiB, strutture e
letture limitate. Campioni indisponibili/incompleti vanno interpretati come tali,
non come consumo zero. Un blocco improvviso puo lasciare solo una timeline
parziale; ogni riga completata viene flushata.

## Dove trovare i risultati

Ogni avvio crea una cartella nuova sotto:
`research/qwen-strata-debug-20260927/runs/debug-AAAAmmgg-HHMMSS-xxxxxx/`.
Il percorso completo viene mostrato nella finestra prima di avviare il modello.

- `samples.jsonl`: evoluzione nel tempo, utilizzabile anche se non termina.
- `summary.json`: riepilogo del monitor, min/max e disponibilita dei contatori.
- `manifest.json`: identita/configurazione, intervalli e collegamento al run server.
- `launcher-result.json`: esito/uscita, contatori REALI del Job dal guardian.
- Run guardian corrispondente: `research/qwen-strata-vision-20260927/runs/debug-*`.

Il monitor NON dimostra la causa del blocco, la qualita del modello o la vita
residua SSD. Working-set e commit non vanno sottratti per calcolare lo swap;
somma working-set dei processi non e RAM fisica unica. Paging e disco sono
globali: possono includere altre app; hard-fault reads possono essere file
mappati e non pagefile. Rate campionate non sono TBW o scritture NAND misurate.

## Controlli senza caricare il modello

Da un terminale Windows nella directory del progetto:

```bat
run_qwen38_98k_ista_strata_debug.bat --check
run_qwen38_98k_ista_strata_debug.bat --monitor-only --seconds 30
```

--check registra i contatori e invoca soltanto i gate/resources del guardian.
--monitor-only registra Windows/GPU senza invocare il guardian o il modello.
Il doppio clic normale sul BAT invece AVVIA il modello, come il launcher1d.
Non usare --monitor-only per inferire il comportamento del modello sotto carico.

Il BAT normale1d rimane intatto. Questo e uno strumento diagnostico, non un nuovo
profilo ottimizzato ne una dichiarazione di accettazione per l'uso quotidiano.

## Verificato il27settembre

28test offline passati (18monitor+10launcher). BAT Desktop reale --check:
gate/resources pass, nessun modello caricato; monitor6campioni/26.5s,13contatori
PDH disponibili dopo il primo campione, entrambe RTX5060Ti, nessun errore sensore.
RAM del solo monitor circa25.9MiB, lettura completa massima78ms nel campione
idle osservato; overhead sotto carico non ancora misurato.

Prova reale su un piccolo processo creato dal test: chiusura del redirector
Python verificata anche sull'interprete figlio; monitor continua20.922s,
11campioni, summary salvato/exit0. PIDredirector e PIDPython possono differire:
handshake ammette soltanto il processo avviato o il suo figlio dichiarato.
Le prime prove di identita/timing fallite sono conservate, non sovrascritte.

Processi protetti/inaccessibili espliciti (nel controllo377enumerati/193letti),
GPU/processi in cache hanno l'eta del campione. SharedRAM WDDM della GPU non
campionata e indicata indisponibile. Il picco pagefile somma picchi per-file,
non necessariamente contemporanei. BAT1d/gate vision stessi SHA256 iniziali;
nessun server/debug residuo alla fine della verifica.
