# Strata originale: preparazione separata, nessuna patch

## Risultato e parametri selezionati per le prove4K

`trial-selection.json` registra8worker CPU più host, GPU1, cache reserve2560MiB,
contesto4096, RAM0/commit4/Job60GiB, stop VRAM0/0 e temperatura80C.
Il confronto completo termina con uscita0 e smoke iniziali passati:
decode27.33/36.71/36.41tok/s (media33.49), prefill295.92/297.38 su3107token.
Con19worker la media decode era26.83. È una selezione fra due configurazioni
provate, non un optimum esaustivo o una prova di stabilità128K.

Per ripetere il solo screen, usare tag nuovi; questi comandi non aprono server
né eseguono gli strumenti generati. L'ammissione è specifica anche ai worker.

```powershell
& research/qwen-strata-20260926/.venv/Scripts/python.exe research/qwen-strata-stock-20260926/stock_guard.py load --profile experimental-no-vram-floor --vram-reserve-mib 2560 --pool-workers 8 --gpu 1 --tag cpu8-verify-load01 --timeout 600
& research/qwen-strata-20260926/.venv/Scripts/python.exe research/qwen-strata-stock-20260926/stock_guard.py bench --profile experimental-no-vram-floor --vram-reserve-mib 2560 --pool-workers 8 --gpu 1 --tag cpu8-verify-bench01 --timeout 900
```

Il binario/source originali sono invariati.14test policy,3frontend e18gate
del sorgente passano; questi ultimi non provano equivalenza della release.
La cache GPU upstream dichiara un difetto di correttezza e uno dei3testi
greedy diverge nel confronto CPU: **nessuna ammissione agentica/Pi/produzione**.
I modelli delle prove sono stati chiusi; i launcher ordinari restano intatti.

## Prova sperimentale autorizzata September26

Con successiva conferma dell'utente, `experimental-no-vram-floor` elimina
gli stop startup/runtime sulla VRAM libera, lasciando temperatura80C,
RAM0/commit4/Job60GiB, timeout, mutex e kill-on-close. Il margine aggiuntivo
3072MiB della cache torna al default upstream700MiB per i buffer CUDA:
non è una soglia di arresto del guardian. WDDM può comunque paginare o una
allocazione CUDA può fallire; nessuna promessa di stabilità o velocità.
Le ammissioni del vecchio profilo non valgono per questa nuova configurazione.
Il primo load con700MiB raggiunge READY ma si ferma a3.53GiB commit, senza
generare. Il dimensionamento cache è quindi ottimizzabile con l'opzione
upstream `--vram-reserve-mib` del guardian (700/1024/1536/2048/2560/3072),
solo in questo profilo e con nuovo load-only per ogni valore. Nessun valore
reintroduce uno stop GPU: è un budget di cache, non una soglia del monitor.

L'utente ha confermato `experimental-memory`: contesto4096, riserva RAM0GiB,
stop commit4GiB e limite Job60GiB. Il profilo ordinario conserva12/16/48.
Temperatura80C, freeVRAM2GiB/device, timeout, mutex e kill-on-close restano.
La memoria può comunque esaurirsi: possibili paging, forte rallentamento,
errore nativo e perdita della richiesta. Nessun cambiamento Windows/pagefile,
chiusura app, sostituzione quantizzazione, launcher ISTA o Pi.

`stock_guard.py` e `native_probe.py` sono osservatori esterni: usano il
binario originale e il suo frontend, senza modificare upstream e senza aprire
un listener. La singola GPU1 evita l'occupazione del display; CPU scheduling
resta quello originale. L'opzione upstream `--vram-reserve-mib 3072` mantiene
la soglia GPU2GiB: è una differenza di configurazione registrata, non il
default700MiB dell'autore. Pack/tokenizer/MTP sono riusati e verificati.

I gate numerici vengono ricompilati dal sorgente intatto con CUDA13.3 e
registrati come **evidenza sul sorgente**, non test interni al binario ufficiale
CUDA13.0. Prima delle misure servono gate disponibili, caricamento4K con
uscita pulita e smoke semantico/tool. L'osservatore legge i timing nativi
`DONE`, conserva ogni richiesta/risposta e non esegue comandi generati.

```powershell
& research/qwen-strata-20260926/.venv/Scripts/python.exe research/qwen-strata-stock-20260926/check_stock.py --allow-experimental-memory
& research/qwen-strata-20260926/.venv/Scripts/python.exe research/qwen-strata-stock-20260926/stock_guard.py load --profile experimental-memory --gpu 1 --tag stock-load4k --timeout 600
```

Il nuovo opt-in VRAM usa `--profile experimental-no-vram-floor`; il preflight
isolato usa insieme `--allow-experimental-memory --allow-no-gpu-floor`.
La prova successiva usa `--vram-reserve-mib 1536`; load e bench devono usare
lo stesso valore. Tutti gli argomenti effettivi e le risorse sono registrati.
Il binario ufficiale ha completato il caricamento4K con uscita0 nel profilo
precedente, dopo18 gate numerici del sorgente e14 test di confine esterni.
Il load-only con riserva3072MiB è passato in36.875s; nessuna velocità di
generazione è implicita nel caricamento. Le evidenze stanno nei singoli run.

**Avvertenza qualità upstream:** il motore originale avvisa esplicitamente
che il percorso GPU degli hit della cache diverge dal controllo senza cache.
I timing sono misure reali, ma uno smoke riuscito non dimostra equivalenza
numerica o qualità agentica. Non promuovere queste prove a produzione.

Le ammissioni sono locali e specifiche al profilo; questo non promuove un
server produttivo o contesti maggiori. Le righe sotto conservano lo stato
storico della preparazione precedente; consultare la wiki per le prove nuove.

Sorgente ufficiale scaricato da Niko1221/Strata, revisione
6da1f667e86558b152ab128edf3ebf77a80a9e57, senza modifiche tracciate.
Binario ufficiale Windows v0.1.2, archivio verificato contro il digest
SHA256 pubblicato dall'API GitHub:
34ad72b75836ce737a5acd0b2cceee0c03a4b1b4420d286f939705b9bb8e60be.
L'eseguibile originale resta byte-identico alla release; non è la build custom.

Il binario supporta sm120 ed è compilato CUDA13.0/AVX2 portable. Il suo
`--help` parte con le librerie CUDA13.3 già presenti. Il setup ufficiale
`--check` riconosce RTX5060Ti/64GB/Intel265KF AVX2 e dichiara IQ3 compatibile
in base alla RAM **totale**, non alla disponibilità attuale. Non è una prova
di caricamento o inferenza. I log sono accanto a questo file.

## Configurazione predisposta, non avviata

`stock-config.json` usa gli argomenti del setup ufficiale: cache auto,
prefill2048, MTP/spec4/min-p0.5, contesto4096, KVfp16 di default,
adattamento e quota PCIe di default. Riusa i GGUF ISTA verificati, il pack
e MTP già preparati con gli strumenti originali: nessun nuovo checkpoint.
I file originali iq_pack.py/mtp_pack.py/mtp_rt.py del precedente checkout
sono invariati. Nessuna opzione dual-GPU, CPU hot set o fallback SSD custom.
I percorsi e la porta8037 sono configurazione esterna, non patch del motore.
Il test futuro deve rendere visibile una sola GPU e restare su loopback,
con autenticazione da ambiente, limiti e monitor esterni.

**Non avviare direttamente il server o START-HERE.bat per aggirare le prove.**
Non c'è un'ammissione runtime e non sono ancora state completate le prove
sintetiche della release. Il download non autorizza un caricamento insicuro.

## Controllo di memoria senza inferenza

```powershell
& research/qwen-strata-20260926/.venv/Scripts/python.exe research/qwen-strata-stock-20260926/check_stock.py
```

Lo script è fuori dal sorgente ufficiale. Verifica release/source e lo stato
dei GGUF contro la precedente prova SHA256; non avvia nulla e non concede
ammissioni. Gli esperti occupano42912972800byte, con2329600byte extra di
scratch nell'arena originale:39,968GiB. Con la riserva RAM12GiB serve almeno
51,968GiB disponibili **prima** delle altre allocazioni. Per il commit il
minimo inferiore è55,968GiB, prima del backing GPU/driver e delle altre
allocazioni. Questi minimi non sono una stima del consumo totale.

Se il controllo fallisce, prima occorrono più risorse libere o una decisione
esplicita dell'utente sul profilo di rischio. Non abbassare automaticamente
le soglie, non chiudere applicazioni, non cambiare Windows/pagefile e non
sostituire quantizzazione. Il launcher ISTA, il prototipo e Pi restano intatti.
Lo stato READY è verificato dal guardian prima che l'osservatore possa
generare o uscire dal load-only; la sola uscita0 non prova i margini GPU.
