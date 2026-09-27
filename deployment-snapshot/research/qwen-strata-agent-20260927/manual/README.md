# ISTA / Strata Vision CPU con il Pi globale WSL2

1. Avvia `C:\Users\costa\Desktop\1d - ISTA Strata - Server 98K.bat`.
2. Attendi `PRONTO` nella finestra del server.
3. Nella directory di lavoro desiderata in Ubuntu-24.04, apri/riapri `pi`.

Il BAT finale e quello Desktop1d gia presente;1e resta la variante debug.
Il BAT normale ora usa `serve_only_guard.py`: carica il modello e avvia API/
ponte WSL SENZA calcolo17+25, immagini OCR o altre generazioni di test. Non
avvia encoder prima di una tua immagine. Dopo ogni richiesta stampa i tempi
nativi in forma `prefill ... tok/s | decode ... tok/s`, ragionamento incluso.
Le precedenti prove load/smoke98K sono conservate e controllate, non rieseguite.
Se diventano stale si arresta con un errore; non lancia prove automaticamente.
Protezioni RAM/commit/Job/temperatura e handshake READY restano attive.1e e i
comandi di qualification separati restano sperimentali, non il BAT normale.

Il comando globale seleziona **lo stesso** provider/modello storico:
`local-qwen38/qwen3.8-flash-next-local`. Nessun nuovo profilo, nessuna copia dei
pesi, nessuna nuova directory di sessioni. Il normale Pi mantiene estensioni,
skill, strumenti, impostazioni e directory di lavoro. Il BAT Desktop "Pi Vision"
preesistente apre invece un Pi isolato: non usarlo per questo collegamento.

Strata pronto viene preferito da `pi` senza opzioni. Un altro provider/modello
scelto esplicitamente viene rispettato; resume/continue non vengono dirottati
automaticamente. Quando Strata e pronto, il suo overlay viene caricato anche
con un altro modello iniziale, cosi Flash resta selezionabile nel menu di Pi.
Timeout/assenza/identita errata di Strata non bloccano piu l'apertura di Pi:
avvia normalmente il modello salvato o esplicito, senza overlay Strata.
Se Strata non e avviato, il wrapper
precedente riprende il suo comportamento: il default salvato e ThinkingCap.
Per scegliere esplicitamente ISTA/llama.cpp col vecchio BAT:
`pi --model local-qwen38/qwen3.8-flash-next-local`.
Una sessione Pi gia aperta non viene dirottata mentre sta lavorando: riaprila
dopo l'avvio del server. Non avviare i due server insieme.
Per usare altri modelli, apri semplicemente `pi` a server Strata spento e usa
il selettore modelli, oppure passa `pi --model provider/modello`. Non serve
avviare Qwen Flash. Per generare con un modello locale, il relativo server
deve naturalmente essere avviato: e distinto dall'apertura dell'interfaccia.

## Limiti del profilo sperimentale

- Contesto nativo **98.304 token**. Rimosso il tetto output1.024: il budget
  effettivo e `98304 - token del prompt completo - 8`, thinking incluso.
  Il massimo teorico richiesto e98.296; il server conta il prompt reale,
  comprese immagini/template/strumenti. Un limite inferiore esplicitamente
  richiesto dal client resta rispettato. Non e un output infinito.
- Pi usa streaming: annullare una risposta interrompe la generazione e libera
  gli embedding temporanei. L'API richiede streaming per richieste output>1024;
  questo non limita l'output streaming usato da Pi.
- Thinking normale predefinito `xhigh` (un'opzione Pi esplicita puo cambiarlo);
  riepiloghi automatici senza thinking per evitare
  l'esaurimento del budget. Il limite reale prompt+output rifiuta l'overflow,
  senza troncamenti. La riserva di compaction globale resta22304token: Pi puo
  riassumere prima di riempire tutto il contesto. Non e stata abbassata per gli
  altri modelli; il precedente test95K rimane una distinta prova isolata.
- Vision sull'encoder **CPU/8thread**, proiettore Q8 gia presente. Pi puo aprire
  immagini con `read`: ad esempio chiedi "Apri /percorso/immagine.png e descrivila".
  Massimo1024token/immagine,8immagini nella richiesta/cronologia,4MiB/immagine,
  8MiB totali compressi,16Mpixel, corpo HTTP16MiB. PNG/JPEG/WebP/BMP/GIF statici;
  immagini oltre2048px vengono ridotte prima dell'encoder. URLs e percorsi
  server-side sono rifiutati: Pi legge il file e invia l'immagine inline.
  L'encoder viene scaricato dalla RAM dopo la codifica; gli embedding restano
  in una cache limitata a8immagini. Un'immagine nuova richiede una riattivazione
  da freddo e almeno6GiB di commit/10GiB di RAM liberi (incluse le soglie4/8GiB).
  Se manca questa memoria la richiesta immagine viene rifiutata, senza avviare
  l'encoder; testo e immagini gia in cache rimangono utilizzabili.
- RAM libera>=8GiB, commit libero>=4GiB, Job<=60GiB, GPU<80C. Nessuna soglia
  di VRAM libera. La riserva700MiB e dimensionamento interno, non uno stop VRAM.
- Nessun trim, modifica Windows/pagefile, driver o chiusura di altre app.
- Durata massima4ore: arresto ordinato circa30secondi prima del watchdog rigido.
  INVIO arresta il server; chiudere la finestra termina l'intero Job posseduto.

La prova agentica95K precedente e il controllo del collegamento globale non
certificano equivalenza numerica CPU/GPU, qualita scientifica o stabilita4ore.
Il warning upstream della cache resta aperto. Verifica autonomamente le analisi.
Una carenza di commit puo ancora provocare uno stop protettivo anche con RAM
fisica libera; non interpretare Jobcommit come RAM residente.

## Verificato il27settembre

Aggiornamento output:27test Python, tokenizer/template reali con prompt95K e
payload del Pi SDK installato verificati senza generazione. Nuovi gate encoder
e4K load/OCR/text-followup passano con il codice aggiornato; nessuna modifica
alle soglie RAM/commit/Job/temperatura o ai file globali Pi. L'utente ha poi
completato98K load/smoke e due avvii manuali, con vision/textfollowup e motore/
encoder/observer exit0. Nessuna richiesta Pi compare in questi ultimi log;
generazioni>1024 e sessioni agentiche lunghe con il nuovo cap non sono provate.
Correzione successiva:8test shell isolati, discovery/overlay SDK e tre avvii
Pi reali senza prompt con cambio modello RPC passano; modelli/settings identici.
Un refresh ThinkingCap non ripristina piu il vecchio wrapper bloccante.
[Ultimo test e Pi indipendente](../../../wiki/raw/2026-09-27-strata-final-bat-independent-pi.md).

La prova manuale debug precedente dura24min/10turniPi: RAM fisica totale
peak40.75GiB, pagefile corrente0.395-0.423GiB; uso riferito abbastanza buono.
Una risposta `finish:length` a1024token prova il vecchio tetto, ora rimosso.
Questa prova non certifica uso98K pieno,4ore o tutti i precedenti rallentamenti.
[Evidenza aggiornamento](../../../wiki/raw/2026-09-27-strata-maximum-context-output-policy.md).

Prove storiche con output1024, **non accettazione per uso quotidiano**: l'utente segnala
desktop quasi inutilizzabile e picchi60GB. Nessun nuovo test pesante avviato;
il controllo dopo l'arresto trova~12GiB RAM usata e paging output0. Pagefile
attualmente~1.1GiB, picco~11.6GiB senza attribuzione certa a Strata. Serve
misurare la responsivita sotto carico prima di dichiarare il profilo adeguato.

Nuova build separata con gate nativi/componenti,11test Python e riavvii encoder;
caricamento e OCR reale di due immagini a4K e98K, poi testo normale; Pi globale
legge una terza immagine non in cache e restituisce i codici corretti insieme
al valore letto da un file. Le due richieste Pi hanno effettivamente xhigh e
output1024. Nessun nuovo profilo; `models.json`/`settings.json` identici.

La prima codifica da freddo ha richiesto circa15s di avvio+6s di encoder CPU,
prima del prefill/risposta del modello. Le immagini in cache non riavviano
l'encoder. Attendi sempre PRONTO: il BAT effettua le prove vision di avvio.
Lo snapshot del processo nativo mostra circa22.72GiB di RAM fisica, distinto
dal Jobpeak52.38GiB di commit (include encoder/servizi). WSL/Windows sono extra;
lo snapshot non e un picco di RAM complessivo. Test breve, non4ore o cronologia
vision piena98K. I valori `thinking` intermittenti della console non sono un
benchmark; i tempi nativi autentici sono in timings.jsonl.

Sono stati rilevati anche molti diff Git contemporanei di Codex. I grossi
tensori intermedi generati da Strata e gli oggetti di compilazione ora sono
ignorati da Git: restano sul disco, sorgenti e prove JSON restano visibili.
Il caricamento successivo e passato; non attribuiamo ogni picco precedente a
una causa unica. Nessun intervento sul pagefile, WSL o applicazioni estranee.

## Configurazione e ripristino

`models.json` e `settings.json` globali sono invariati. `pi_strata.mjs` sostituisce
il provider solo nella memoria del processo Pi quando il servizio autenticato
e riconosciuto. La nuova build vision e isolata in `qwen-strata-vision-20260927`;
build/config/gate testuali originali rimangono invariati, i pesi non sono copiati.
Endpoint Windows127.0.0.1:8037; ponte autenticato sulla sola interfaccia privata
WSL:8038, indirizzo rilevato ogni volta. Nessun listener LAN/all-address.

Il BAT ISTA precedente `1c - ISTA Ottimizzato - Server Vision 98K.bat` e intatto.
Per rimuovere solo l'integrazione globale, con server/Pi arrestati, in WSL:

```bash
cp -p /home/costapad/.local/bin/pi.before-ista-strata-20260927.bak /home/costapad/.local/bin/pi
```

Per tornare a Strata **solo testo**, dal terminale Windows esegui il BAT con
`--text-only` (non inviare immagini in quel caso): il wrapper globale corrente
riconosce anche la vecchia build testo. Non ripristinare il solo backup intermedio
vision senza gli adapter corrispondenti; il protocollo di rilevamento e cambiato.

I log vision sono in `research/qwen-strata-vision-20260927/runs/vision-manual-*`;
quelli del rollback testo in `research/qwen-strata-agent-20260927/runs/manual-*`.
Verifica preliminare senza caricamento: esegui il BAT con `--check` da terminale.
Questo controllo distingue componenti/risorse validi dall'ammissione98K:
`first_user_start_requires98k_checks:true` indica verifiche ancora necessarie
al primo avvio, non un'ammissione completa gia ottenuta.
L'ultimo controllo dopo il test utente riporta invece false e ammissione98K true.
Il launcher ammette esclusivamente la build e il profilo98304 con gate vision;
non ricostruisce automaticamente il motore, non scarica pesi e non ignora i gate.
Il derivato serve-only e verificato offline come copia esatta del precedente
observer meno i prompt startup, piu stampa tempi. Non attribuiamo al nuovo
percorso una prova live che non e stata eseguita:3test offline e Desktop--check0.
Backup mirati precedenti alla modifica: `backups/output-budget-20260927/`.
Backup wrapper precedente all'indipendenza: WSL
`/home/costapad/.local/bin/pi.before-independent-providers-20260927.bak`;
sorgenti precedenti in `backups/pi-independent-20260927/`.
