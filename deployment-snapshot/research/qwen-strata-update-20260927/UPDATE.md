# Aggiornamento selettivo Strata cache/idle

Stato: promossa localmente il 27 settembre 2026 dopo prove 4K/98K e Pi reale.
Engine SHA256: `2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77`.

Non e il rebase completo su v0.1.8. Integra il riuso conversazionale e i worker
CPU a riposo di v0.1.3 nella nostra build dual-GPU/vision, con correzioni locali
per il confine MTP e le risposte terminate. Modello e quantizzazione invariati.

## Uso previsto dopo la promozione

Avvia lo stesso BAT Desktop `1d - ISTA Strata - Server 98K.bat`, attendi
`PRONTO`, poi riapri `pi` in Ubuntu-24.04 nella tua directory di lavoro.
Profilo invariato: `local-qwen38/qwen3.8-flash-next-local`. Gli altri modelli
restano utilizzabili anche quando Strata non e avviato.

Il BAT normale avvia soltanto il server, senza prompt o test automatici.
La console distingue token nuovi, token riusati, prefill e decode. Il throughput
prefill conta soltanto i token effettivamente processati, non quelli in cache.

Contesto 98.304; output fino al contesto realmente residuo meno 8 token,
ragionamento incluso. Nessun tetto 1.024/4.096/8.192 reintrodotto. `xhigh`
resta il default, con override Pi esplicito rispettato. La compaction globale
di Pi e le impostazioni di altri modelli non vengono cambiate.

Vision CPU su richiesta, senza encoder residente a riposo. Massimo 8 immagini,
1.024 token per immagine, stessi limiti di dimensione e validazione dei file.

## Cosa viene riusato

KV target/MTP rimane sulle GPU; in RAM ci sono al massimo 6 checkpoint degli
stati ricorrenti/PLE/QSA, oltre alle identita di token/immagini. Non e una copia
illimitata di tutto il KV in RAM. Riserva cautelativa extra 1 GiB, non consumo
misurato. La cache e volatile: termina con il server e non attraversa i riavvii.

Riuso solo per prefissi compatibili: modifiche a messaggi, immagini, template,
compaction o cambio sessione possono comportare una rilettura parziale/totale.
Una sola conversazione attiva; i checkpoint non sono una cache per ogni chat.

Protezioni conservate: RAM libera 8 GiB, commit libero 4 GiB, Job 60 GiB,
GPU sotto 80 C, nessuna soglia di VRAM libera, mutex e arresto dei figli alla
chiusura. Nessuna modifica a Windows/pagefile o obbligo di chiudere Codex.

## Qualificazione e ripristino

`protocol.md` descrive la campagna separata; `qualify_runtime.py` non viene
chiamato dal BAT. Prove cache/fresh greedy bounded, non una certificazione
universale di correttezza, analisi scientifiche o qualita del modello.
La capacita allocata 98K e distinta dalla lunghezza reale dei prompt testati.

`qualify_pi.py` + `pi_request_proof.mjs` attestano l'endpoint WSL realmente
usato e lo SHA nativo. L'observer viene caricato soltanto nelle prove, non
nel normale Pi. `install-pi-cache.sh` e riservato alla promozione dopo le prove;
verifica il wrapper originale e mantiene una copia per rollback.

Il vecchio runtime F107 e i suoi asset rimangono conservati. Nessun commit,
push o nuovo rilascio GitHub e incluso in questo aggiornamento locale.

Backup avviabile del BAT nel progetto:
`run_qwen38_98k_ista_strata_server.before-cache-20260927.bat`.
Anche Desktop 1e debug usa ora lo stesso server nuovo, senza test di avvio.
Backup del wrapper WSL: `/home/costapad/.local/bin/pi.before-cache-idle-20260927.bak`.
Pesi, modelli/configurazioni Pi, applicazioni e protezioni non modificati.

Con Codex aperto, Pi reale ha restituito correttamente due codici immagine e il
valore di un file, con riuso verificato e SHA dell'endpoint effettivo. Nel run Pi
RAM libera minima 22,04 GiB; commit libero minimo 4,57 GiB, quindi il margine di
commit resta piu stretto di quello di RAM per analisi pesanti. Nessuno stop di
protezione. Cache testata fino a 34.817 token reali in un server allocato 98K;
non dichiarare testata una cronologia interamente popolata 98K o una sessione 4h.
