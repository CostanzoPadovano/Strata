# Strata dual-GPU per questo PC

Runtime sperimentale separato da ISTA/llama.cpp. Non modifica il launcher sul
Desktop né il profilo globale di Pi. Non è una promessa di65token/s.

## Architettura

- Core Ultra7 265KF: AVX2,8worker, affinità del Job a12core.
- RTX5060Ti GPU0: proiezioni, MTP, KV INT8, cache esperti≤2304×blob massimo
  (circa5GiB,2871slot a dimensione variabile nel candidato corrente).
- RTX5060Ti GPU1: strati esperti0–15,11,43GiB, senza copia degli stessi pesi
  in RAM. Finestre di calcolo mediante grafi CUDA e buffer fissi.
- RAM: fino a14GiB di esperti scelti dal profilo, escludendo i pesi delle due
  GPU. Circa9,54GiB non sono residenti in RAM/VRAM; il fallback SSD comprende
  anche i5GiB della cache GPU0 quando questa viene prestata al prefill.
  Letture dirette senza mmap/cache del sistema operativo, finestra fredda
  di80slot e una sola lettura in volo.
- PLE su SSD; MTP Q2_0 derivato da revisione ufficiale fissata. IQ3_XXS
  originale riusato, nessun nuovo download del checkpoint completo.

## Avvio

Dal progetto, `run_qwen38_strata_server.bat` avvia il server su loopback8035.
Poi `run_qwen38_strata_pi.bat` apre Pi in WSL2 tramite ponte autenticato8036,
con provider dedicato e `xhigh` nativo. Il server rifiuta i contesti senza un
record di ammissione corrispondente alla build **e** al manifest corrente.
Se32K non è ancora ammesso, il BAT non aggira il controllo.

Chiudere la finestra del server o premere Ctrl+C libera il Job e i processi
figli. Il vecchio launcher rimane disponibile e non è stato sostituito.
I log e i risultati sono in `runs/<tag>/`; `failure.json` registra gli arresti
della nuova revisione del guardian.

## Limiti e verifiche

Job48GiB; RAM disponibile≥12GiB, commit disponibile≥16GiB, VRAM libera≥2GiB
per GPU, temperatura<80°C. Nessun trim o cambio di pagefile/Windows.
Preflight conservativo: overhead di commit stimato24GiB oltre agli esperti
RAM, più16GiB di riserva. È una stima: contano comunque i campioni reali.

Le prove sintetiche comprendono35confronti GPU0/GPU1 bit-identici,7formati,
oracolo float indipendente,80esperti distinti, letture SSD/cancellazione,
36test componenti su entrambe le GPU,10template e7controlli HTTP/tool/xhigh,
oltre a3test del guardian. `gates.json` è valido solo per l'eseguibile indicato.

Il primo candidato20GiB ha passato caricamento e8richieste reali a4K,
inclusi tool e follow-up. Misura preliminare: decode7,97–8,19token/s;
prefill~107–109token/s su3107token. RAM minima25,38GiB e commit minimo20,61GiB
disponibili. Questi sono risultati del candidato **senza** grafi GPU1, non
della revisione successiva e non un confronto appaiato con llama.cpp.

Il candidato corrente14GiB RAM/cache5GiB passa gli8smoke a4K ma non migliora
la velocità:7,48/7,62/8,18token/s su3uscite da256token;
prefill100,87/100,31token/s su3107token. RAM minima30,12GiB e commit minimo
19,81GiB disponibili. `admission-4096.json` è solo un'ammissione operativa
sperimentale, non un'approvazione della qualità o di un aumento di velocità.

Il successivo test32K (`runs/smoke32k-pi`) si arresta a15,68GiB di commit
disponibile durante il primo prefill da3107token, prima della prova reale
da30Ktoken e del ciclo agentico Pi. Il Job viene chiuso e la VRAM liberata.
**32K non è ammesso; il BAT server predefinito rimane quindi bloccato.**
Il collegamento HTTP da WSL è stato verificato separatamente, non equivale
alla validazione di Pi end-to-end.

Il benchmark pubblicato da Strata usa anch'esso ISTA GSQ-RCO, non Unsloth.
Questa variante non è stata ancora provata con Unsloth; i vecchi test
Unsloth/llama.cpp non sono un controllo per questo motore.

Il diagnostico non invasivo `profile_io.py` osserva38,56GB di letture del
processo nella richiesta42prompt/256outputtoken, a7,35token/s decode;
su3107prompttoken osserva24,96GB di letture e101,90token/s prefill.
I contatori includono PLE e tutte le letture del processo, non isolano
latenza SSD né il solo decode. Questa è un'evidenza di traffico elevato,
non la quantificazione definitiva di ciascun collo di bottiglia.

Restano distinte la correttezza dei kernel, la qualità del modello e la
stabilità dei contesti lunghi. Il server è solo testo, greedy, una richiesta
di inferenza alla volta; ricostruisce il prompt a ogni turno (nessun riuso
KV/prefix fra richieste). L'avviso upstream sul percorso cache rimane visibile:
il percorso nativo usa kernel diversi e passa l'oracolo sintetico, ma manca
ancora un confronto numerico completo cache-on/off e contro llama.cpp/BF16.

Procedura e prove fallite conservate: [protocol.md](protocol.md).
Build: `build.ps1`; verifiche: `.venv/Scripts/python.exe prepare.py validate`;
binding: `prepare.py bind`. Non usare il `setup.py` upstream per sostituire
questo profilo locale e i suoi limiti.
