# Isolated Strata cache/idle update, 2026-09-27

User authorizes the update. Keep F107/server/BAT/Pi untouched until this candidate
passes its own gates; keep Codex open and avoid builds/packaging/Git scans during
model measurements. Global Windows pressure is not automatically a Strata defect.

This is a selective cache/idle backport from upstream v0.1.3
`54cf7b6f0700f4442ffa8bc8b5af257d5db91cf4` onto our v0.1.2 dual-GPU/vision
fork. Latest checked release is v0.1.8 `89ba2dc54daf3210b052e542605445a55953a1b4`;
do not call this a complete v0.1.8 upgrade. No host-KV streaming, Q4 KV,
sampling or prompt-lookup changes. No model downloads, OS/pagefile changes,
app closures or GitHub writes.

Preserve exact GPU0 cache3500/host dedup, GPU1 selected12, MTP0/spec4/min_p0.5,
prefill512/no borrowing, CPU8/pin16384MiB, bounded direct PLE reads and CPU vision.
Hard limits remain RAMfree8GiB/commitfree4GiB/Job60GiB/GPU<80C/VRAMfloors0,
exclusive mutex, verified kill-on-close, READY check and finite timeouts.

Conversation cache: at most6 running-state checkpoints in host RAM; target/MTP
positional KV stays on GPU. Reserve1GiB additional RAM/commit for six checkpoints,
one transient seventh, token/image identities and residuals. This is a conservative
reservation, not a measured consumption. Cache-disabled control remains available.
Short-read is initially0, interval16384. Save/repair MTP boundary residuals, fill
complete prompt draft KV in cache mode and catch up terminal committed rows.
Image identity includes validated pixel embeddings/MRoPE, not just pad IDs.
CPU phases wait for all workers' epoch completion before reusing job pointers.

Admission order: fresh bound build; synthetic cache/idle/vision/transport/loader;
fresh MTP/selected12 numerical tests; Python + WSL Pi routing/overlay offline;
encoder-only Job4GiB; new4K load-only, vision/text smoke and cache versus fresh
continuation/branch/cancel checks; then new98304 allocation/load/smoke and bounded
long-to-short cache checks. Old intermediate context-shape evidence is retained
only for unchanged KV allocation, never as proof of new cache semantics. New
98K capacity is not a populated98K-history or unlimited-output quality certificate.

Only after all these pass may normal BAT use the candidate. It must still start
only the server, no test prompts, and report actual fresh-prefill/decode/reused
tokens. Pi keeps the same provider/model and remains independent for other models.
No old proof may be rebound or relabeled. Stop on mismatch or resource-floor stop.
