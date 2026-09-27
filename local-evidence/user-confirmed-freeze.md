# User-confirmed operational baseline, 2026-09-27

After the qualified cache/idle update was handed off for personal testing, the
owner reports that it works very well, including Pi `/compact`. The owner
explicitly asks to stop further changes and save the working setup on GitHub.

Evidence kind: user report, not a newly instrumented experiment. The report does
not establish a measured context length, duration, throughput, physical-RAM
peak, SSD paging volume or scientific/model-quality score.

Frozen native identity:
`2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77`.
No runtime/source/BAT/Pi/model/guard/OS change, build or inference test was made
for publication. Only dedicated publication copies, checksum verification,
documentation and GitHub history/release assets change.

Runtime asset: `windows-cache-runtime.zip`, 137,230,948 bytes, 562 exact file
members plus manifest, SHA256
`a49438dcdbe0e8a38bb109fb363945693da3427e3bfdef334319ad52e89b1164`.
The prior F107 release is retained for rollback. No model weights, private
conversations, authentication or global Pi settings are published.

Archive preflight initially stopped before compression because the existing
Pi timing log also contained three earlier helper-fixture attempts. The exporter
was corrected to select the two requests correlated in the successful immutable
fixture02 attestation, not silently count all five log rows as successful work.
Original evidence/failure records remain unchanged; no new inference was run.

The first generated package is retained locally, not published. A focused
read-only archive review found that generic size/model-directory exclusions
also omitted the 90,725,376-byte official vision encoder and six synthetic
MTP qualification proof files. V2 uses narrow actual-model-path exclusions and
an exact path/size/hash exception for the encoder; all are now included.
The old source baseline manifest is retained without overwrite. Nested archived
Git attributes intentionally restore BATs with CRLF; source verification checks
exact checkout bytes rather than falsely equating them to normalized Git blobs.
Original patch context whitespace and historical EOFs remain byte-exact.

Future upstream updates are separate qualified changes after user direction.
No automatic update or recurring monitoring has been enabled.
