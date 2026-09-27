Frozen working cache/idle + vision snapshot for this Windows PC: Core Ultra 7
265KF, 2 x RTX 5060 Ti 16 GB, 64 GiB DDR5-4800, Pi 0.87.1 in WSL2.

The owner confirms successful normal use **including `/compact`**. This is a
user report, not a new benchmark. No further runtime tuning, inference tests,
build, BAT/Pi changes or OS/pagefile changes were performed for publication.

Exact native SHA256:
`2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77`.
Selective v0.1.3 conversation cache/idle workers on the qualified dual-GPU/vision
v0.1.2-derived fork; **not full v0.1.8**. 98,304 capacity, remaining-context
output, on-demand vision, independent global Pi and existing guards retained.

Previously completed fresh gates and actual Pi evidence are documented in
[CACHE-IDLE.md](https://github.com/CostanzoPadovano/Strata/blob/local-dual5060ti-cache-idle-20260927/CACHE-IDLE.md).
Finite cache rewind: 16,384 reused of 16,417, identical output, 66,861→715 ms
prefill. Longest populated prompt 34,817; not organic 98K/4-hour certification.
Two warm actual Pi requests decoded at 39.93/40.44 tok/s, not a paired benchmark.

Source/config/tests, numeric-only evidence and recovery guide are pinned by the
tag. `windows-cache-runtime.zip` preserves exact engine/test binaries and selected
proof closure with checksums. **No model weights, credentials or conversations.**
Read RESTORE.md before recovery; archive verification is not a fresh-PC runtime
test. The prior `local-dual5060ti-20260927` release remains intact for rollback.
Experimental pre-release status reflects portability/quality limits, not an
observed failure in the owner's successful trial.
