# Isolated cache update: preflight recovery and EOS reuse rejection

This dated evidence supplements the earlier component milestone; it does not
supersede the working F107 runtime or constitute a release admission.

- Candidate before EOS fix: native SHA256
  `7d406a5478968bd346b67edc580df379a3cbcc20b432d7d489b384b69439707d`.
- `update-4k-load01` stopped before native launch: a generated Python `.pyc`
  was included in the source manifest. The manifest now excludes bytecode;
  fresh `cache-idle03` component/numerical qualification passes.
- Fresh Python/serve-only proofs, `update-encoder02`, `update-4k-load02`, and
  `update-4k-smoke01` pass, including both image codes and text after images.
- `update-4k-cache01` exits 1, with native clean exit 0. Checkpoint-branch and
  live-after-length token comparisons pass. Checkpoint reuse is 3066/3073;
  measured native prefill times are 12747 ms cold versus 301 ms warm.
- EOS follow-up outputs are exactly equal (same output-token SHA256), but
  warm reuse is only 20 rather than the live prefix expected by the probe.
  This is a reuse rejection, not an output mismatch or a RAM crash.
- Diagnosis: accepted speculative input rows were committed before visible
  emission was capped at EOS/output budget. Hidden post-terminal rows made
  the live prefix incompatible with a continuation of the visible reply.
  The verifier explicitly supports committing a shorter prefix, including
  GDN/PLE/QSA state. The candidate now bounds its terminal commit before
  publishing live state; fresh qualification is required after rebuilding.
- During this finite run Codex remained open: one targeted process snapshot
  found 11 ChatGPT GUI processes (2.18 GiB summed working set, 1.74 GiB private),
  one codex process (0.32/0.24 GiB), and no Git process. This is one sample,
  not a GUI peak or a causal memory attribution.

Evidence is under `research/qwen-strata-update-20260927/runs/` with the tags
above. Pre-fix gate/manifests/patch copies are preserved in
`gate-runs/cache-idle03/qualification-before-eos-fix/`. No runtime promotion,
global Pi change, normal BAT change, OS/pagefile edit, commit, or push occurred.
