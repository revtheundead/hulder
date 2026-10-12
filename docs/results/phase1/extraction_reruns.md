# Fresh-engine captures are not bit-identical at scale (Phase 1)

Three fp16 captures of the 16,332 probe prompts, each a fresh engine with an identical `CaptureConfig`. The root cause is open (docs/design.md §10). Full numbers in `extraction_reruns.json`.

Reproduce from the repo root (each `extract.py` run writes `.cache/probe/fp16`; rename runs 1 and 3 after capture):

```
source scripts/dev-env.sh
uv run python experiments/phase1/probe/extract.py --config fp16    # x3, keeping run 2 as fp16
uv run python experiments/phase1/probe/compare_runs.py .cache/probe/fp16_run1 .cache/probe/fp16 .cache/probe/fp16_run3
```

**`fp16_run3` was deleted on 2026-10-05 to save disk.** `fp16_run1` is kept as the real `noise_floor=` example. The snapshot slice's real-data check uses `fp16` as reference, `fp16_run1` as floor and AWQ as candidate. Run 2 is the canonical `fp16` itself. Its `fp16_run2` symlink was removed on 2026-10-08, so no path can pass the reference off as an independent rerun, and the command above names `fp16` instead (the JSON then labels run 2 `fp16`). To regenerate run 3, move the canonical capture aside so `extract.py` doesn't overwrite it (about 670 MB; check free space first):

```
source scripts/dev-env.sh
df -h .
mv .cache/probe/fp16 .cache/probe/fp16_canonical
uv run python experiments/phase1/probe/extract.py --config fp16
mv .cache/probe/fp16 .cache/probe/fp16_run3
mv .cache/probe/fp16_canonical .cache/probe/fp16
```

A regenerated run 3 is a fresh engine, so it reproduces the run 3 rows below to within the noise this page reports, not bitwise.

| Pair | Prompts differing | Batches differing | Main probe mean / p99 / max \|Δscore\| | Decision flips at 0.769 |
|---|---|---|---|---|
| run 1 vs 2 | 54.9% | 1,471 / 2,042 | 8.2e-5 / 1.3e-3 / 4.1e-3 | 1 |
| run 1 vs 3 | 56.0% | 1,473 / 2,042 | 8.6e-5 / 1.4e-3 / 4.1e-3 | 1 |
| run 2 vs 3 | 52.6% | 1,389 / 2,042 | 8.1e-5 / 1.3e-3 / 3.6e-3 | 2 |

- **Not the compile cache.** Run 1 started with a cold cache. Runs 2 and 3 both had a warm cache and differ just as much.
- **Not batch size in tokens.** About 70% of batches differ at every batch token count from 192 to over 512.
- **Same order as batch composition.** Relative L2 at the last token reaches about 3e-3 at layer 20 and 6e-3 at layer 28, comparable to the batched-vs-single effect in `extraction_prototype.md`.

The extraction prototype recorded bit-identical reruns, checked on 6 prompts in one batch. That holds for that case and not for 16k prompts. A reference snapshot therefore reproduces to within a noise floor, not bitwise, and the gate's `noise_floor=` input is needed even between identical serving configs. The probe effect is small: 1 or 2 of 16,332 decisions flip at the main probe's threshold. Run 1 was captured at a pre-rebase commit that no longer exists. Run 2 (commit `eaa1356`) is the canonical capture behind the committed probe results.
