# The second INCONCLUSIVE estimate: a calibration check on fresh data

**Whether the chance a report states, that a given extension of the data settles an undecided condition, matches how often it does.** It doesn't. The estimator misses its bar in 4 of the 8 calibration bins and in 3 of the 12 wrong-way rows. The bar was committed before the run (`settle_v2_expected.json`, commit `b78cc65`), on seeds the first run never used, and hasn't moved. Full output in `settle_v2.json`.

## The estimator and the bar

For each condition an INCONCLUSIVE report leaves undecided, `hulder.gate.settle` at `b78cc65` stated the smallest extension, up to ten times the data, with at least an 80% chance to settle it, or the chance at ten times. The chance is averaged over the true values the data still allows:
- a Jeffreys posterior for FPR;
- a Dirichlet posterior on the paired attack outcomes for FNR;
- the bootstrap interval read as normal for ΔAUROC.

Each seed is extended exactly as stated with fresh synthetic examples at the original threshold, and the condition is read again. There are ten cases, three of them with the truth on a line, at two sizes:
- 1,191 benign and 910 attack at a 1% target;
- 180 per class at the quickstart's 5%.

The bar has two parts:
- **Calibration.** Per size, pooling all cases, the stated chances are binned in fifths, and the share settled must be within 10 points of each bin's mean stated chance.
- **Wrong way.** At most 1% may settle on the wrong side, per case, size and condition with a population side.

## Result

**Calibration, every case pooled per size**

| Size | Stated chance | Seeds | Mean stated | Settled | Bar |
|---|---|---|---|---|---|
| 1,191 / 910 at 1% | 20 to 40% | 986 | 30.4% | 23.2% | meets |
| | 40 to 60% | 2,243 | 53.0% | 46.4% | meets |
| | 60 to 80% | 1,897 | 69.8% | 62.3% | meets |
| | 80 to 100% | 1,747 | 80.3% | 63.0% | **misses** |
| 180 / 180 at 5% | 20 to 40% | 262 | 37.3% | 32.4% | meets |
| | 40 to 60% | 2,988 | 51.3% | 39.6% | **misses** |
| | 60 to 80% | 2,641 | 69.0% | 55.3% | **misses** |
| | 80 to 100% | 1,917 | 80.8% | 62.8% | **misses** |

**Wrong way.** It held at 0% to 0.4% on 9 rows and missed on 3, all at 180 per class:
- FNR rise of 0.04: 44 of 373, 11.8%;
- FNR rise of 0.06: 36 of 388, 9.3%;
- ΔAUROC at −0.025: 51 of 395, 12.9%.

No ΔAUROC extension hit the 40,000-example memory cap, since ten times either size stays under it. PASS was never out of reach at these sizes.

## Why it misses

**The pooled bins average two opposite errors.** Each case's own stated chance against its realised rate, pooled over its bins:

| Case | Size | Mean stated | Settled |
|---|---|---|---|
| FPR at 1.5× the target | 1,191 | 69% | 94% |
| FPR at 2.5× | 1,191 | 67% | 89% |
| ΔAUROC at −0.025 | 1,191 | 69% | 90% |
| ΔAUROC at −0.01 | 180 | 71% | 91% |
| FPR on the line | 1,191 | 62% | 4% |
| FNR on the line | 1,191 | 48% | 15% |
| ΔAUROC on the line | 1,191 | 65% | 43% |

- **Off the line,** the estimator is now too cautious. The posterior spreads over truths near the line that these cases don't have.
- **On the line,** it's overconfident, as any estimator must be. The data can't tell a truth on the line from one just off it, and on the line nothing settles.
- **So whether a pooled bin passes depends on the case mix,** how many cases sit on a line against how many sit off it, not on the estimator alone. A chance averaged over plausible truths is calibrated only for truths distributed the way the prior assumes, and these cases are placed deliberately at and near the lines. This run can't say how real serving changes are distributed, so a bar on pooled calibration can't be met or failed on the estimator's merits.
- **The wrong-way misses at 180 per class come from the gate's rules, not the estimator.**
  - The FNR condition compares the candidate's bound with the reference's point FNR, which is itself noisy at 180 attacks, so an extended check can land on the wrong side (`docs/design.md` §10, where a paired FNR condition is planned).
  - ΔAUROC at −0.025 settled the wrong way 12.9% of the time at 180 per class, from the bootstrap interval at that size.
  - Any extension estimate inherits these.

## What this means

Neither estimator can back a stated chance:
- the first, plugging in the observed value, was too optimistic at small sizes and on the lines (`settle.md`);
- this one is too cautious off the lines and still overconfident on them.

**What the data does support:**
- **Off the lines, the extensions are generous.** About 90% of seeds settled with the stated extension, against stated chances near 70%.
- **The extension needed grows sharply near a line,** and when even ten times the data isn't enough, that is a real sign the truth is close to the line or the sample is small.
- **The exact size check is right by construction.** It says when PASS is out of reach at the current size even with no errors, and the smallest size that could clear it.

## Reproduce

```bash
git worktree add .cache/settle-b78cc65 b78cc65
cd .cache/settle-b78cc65 && source scripts/dev-env.sh && OMP_NUM_THREADS=1 uv run python experiments/phase2/settle_validation.py run
```

The estimator it tested left the library after it missed, and the script went with it, so it runs from a worktree at `b78cc65`.

Run at commit `b78cc65` on a clean tree, CPU only, about an hour with 5 workers, against the bar committed at the same commit. Each case and size is written to `.cache/settle-validation-v2/rows.jsonl` inside that worktree as it finishes.
