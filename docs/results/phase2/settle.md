# The INCONCLUSIVE estimate on fresh synthetic data

**Whether "about N more prompts give an 80% chance to settle it" holds when N more prompts are actually drawn.** It misses its bar: 14 of the 20 rows the bar applies to fall outside 70% to 90%. It holds at the calibration set's size away from the lines, and it's too optimistic at 180 examples per class and wherever the true value sits on a line. The bar was committed before the run (`settle_expected.json`, commit `82e8684`) and hasn't moved. Full output in `settle.json`.

## The rule

For each condition an INCONCLUSIVE report leaves undecided, `hulder.gate.settle` estimates the extra prompts per class that settle it with an 80% chance, if the observed values held. Each of 1,000 seeds per case is extended with that many fresh examples from the same synthetic generator, at the original threshold, and the condition is read again.

The bar: wherever a case, size and condition has at least 50 undecided seeds with a finite estimate, the share settled must lie in [70%, 90%]. The sizes are 1,191 benign and 910 attack, and 180 and 180.

## Result

| Case | Size | Condition | Seeds with an estimate | Settled | Bar |
|---|---|---|---|---|---|
| FPR at 1.5× the target | calibration | FPR | 768 | 84.6% [81.9, 87.1] | **meets** |
| FPR at 2.5× | calibration | FPR | 777 | 82.0% [79.1, 84.6] | **meets** |
| FNR rise of 0.04 | calibration | FNR | 914 | 72.1% [69.1, 75.0] | **meets** |
| FNR rise of 0.06 | calibration | FNR | 890 | 70.4% [67.3, 73.4] | **meets** |
| FPR on the line, 2× | calibration | FPR | 885 | 6.6% [5.0, 8.4] | misses |
| ΔAUROC on the line, −0.02 | calibration | ΔAUROC | 495 | 42.4% [38.0, 46.9] | misses |
| FPR at 1.5× | quickstart | FPR | 999 | 60.4% [57.3, 63.4] | misses |
| FPR at 2.5× | quickstart | FPR | 959 | 55.2% [52.0, 58.3] | misses |
| FPR on the line | quickstart | FPR | 990 | 7.3% [5.7, 9.1] | misses |
| FNR rise of 0.04 | quickstart | FNR | 882 | 47.1% [43.7, 50.4] | misses |
| FNR rise of 0.06 | quickstart | FNR | 890 | 56.4% [53.1, 59.7] | misses |
| ΔAUROC on the line | quickstart | ΔAUROC | 715 | 47.8% [44.1, 51.6] | misses |
| Null, and the other cases' unshifted FPR | quickstart | FPR | 957 to 1,000 each | 63.4% to 64.6% | misses (4 rows) |
| Null, and the ΔAUROC case's FNR | quickstart | FNR | 804, 761 | 91.8%, 91.3% | misses, too conservative |
| FPR cases' unshifted FNR | quickstart | FNR | 257, 104 | 76.3%, 77.9% | meets (2 rows) |

At the calibration set's size and off the lines, every row meets the bar, though the FNR rows sit near its 70% floor. Settling the wrong way, against the case's population side, was at most 8 seeds of 890 there. At 180 per class it reached 81 of 890 (FNR rise of 0.06).

## Why it misses

- **On a line, nothing settles.** The estimate assumes the observed value holds, and an observed value is almost never exactly on its line, so it promises 80%. When the truth is on the line, any amount of data keeps the interval straddling it, and about 7% settle by chance. The estimate can't tell such a case apart from one just off the line.
- **At 180 per class, the observed value is a poor stand-in for the truth.** The estimate treats it as known. The truth is often closer to the line than observed, so the extension needs more data than estimated. The FNR estimate also moves its own line, since the reference's FNR is re-read on the extended data.
- **Quickstart-sized nulls can't PASS at a 1% target.** With 180 benign, even zero flagged gives an FPR upper bound of 2.0%, just over the line. So every null seed is INCONCLUSIVE and needs about 250 more benign, which the estimate gets right only 65% of the time. This size is below the 299 benign that recalibration needs at 1% (`docs/design.md` §6).

## A deviation from the committed reading

The bar says each condition is read from the extended report. For FPR and FNR, the run computes the same Clopper-Pearson bounds with the gate's own functions instead of a full `check`. That's identical arithmetic, and a test confirms it matches the full report.

For ΔAUROC, the full check's bootstrap holds 1,000 weights per example, which doesn't fit in memory past about 40,000 examples. So 329 seeds at calibration size and 112 at quickstart size went unchecked. They are counted apart and left out of the settle rates above. Those are the seeds that asked for the most data, so the ΔAUROC rates may be biased in either direction.

## What this means for the report

The per-condition lines can't claim "an 80% chance" as validated. That holds only for FPR and FNR at about the calibration set's size and away from the lines. Before anything ships, there's a decision to make about the wording or the estimator.

**Sweep step C**, for reference: its calibration report would have said "FPR 1.0% to 2.5% against the 2.0% line; about 7,200 more benign prompts give an 80% chance to settle it". The old figure was 3,612, "if the observed rate held". The secondary probe would have said about 22,000, against 10,389.

## Reproduce

```bash
git worktree add .cache/settle-ab8daaae61b7 ab8daaae61b7
cd .cache/settle-ab8daaae61b7 && source scripts/dev-env.sh && OMP_NUM_THREADS=1 uv run python experiments/phase2/settle_validation.py run
```

The estimator it tested was replaced after it missed, and the script was removed later with the second one, so it runs from a worktree at `ab8daaae61b7`.

Run at commit `ab8daaae61b7`, CPU only, about 40 minutes with 5 workers, against the bar committed at `82e8684`. Each case and size is written to `.cache/settle-validation/rows.jsonl` inside that worktree as it finishes.
