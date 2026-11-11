# Coverage of a fourth paired interval: Newcombe's with continuity correction

**It misses the bar in 2 of 1,536 cell-ends, so the paired FPR and FNR conditions don't ship, and the work on them closes here.** This candidate was chosen after seeing the first coverage run's cells (`paired_coverage.md`), where Newcombe's interval without continuity correction missed in 5 cell-ends. It was the only candidate in this run. The bar is the first run's, unchanged, committed before the run (`paired_coverage_cc_expected.json`, commit `2c6e700`), on seeds 9,000 to 9,999, which no other run used. Full output in `paired_coverage_cc.json`.

## The candidate and the bar

Newcombe's square-and-add interval for a paired difference, built on Wilson intervals with continuity correction instead of without, and the same correction for the two runs' correlation (`experiments/phase2/paired_intervals.py`, `newcombe_cc`). The Wilson intervals match scipy's `wilsoncc`. That this is the paper's paired method 11 is my reading, not checked against its text.

The tables and the bar are those of `paired_coverage.md`. There are 768 cells at n = 180, 910, 1,050 and 1,191, with 10,000 tables each. In every cell, each end may miss the true change in at most 3.5% of tables, and over all cells each end's mean miss rate may be at most 2.75%.

## Result

| Interval | Mean miss, low / high | Cell-ends over 3.5% | Worst cell-end | Mean width | Meets the bar |
|---|---|---|---|---|---|
| Newcombe's with continuity correction | 1.08% / 1.09% | 2 of 1,536 | 3.68% | 3.01 points | No |
| Newcombe's, first run | 1.32% / 1.32% | 5 of 1,536 | 4.36% | 2.84 points | No |

The two cell-ends over the bar are mirror images, at n = 1,191 with 15% flagged by both:

| Cell | End | Miss |
|---|---|---|
| a fall of 2 points | low | 3.68% |
| a rise of 2 points | high | 3.64% |

The second one, the upper bound below a real rise, is on the PASS side.

- **Both are within about 1 standard error of the bar,** which is about 0.18 points there. **They still miss it, and the bar doesn't move.**
- **They sit in the same corner as the first Newcombe's failures:** many examples flagged by both runs, a one-way change of 0.75 to 2 points, and n over 1,000. So this looks like a property of the interval's correlation correction rather than chance, though this run can't separate the two.
- **10 more cell-ends sit between 3% and 3.5%.** Elsewhere the interval is conservative, with a mean miss of about 1.1% per end against a nominal 2.5%.

## The low-discordance cells at n = 180

These are the cells with a total discordant rate of 0.5% to 1%, as in the first run. The ranges are over 1%, 5% and 15% flagged by both. Every cell is in `paired_coverage_cc.json` under `low_discordance_n180`.

| Candidate only | Reference only | No discordant | Misses, low / high |
|---|---|---|---|
| 0.5% | 0 | 40.5% to 41.2% | at most 0.01% / 0% |
| 0.75% | 0 | 25.6% to 26.3% | at most 0.01% / 0% |
| 1% | 0 | 15.2% to 16.6% | at most 0.01% / 0% |
| 0.5% | 0.5% | 16.2% to 16.7% | at most 0.02% / at most 0.01% |

Falls mirror rises, with misses of at most 0.05%. At n = 180 this interval almost never misses: it keeps a width with no discordant example, and the continuity correction widens it further.

## What this means

- **The paired FPR and FNR conditions don't ship.** The FPR and FNR conditions of `docs/design.md` §6 stay as they are. The planted-fault test null's 11.8%, the 43% INCONCLUSIVE after a promotion at 180 per class, and the FNR wrong-way rate at 180 per class remain known limits (§10).
- **None of the gate checks in `paired_rule_expected.json` ran.** They were conditional on an interval meeting the coverage bar.
- **No more candidates are tried under this bar.** Two runs, four intervals, and a candidate picked after seeing the first run's cells make further tries a search for an interval that passes, which a bar fixed in advance can't guard against.

## Reproduce

```bash
source scripts/dev-env.sh
uv run python experiments/phase2/paired_coverage.py run --run continuity_corrected
```

Run at commit `2c6e700` on a clean tree, CPU only, with 5 workers, against the bar committed in the same commit. Each cell is written to `.cache/paired-coverage-cc/rows.jsonl` as it finishes.
