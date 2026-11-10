# Coverage of three paired intervals for the FPR and FNR change

**No candidate interval meets the bar, so the paired FPR and FNR conditions don't ship as specified.** The bar and the choice rule were committed before the run (`paired_coverage_expected.json`, commit `37c1546`), on seeds no earlier run used, and haven't moved. Full output in `paired_coverage.json`.

## The question and the bar

The proposed paired conditions (`paired_rule_expected.json`) bound the change in FPR or FNR on the same examples, (b − c) / n, from b examples newly flagged and c newly unflagged. Each candidate gives a two-sided 95% interval, and the gate reads one end per condition: the upper bound for PASS and the lower bound for THRESHOLD STALE. So each end should miss the true change in about 2.5% of samples.

- **Intervals:** exact conditional (McNemar-style, a Clopper-Pearson interval on the discordant split), Tango's score interval, and Newcombe's square-and-add interval (`experiments/phase2/paired_intervals.py`).
- **Tables:** 768 cells at n = 180, 910, 1,050 and 1,191, with 1%, 5% or 15% flagged by both runs and each one-way rate from 0 to 5%. 10,000 multinomial tables per cell, seeds 7,000 to 7,999.
- **Bar:** in every cell, each end misses in at most 3.5% of tables, and over all cells each end's mean miss rate is at most 2.75%. Of the intervals meeting both, the narrowest on average would be used. If none meets both, the paired rule doesn't ship.

## Result

| Interval | Mean miss, low / high | Cell-ends over 3.5% | Worst cell-end | Mean width | Meets the bar |
|---|---|---|---|---|---|
| Exact conditional | 12.82% / 12.80% | 740 of 1,536 | 84.0% | 1.57 points | No |
| Tango's score | 2.01% / 2.01% | 110 of 1,536 | 4.83% | 2.77 points | No |
| Newcombe's square-and-add | 1.32% / 1.32% | 5 of 1,536 | 4.36% | 2.84 points | No |

With 10,000 tables a cell's miss rate has a standard error of about 0.18 points near 3.5%.

**Exact conditional** fails wherever discordant examples are rare. With none observed it gives the single point 0, so it misses every real change in those tables.

**Tango's** misses are one-sided. Its lower bound sits above a real rise, and its upper bound below a real fall, in up to 4.8% of tables. That happens mostly at changes of 0.5 to 1 point at n = 910 to 1,191. The lower bound at a rise is the end THRESHOLD STALE reads. Its worst upper-bound miss at a rise, the end PASS reads, is 2.78%.

**Newcombe's** five cell-ends over the bar are all at 15% flagged by both and n of 1,050 or 1,191:

| Cell | End | Miss | Tango's |
|---|---|---|---|
| n = 1,050, fall of 0.75 points | low | 4.36% | 1.58% |
| n = 1,050, fall of 2 points | low | 3.65% | 2.20% |
| n = 1,191, fall of 2 points | low | 3.64% | 2.02% |
| n = 1,050, rise of 0.75 points | high | 3.57% | 1.40% |
| n = 1,050, rise of 2 points | high | 3.57% | 2.21% |

Four are within about 1 standard error of the bar and one is clearly over. Two of them, the high end at a rise, are on the PASS side, so they lean towards passing a real rise. 24 more cell-ends sit between 3% and 3.5%. Elsewhere Newcombe's is conservative, with a mean miss of 1.32% per end.

## The low-discordance cells at n = 180

The cells with a total discordant rate of 0.5% to 1%, where no discordant example appears in 15% to 41% of tables. Misses in percent, low / high.

| Candidate only | Reference only | No discordant | Exact conditional | Tango's | Newcombe's |
|---|---|---|---|---|---|
| 0.5% | 0 | 40% to 41% | 0.0 / 40.3 to 41.4 | 0.2 / 0.0 | 0.0 / 0.0 |
| 0.75% | 0 | 25% to 26% | 0.0 / 60.6 to 60.9 | 0.2 to 0.3 / 0.0 | 0.0 to 0.1 / 0.0 |
| 1% | 0 | 16% to 17% | 0.0 to 0.1 / 46.0 to 46.6 | 0.9 to 1.1 / 0.0 | 0.1 / 0.0 |
| 0.5% | 0.5% | 16% to 17% | 0.0 / 0.0 | 0.5 to 0.6 / 0.5 to 0.6 | 0.0 / 0.0 to 0.1 |

Ranges are over 1%, 5% and 15% flagged by both. Falls mirror rises with the ends swapped. Every cell is in `paired_coverage.json` under `low_discordance_n180`.

- **The exact conditional interval misses a real rise of 0.5% to 1% in 40% to 61% of tables.** That's more than the share with no discordant example, because a few discordant examples still give an upper bound below the truth. The gate would read that as no change and PASS.
- **Tango's and Newcombe's hold at n = 180.** Both stay at or under 1.2% per end. They keep a width with no discordant example.

## What this means

- **The paired rule doesn't ship as specified, and the current FPR and FNR conditions stay.** Nothing here is retuned.
- **The exact conditional interval is unusable for the gate.** It passes small real rises whenever few examples flip.
- **The two score intervals miss the bar in different places.** Tango's is slightly liberal on one end over a broad band of one-way changes. Newcombe's is conservative on average and slightly liberal in a few cells with many examples flagged by both. Neither failure is at n = 180.
- **Any further candidate interval needs its own bar,** committed before it runs, on seeds this run didn't use.

## Reproduce

```bash
source scripts/dev-env.sh
uv run python experiments/phase2/paired_coverage.py run
```

Run at commit `7d68f89` on a clean tree, CPU only, with 5 workers, against the bar committed at `37c1546`. Each cell is written to `.cache/paired-coverage/rows.jsonl` as it finishes.
