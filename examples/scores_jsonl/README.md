# Bring your own scores

`hulder check` on scores from any scorer, one JSONL file per serving config. The scores here are synthetic, made by `make_example.py` from a fixed seed, so the example needs no model and no GPU. The candidate's scores were raised on purpose, so the threshold is stale.

From this directory:

```bash
hulder check --reference reference.jsonl --noise-floor rerun.jsonl --candidate candidate.jsonl \
    --threshold 0.12906 --target-fpr 0.05
```

It says THRESHOLD STALE, exits 10, and proposes a recalibrated threshold of 0.37 for the same 5% target.

## The format

One JSON object per line, with an `id` (a string or an integer), a `score` (a finite number, flagged when strictly above the threshold) and a `label` (0 benign, 1 attack). Other keys and blank lines are ignored. An optional first line `{"metadata": {...}}` describes the run, so the report can list the change under test and warn about any other difference. Without `base_model` in it, the report warns that it can't confirm both runs serve the same model.

```
{"metadata": {"model": "example-scorer", "base_model": "example-scorer", "quantization": "int8"}}
{"id": "q0000", "score": 0.156187, "label": 0}
{"id": "q0001", "score": 0.885365, "label": 1}
```

The candidate's and the rerun's rows are matched to the reference's by `id`, in any order, and every file must hold the same ids with the same labels. The format comes from the extension, `.npz` for a snapshot and `.jsonl` for scores, and `--input-format` overrides it. The threshold is the reference's benign 95th percentile, which `make_example.py` prints. Labels are binary, so for several independent labels, run `hulder check` once per label. Multi-class scorers are out of scope.
