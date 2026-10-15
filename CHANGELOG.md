# Changelog

Notable changes to hulder. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). `hulder.capture` is experimental, so its names may change in any 0.x release (`docs/design.md` §4).

## [0.2.0] - 2026-10-15

### Added

- THRESHOLD STALE, and INCONCLUSIVE labelled optional, propose a recalibrated threshold for the same target FPR, fitted on the candidate's benign scores with a binomial-tail bound. With 95% confidence over calibration sets its FPR is at most the target, whatever the score distribution, provided future benign traffic resembles the calibration set. The report gives the candidate's FNR at the new threshold, or, when there are too few benign examples to certify the target, how many would (299 at a 1% target). It was fitted on the scores the verdict read, so confirm it on a fresh candidate snapshot before relying on it. Measured in `docs/results/phase2/recalibration.md`.
- When there are too few attacks to confirm the FNR condition at the new threshold, the report warns that later checks against it will likely be INCONCLUSIVE, and gives the number of attacks that would avoid that.
- `Report.recalibration`, a new public `Recalibration` with the method, confidence, threshold, benign count above it, candidate FNR, and the benign or attack counts needed. `--json` carries it, and `schema_version` stays 1.
- `hulder check` reads bring-your-own scores from any scorer, one JSONL file per serving config with an `id`, `score` and `label` per line. The other runs' rows are matched to the reference's by id. The format comes from each file's extension, `.npz` or `.jsonl`, and `--input-format` overrides it. Any other extension is a usage error. `examples/scores_jsonl/` has the format and a synthetic example.
- An optional first line `{"metadata": {...}}` in a JSONL file describes the run, so the report can list the change under test and confirm both runs serve the same `base_model`.

### Changed

- `Report` is unhashable by design. 0.1 advertised hashing, but `hash()` raised `TypeError` whenever a change under test held a dict.
- The warning that a threshold wasn't calibrated on the reference snapshot no longer fires for thresholds that are conservative on purpose, such as a promoted recalibrated threshold, so they check cleanly.

### Known limits

- At small calibration sets, a promoted recalibrated threshold often can't be confirmed. At 180 benign and 180 attacks, an unchanged rerun checked against it was INCONCLUSIVE 43% of the time in the study, because the FNR condition's interval is about as wide as its 0.05 tolerance. The warning above fired before 96% of those cases. At 910 attacks the problem didn't occur.
- A serving config whose engine lands on one of two numerical paths at each start isn't detected. On L4s, 2 of 18 bf16 captures of one Qwen3-4B config took a second path about 1.3% away in activation space. A rerun on the same path as the reference measures no noise, so a candidate on the other path shows a shift the serving change didn't cause. See `docs/design.md` §10.

## [0.1.1] - 2026-10-15

### Fixed

- `hulder.capture` works with bf16 models. 0.1.0 failed on every bf16 model, because numpy has no bfloat16 type. Hidden states now pass through float32 and are stored as float16, and a value beyond float16's range raises an error naming its prompt, layer and position instead of being stored as infinity.
- `base_model_for` reads a model card's `base_model` written as a one-element list, the form official FP8 checkpoints such as `Qwen/Qwen3-4B-FP8` use. 0.1.0 recorded no base model and warned.
- `base_model_for` treats online quantization, where vLLM quantizes an unquantized checkpoint at load (for example `quantization="fp8"`), as the model itself. 0.1.0 read the card's pretrained parent instead, so `check` reported a model swap against the unquantized reference.
- A checkpoint quantized before upload and captured with `quantization=None`, which vLLM detects from its `config.json`, takes its card's base model. 0.1.0 recorded the checkpoint itself, so `check` reported a model swap.
- `check` gives its base-model warning when a snapshot records `base_model` as None, as it does when the key is missing. 0.1.0 gave it only for a missing key, so a None was flagged only by the probe comparison, when a probe was applied. Verdicts, exit codes and report figures are unchanged.

### Changed

- `base_model_for` decides from the checkpoint's `config.json`. With a `quantization_config` the checkpoint was quantized before upload, and its card names the base model. Without one, the model is its own base.
- `base_model_for` never raises. A card naming several base models or none, no card, a damaged card, an unreachable Hugging Face Hub, or `HF_HUB_OFFLINE` without a cached copy each return a warning that says what was assumed and how to pass `override=`.
- Experimental API: the injected `card` function of `base_model_for` returns the card's `base_model` field as written (a string, a list or None) instead of `str | None`. A new keyword argument, `checkpoint_quantized`, takes a function `(model, revision) -> bool`.
- The live tutorial's capture prints a `base_model` warning and carries on.
- The PyPI description is shorter and reaches a working example in one screen.

## [0.1.0] - 2026-10-12

### Added

- `hulder check`, a release gate that compares a scorer on a reference and a candidate serving config over the same labelled examples. It returns PASS, THRESHOLD STALE, SCORER DEGRADED or INCONCLUSIVE with confidence intervals, a paired shift test against a measured noise floor, exit codes 0, 10, 11 and 12, and `--json`.
- `hulder quickstart`, two checks on bundled activations from Qwen2.5-1.5B-Instruct, on CPU.
- The Python API: `check`, `check_snapshots`, `apply_probe`, `Snapshot`, `Report`, `Probe`, `Estimate`, `Verdict`, `PairingError` and `prompt_ids_sha256`.
- `hulder[vllm]`, with the experimental `hulder.capture` for offline activation capture from vLLM 0.30.0.
- The live tutorial, which captures the demo with vLLM on a GPU.

[0.2.0]: https://github.com/revtheundead/hulder/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/revtheundead/hulder/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/revtheundead/hulder/releases/tag/v0.1.0
