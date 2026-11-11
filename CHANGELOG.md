# Changelog

Notable changes to hulder. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). `hulder.capture` is experimental, so its names may change in any 0.x release (`docs/design.md` §4).

## [Unreleased]

### Added

- An informational paired change under the verdict. For FPR and FNR the report counts the examples that flipped each way and gives the change with a 95% interval, Newcombe's square-and-add on continuity-corrected Wilson intervals, for example `FPR  +2.78 points [-1.49, +7.42]   8 newly flagged, 3 newly unflagged of 180 benign`. `Report.paired_changes` and `--json` carry it, and `schema_version` stays 1. It doesn't affect the verdict, exit codes or floors. As a verdict rule it missed a coverage bar fixed beforehand, with one end missing in up to 3.68% of samples against a nominal 2.5% (`docs/results/phase2/paired_coverage_cc.md`), so paired FPR and FNR conditions don't ship (`docs/design.md` §10).
- When PASS is out of reach because of the target FPR, an INCONCLUSIVE report names both ways out: the minimum size, and the smallest `--target-fpr` at which PASS becomes possible at the current size, for example `PASS needs at least 183 benign in all, or at 180 benign a --target-fpr of at least 0.0102, if that tolerance fits your release policy.` It is exact, like the size check, rounded up to three significant figures. `OutOfReach.target_fpr_for_pass` and `--json` carry it, and `schema_version` stays 1.

### Changed

- The report's Bands line gives both FPR lines, for example `candidate FPR between 0.5× and 2× target (2.5% to 10.0%)`. The low line was always part of the rule, as THRESHOLD STALE, but wasn't printed.
- The summary at the top of an INCONCLUSIVE report with a real shift names the conditions that are undecided, for example `whether its FPR and FNR crossed their tolerances is undecided`, instead of always FPR.

## [0.3.0] - 2026-11-10

### Added

- `hulder.runtime`, in the `vllm` extra and experimental like `hulder.capture`: in-server probe scoring for `vllm serve` on vLLM 0.30.0. A KV connector scores a linear probe on each request's last prompt token at the end of prefill, with no host sync, and puts the score in a non-streamed reply's `kv_transfer_params` as `hulder_score` and on `/metrics`. It is monitor-only, so nothing waits for the score, and a streamed or very short reply can come back without one. `hulder.runtime.serve_args(probe_path)` returns the `--speculative-config` and `--kv-transfer-config` arguments that load it, with no drafted tokens, and refuses a probe that doesn't record a layer, position -1 and a threshold.
- A draft counter in `hulder.runtime`. vLLM 0.30.0 drops the zero-draft schedule without an error when async scheduling is off or `data_parallel_size` is above 1, and each decode step then carries a drafted token again. The connector counts every scheduled draft in `hulder_draft_tokens` on `/metrics` and logs one warning naming the likely causes (`docs/results/phase3/draft_counter.md`).
- `docs/runtime.md`, the `vllm serve` command, metrics and limits, with example Prometheus alerting rules in `examples/prometheus/hulder-rules.yml` for the flag rate, missing scores and draft tokens.

## [0.2.1] - 2026-11-10

### Added

- INCONCLUSIVE reports name each undecided condition. `Report.undecided` lists each condition whose interval straddles its line (FPR, FNR or ΔAUROC), with the interval and the line. For FPR and FNR it adds a `floor`, the fewest extra benign or attack prompts that could settle the condition if every new one went the most favourable way. Fewer cannot settle it, and more may not. The CLI prints one line per condition under the verdict, for example `FPR 3.9% to 12.0% against the 10.0% line; at least 16 more benign prompts are needed, as fewer cannot settle it.` ΔAUROC gets its interval line only. `--json` carries it, and `schema_version` stays 1.
- When PASS is impossible at the current size even with no errors, an INCONCLUSIVE report says so before the per-condition lines, with the size at which it becomes possible, for example `PASS is out of reach at 180 benign, even with no false positives: the FPR bound stays above the 2.0% line below 183 benign.` `Report.out_of_reach` carries it.
- The report gives no estimate of how much data would probably settle a condition. Two were tested on fresh synthetic data against bars fixed beforehand and missed (`docs/results/phase2/settle.md`, `settle_v2.md`). The same runs showed that at about 180 prompts per class the FNR and ΔAUROC conditions can settle the wrong way, 9% to 13% of the time (`docs/design.md` §10).
- `hulder.capture.Capture.prompt_token_counts`, each prompt's templated length in tokens as vLLM tokenized it. With chunked prefill off, a batch over `max_num_batched_tokens` is prefilled over more than one step, so this shows whether a batch ran as one forward pass.
- `hulder.capture.Capture.engine_seconds` and `batch_seconds`. The first is the time to construct the vLLM engine, covering weight loading, profiling, KV-cache allocation, compilation and warmup, and the second is the wall time of each batch's `llm.chat` call including reading back its hidden states, so a capture's rate can be measured apart from its startup.
- `hulder.clopper_pearson(k, n, alpha=0.05)`, the exact binomial interval behind an INCONCLUSIVE report's floors and size check, for any rate's interval. It broadcasts over arrays.
- `hulder.capture.Capture.prompt_token_ids`, each prompt's templated token ids as vLLM tokenized it, so two captures can be shown to have fed the engine the same tokens rather than only the same prompt ids.

### Changed

- The CLI no longer prints the single count of benign examples that would settle the FPR condition "if the observed rate held", nor its tolerance-edge caveat. The per-condition lines replace it. `Report.more_benign_needed` keeps its old meaning for compatibility.
- Release wheels and sdists are built by a workflow on GitHub's runners in the public repository, with their SHA-256s in the run's summary, instead of on a developer's machine.

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

[0.3.0]: https://github.com/revtheundead/hulder/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/revtheundead/hulder/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/revtheundead/hulder/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/revtheundead/hulder/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/revtheundead/hulder/releases/tag/v0.1.0
