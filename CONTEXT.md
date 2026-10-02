# Hulder

Operating activation probes on self-hosted LLM serving, and telling when a serving change has silently broken one.

## Language

### Serving

**Serving config**:
Everything about how the model is served that can change the activations a probe reads: weight format, KV-cache format, engine version, batching. Reference and candidate are roles a serving config plays, not separate concepts.
_Avoid_: configuration, setup, deployment

**Reference**:
The serving config currently live, with its current threshold. Promoting a candidate makes it the new reference.
_Avoid_: baseline config, old config

**Training config**:
The serving config the probe's weights were trained on. It stays fixed while the reference moves with each rollout.
_Avoid_: original config, base config

**Candidate**:
The serving config being evaluated for rollout against a reference.
_Avoid_: new config, target

### Probe task

**Prompt attack**:
A prompt that tries to make the model override its instructions or safety behaviour, by injection or jailbreak. The positive class of the example probe.
_Avoid_: prompt injection (for the task as a whole), jailbreak (alone), adversarial prompt

**Hard negative**:
A benign prompt that looks like a prompt attack, for example one that discusses overriding or ignoring something.
_Avoid_: benign-but-scary, tricky benign

### Gate

**Reference snapshot**:
The recorded probe behaviour on the reference serving config over a fixed labelled calibration set, against which candidates are compared.
_Avoid_: baseline, ref file

**Noise floor**:
The run-to-run variation in probe scores when the same serving config is run twice on the same prompts.
_Avoid_: jitter, variance

**Shift**:
An offline, paired change in probe scores between reference and candidate on the same prompts, counted only when it exceeds the noise floor.
_Avoid_: drift (in gate contexts)

**Verdict**:
The gate's single categorical judgement on one candidate, decided by probe performance rather than by shift: PASS, THRESHOLD STALE, PROBE DEGRADED or INCONCLUSIVE. Shift is reported alongside as a diagnostic.
_Avoid_: result, status, NO SHIFT

**Decision flip rate**:
The fraction of examples whose probe decision (above or below the threshold) differs between reference and candidate on the same prompts. A diagnostic reported alongside the verdict, never an input to it.
_Avoid_: verdict flip rate, churn

**Recalibration**:
Choosing a new threshold for the same target false-positive rate on the candidate, keeping the probe's weights unchanged.
_Avoid_: retraining, retuning

### Runtime

**Drift**:
An online, unpaired change in the probe score distribution over live traffic, which raises an alarm rather than a verdict.
_Avoid_: shift (in runtime contexts)
