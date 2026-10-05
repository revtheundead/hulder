# Hulder

Operating activation probes on self-hosted LLM serving, and telling when a serving change has silently broken one, or any other scorer.

## Language

### Serving

**Serving config**:
Everything about how a model is served that can change a scorer's scores: weight format, KV-cache format, engine version, batching. Reference and candidate are roles a serving config plays, not separate concepts.
_Avoid_: configuration, setup, deployment

**Reference**:
The serving config currently live, with its current threshold. Promoting a candidate makes it the new reference.
_Avoid_: baseline config, old config

**Training config**:
The serving config a probe's weights were trained on. It stays fixed while the reference moves with each rollout.
_Avoid_: original config, base config

**Candidate**:
The serving config being evaluated for rollout against a reference.
_Avoid_: new config, target

### Scoring

**Scorer**:
Anything that turns a request into a score derived from a served model and compares it against a threshold. An activation probe, a guard model's probability of "unsafe", or a semantic-cache similarity are all scorers.
_Avoid_: classifier, detector, monitor

**Probe**:
A scorer that reads the served model's own hidden states at one layer and position. In Hulder it is linear (logistic regression) and is the flagship scorer.
_Avoid_: activation classifier, linear head

**Guard model**:
A separate model served to classify requests as safe or unsafe, such as Llama Guard. Its score is the probability it assigns to "unsafe".
_Avoid_: judge (the judge is the guard model's role in the cascade, not the model)

### Probe task

**Prompt attack**:
A prompt that tries to make the model override its instructions or safety behaviour, by injection or jailbreak. The positive class of the example probe.
_Avoid_: prompt injection (for the task as a whole), jailbreak (alone), adversarial prompt

**Hard negative**:
A benign prompt that looks like a prompt attack, for example one that discusses overriding or ignoring something.
_Avoid_: benign-but-scary, tricky benign

### Gate

**Reference snapshot**:
The recorded scorer behaviour on the reference serving config over a fixed labelled calibration set, against which candidates are compared.
_Avoid_: baseline, ref file

**Noise floor**:
The run-to-run variation in scores when the same serving config is run twice on the same prompts.
_Avoid_: jitter, variance

**Shift**:
An offline, paired change in scores between reference and candidate on the same prompts, counted only when it exceeds the noise floor.
_Avoid_: drift (in gate contexts)

**Verdict**:
The gate's single categorical judgement on one candidate, decided by the scorer's performance rather than by shift: PASS, THRESHOLD STALE, PROBE DEGRADED or INCONCLUSIVE. Shift is reported alongside as a diagnostic.
_Avoid_: result, status, NO SHIFT

**Decision flip rate**:
The fraction of examples whose decision (score above or below the threshold) differs between reference and candidate on the same prompts. A diagnostic reported alongside the verdict, never an input to it.
_Avoid_: verdict flip rate, churn

**Recalibration**:
Choosing a new threshold for the same target false-positive rate on the candidate, keeping the scorer itself unchanged.
_Avoid_: retraining, retuning

### Runtime

**Drift**:
An online, unpaired change in the score distribution over live traffic, which raises an alarm rather than a verdict.
_Avoid_: shift (in runtime contexts)
