# agentaudit

Tools for auditing what an automated support agent decides, one decision at a time.

## What this measures

An agent embedded in a staged classroom task can be described as adaptive on two
quite different grounds. It may adjust the amount of help it gives to the current
state of a learner's work, and it may reduce that help as the learner progresses.
These are separate properties. A system can satisfy the first completely while
never satisfying the second, and a single adaptivity score will hide the difference.

This package estimates the two separately:

- **Contingency** is the association between the state of the work at the moment of
  a decision and the intensity of support the agent then supplies.
- **Fading** is the change in the agent's willingness to release the learner across
  the sequence of decisions it makes for that learner.

Both are reported per group and pooled, with intervals from cluster resampling
rather than asymptotic approximations, because a classroom yields few clusters.

## Installation

```
python -m pip install -e ".[dev]"
```

## Running

The package reads a corpus of decision records. Point it at one through the
environment:

```
set AGENTAUDIT_DATA_DIR=<directory holding the decision, group and interaction tables>
python scripts/run_pipeline.py
```

With the variable unset the pipeline runs against the synthetic corpus committed
under `synthetic/`, so a fresh clone executes without any additional input.

Regenerate that corpus with:

```
python scripts/make_synthetic.py
```

## Expected input

Three tables, as spreadsheet or delimited text, named in `src/agentaudit/config.py`.
Only the columns listed in the column map are read; anything else is ignored and
never loaded. Each decision record is expected to carry an action label, an
activation trigger, a snapshot of task state, a provenance tag distinguishing
rule-based from generated output, and a timestamp.

## What the pipeline produces

Results are written to the output directory, which version control ignores. The
repository itself receives only the provenance package under `provenance/`:

| Artefact | Contents |
|---|---|
| `column_map.json` | the columns the analysis was permitted to read |
| `record_digests.json` | per-record and aggregate digests fixing the analysed set |
| `run_manifest.json` | code state, interpreter, package versions, seed and parameters |

## Release policy

Source code, configuration, environment specifications, synthetic demonstration
data and run provenance are published. Record-level data, derived data, estimates,
tables, figures, model artefacts and logs are not, in any form.

Provenance is published rather than withheld because a code availability statement
that nobody can check is not a disclosure. The digests let a reader confirm that a
later run examined the same records, without any record leaving restricted custody.

The policy is enforced mechanically. `tests/test_release_guard.py` fails if a
forbidden file type or directory is tracked, if the provenance package is stretched
beyond a column map, digests and manifests, or if a path to a restricted location
is committed. Run the suite before every push:

```
python -m pytest
```

## Layout

```
src/agentaudit/     analysis package
  config.py         paths, column map, parameters
  ingest.py         loading and digest fixing
  derive.py         analysis frame construction
  indices.py        contingency and fading
  sequence.py       transition structure between consecutive decisions
  models.py         policy model and association tests
  resample.py       cluster bootstrap and exact intervals
  provenance.py     the release package writer
  pipeline.py       stage ordering
scripts/            entry points
synthetic/          demonstration corpus, fabricated
provenance/         released disclosure artefacts
tests/              behavioural tests and release guards
```

## Licence

MIT. See `LICENSE`.
