# Headless termination-capture smoke audit — 2026-10-23

This bounded test checked that the environment callback captures a terminal sample before Isaac Lab automatically resets the environment. It used the registered Flat L1 cell and frozen `model_1998.pt`; it injected a four-step manager timeout and disabled the three normal failure terms listed in the termination contract.

The command ran with Isaac Lab's headless Kit experience. It exited successfully after four steps with the active termination term `time_out`. The pre-reset capture was observed, and its root position differed from the post-reset position:

- Last pre-reset root position: `[-3.4946713, -0.0005049, 0.3574216]` m
- Post-reset root position: `[-3.5, 0.0, 0.375]` m

The report confirms unchanged actor-state hashes, `policy_updated: false`, `sealed_accessed: false`, and `qualification_claim: false`. The machine-readable record is [termination_capture_smoke_20261023.json](termination_capture_smoke_20261023.json).

This verifies only the injected-timeout capture path. It does not qualify ordinary failure termination, terrain performance, or the full Source scan, and the preparation manifest remains `draft_not_for_run` with `execution_authorized: false`.

At the time of this smoke, the offline preparation audit was schema-valid but not execution-ready. Runner registration and configuration hash refresh were still pending. The subsequent offline draft update registered the runner, refreshed the metrics and termination configuration hashes, and linked both bounded smoke reports. Runtime qualification, proposed-cutoff review, and separate scan approval remain outstanding. This smoke report retains the previous manifest hash as historical provenance.

Validation also passed for Python compilation, JSON parsing, `git diff --check`, and 110 focused tests across the Source scan runner, bridge, manifest, and Isaac Lab adapter.
