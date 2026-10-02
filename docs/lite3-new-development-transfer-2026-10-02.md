# Frozen transfer to four registered new-development terrains

Protocol: `lite3_flat_anchor_transfer_v2_protocol_20261002.json`. The successful independent anchored policies (training seeds283/307) are fixed before evaluation. All four previously registered terrain seeds are retained; no extra training, checkpoint selection or replacement starts is permitted. Two slots run one simulation GPU each, with a total cap192,000 controls. Original controls are matched separately to each learned policy's reset master.

The initial launch attempt failed before any ledger/physics because the manifest validator omitted the new split; all16 logs remain under `artifacts/flat_anchor_new_development_20261002`, with0 actual/uncertain controls. The v2 attempt uses identical model, terrain and reset identities and retires the unused original allocation.

## Partial audited evidence, not final qualification

`artifacts/flat_anchor_new_development_v2_20261002/interim_seed283_first_two_audit.json` verifies the completed seed283 candidate/base pairs:

| New terrain seed | Candidate | Original base | Flat retention | Confirmed controls |
|---|---:|---:|---|---:|
|834818272|5/5|4/5|passed|18869|
|1490653948|5/5|3/5|passed|19079|

All expected trials in these pairs completed, phase-local starts matched, zero-update/residual freeze checks passed, and no uncertain controls were found. This partial audit is not a decision to stop or qualify: both learned policies and all four terrain instances must satisfy the registered final criteria. The remaining jobs are still running.

Repeated flat windows across terrain instances use the same reset master within each model/control pair; they must not be reported as additional independent flat-start samples. Every execution still remains charged and audited.
