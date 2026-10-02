# Frozen transfer to four registered new-development terrains

Protocol: `lite3_flat_anchor_transfer_v2_protocol_20261002.json`. The successful independent anchored policies (training seeds283/307) are fixed before evaluation. All four previously registered terrain seeds are retained; no extra training, checkpoint selection or replacement starts is permitted. Two slots run one simulation GPU each, with a total cap192,000 controls. Original controls are matched separately to each learned policy's reset master.

The initial launch attempt failed before any ledger/physics because the manifest validator omitted the new split; all16 logs remain under `artifacts/flat_anchor_new_development_20261002`, with0 actual/uncertain controls. The v2 attempt uses identical model, terrain and reset identities and retires the unused original allocation.

## Partial audited evidence, not final qualification

`artifacts/flat_anchor_new_development_v2_20261002/interim_seed283_all_four_audit.json` verifies the completed seed283 candidate/base pairs:

| New terrain seed | Candidate | Original base | Flat retention | Confirmed controls |
|---|---:|---:|---|---:|
|834818272|5/5|4/5|passed|18869|
|1490653948|5/5|3/5|passed|19079|
|1132853762|5/5|3/5|passed|19280|
|874689437|4/5|0/5|passed|17650|

All expected trials in these pairs completed, phase-local starts matched, zero-update/residual freeze checks passed, and no uncertain controls were found. This partial audit is not a decision to stop or qualify: both learned policies and all four terrain instances must satisfy the registered final criteria. At the time of this interim audit, the remaining jobs were still running. The final result is recorded below.

Repeated flat windows across terrain instances use the same reset master within each model/control pair; they must not be reported as additional independent flat-start samples. Every execution still remains charged and audited.

The seed283 failure on874689437 was a physical fall after approximately6.12m in the fifth registered task trial; it is retained without replacement. Seed307 has now completed5/5 task trials on that terrain, meeting the combined9/10 task threshold. That observation preceded completion of the remaining terrain/retention work and final audit.

## Final audit passed

All16 sessions and112 planned trials completed. Candidate totals by new terrain are10/10,10/10,10/10,9/10; paired base totals6/10,6/10,6/10,0/10. Every paired flat-retention comparison passed, and all checkpoint/source/zero-update/initial-state checks passed. Actual152,079 controls, uncertain0. The single candidate fall remains in the dataset. See `lite3-stable-rough-walking-result-2026-10-03.md` and `lite3_registered_acceptance_20261003.json` for the six-terrain completion audit and scope.
