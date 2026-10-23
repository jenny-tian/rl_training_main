# Source safety-height correction

Source scans now explicitly use terrain-relative vertical clearance: root Z
minus the highest terrain hit in the existing nine-ray, 0.1 m square base patch.
The scanner updates at each physics step (0.005 s). The 0.20 m protection threshold
is unchanged. Missing, malformed, nonfinite or spatially inconsistent scans
yield zero clearance and stop execution. Other adapter clients keep their legacy
default until separately configured. This simulator measurement is not a
qualified deployment height estimator.

## Verification

121 focused tests passed, including 11 new cases for flat filter parity, terrain
translation invariance, elevated low clearance, highest-surface edge handling,
and invalid scan protection. Compilation, whitespace checks, preparation audit,
and current registered artifact hashes passed. The draft remains non-executable.

All simulation runs used the same downhill L1 geometry/state seeds 10070/20070,
5 degree slope, 0.3 m/s command and 1600-step limit. Models were frozen, their
hashes unchanged, and no training or Sealed access occurred.

| Run | Steps | Progress (m) | Result |
| --- | ---: | ---: | --- |
| Original world-Z guard | 403 | 3.062 | Height stop despite about 0.314 m ground clearance |
| Relative height, control-period scanner | 811 | 7.755 | Invalid spatial alignment, fail-closed height stop |
| Raw-scan diagnostic replay | 811 | 7.755 | Reproduced scanner lag, max ray distance 0.08029 m |
| Relative height, physics-period scanner | 867 | 8.371 | Normal body-contact termination |

The first relative-height implementation exposed a scanner timing issue. Its
finite ray grid lagged the root by about 0.0115 m, exceeding the strict 0.08 m
root-to-ray spatial bound at one corner. Diagnostic replay and physics-period
correction were implementation checks, not extra independent capability trials.
All intermediate reports are retained under `family_runtime_20261023/`.

The final run had minimum conservative clearance 0.302089 m, zero height-filter
stops, zero invalid-state counts, and peak lateral offset 0.172490 m. It reached
8.371303 m progress but did not satisfy stable completion before non-foot contact
terminated the environment. Therefore `completed=false` remains the result.

The normal manager termination exercised the pre-reset callback without an
injected timeout. `pre_reset_capture_observed=true` and
`pre_reset_state_differs_from_post_reset=true`; the terminal root was
`[4.871303, 0.148312, -0.264447]` m, distinct from the reset state. Explicit reset
also passed finite-observation, zero-length and root-XY checks. The report's
`height_diagnostic_before_explicit_reset` is after automatic reset in this run;
its raw rays must not be mislabelled as terminal rays. Terminal metrics are in
`record`, captured before automatic reset.

Final report: `family_runtime_20261023/slope_L1_downhill_height_fix_current_scan.json`.
SHA-256: `f8810236d9d5ca0e3f7654166277e5c9f8bf9e8e3e70d6a38841ea8dbd0d127b`.
Final manifest SHA-256:
`2a41b603a381b7eace650a99c5c5b1540d2ffa1b3026659a700fccded77492ad`.
Earlier reports retain historical manifest/source hashes and do not claim the
current implementation was used.

Next proposed step: diagnose the downhill contact near the track endpoint and
the completion/hold geometry, then decide whether a track-exit correction is
needed. Do not relax safety limits or count endpoint crossing alone as success.
This step has not started; full Source qualification remains outstanding.
