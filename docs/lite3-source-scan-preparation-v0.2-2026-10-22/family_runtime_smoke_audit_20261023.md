# Headless Source family runtime diagnostics

The four predeclared L1 diagnostics ran once each with frozen `model_1998.pt`,
0.3 m/s forward commands, unchanged safety limits and a 1600-step cap. No timeout
was injected. No training, formal 225-episode scan, or Sealed access occurred.
The preparation manifest remains `draft_not_for_run`, `execution_authorized=false`.

| Condition | Steps | Forward progress (m) | Ending | Peak lateral offset (m) |
| --- | ---: | ---: | --- | ---: |
| 5 degree uphill | 1154 | 4.148 | lateral_out_of_bounds | 1.803 |
| 5 degree downhill | 403 | 3.062 | safety_termination: height | 0.136 |
| Three 2 cm ascending stairs | 1544 | 8.301 | completion | 0.932 |
| Three 2 cm descending stairs | 1268 | 8.305 | completion | 0.293 |

The stair episodes crossed all three tread exit boundaries and satisfied the
runner's full-track completion and stable-hold condition. Ascending/descending
front-foot collision proxies counted 0/1 events; contact-calibrated mean swing
clearances were 0.02843/0.02876 m. The runtime field description still omits the
contact calibration qualifier; the actual collector subtracts the per-foot
median contact offset, as specified in `metrics_config.json`. These are proxies,
not direct contact-point collision or sole-distance measurements.

Common slip fractions were 0.01484, 0.02550, 0.02184 and 0.02828 in the order above.
All records had finite JSON, zero invalid-state counts, available required
slope/stair metric fields, consistent slip numerators/denominators, and unchanged
frozen-model/checkpoint/source hashes. Each explicit seeded reset returned finite
observations, a zero episode-length buffer, and the registered root XY origin.

## Blocking height-reference finding

The downhill safety stop cannot be used as a policy capability failure.
`Lite3IsaacLabAdapter.safety_state` passes root world Z as `base_height`
(`isaaclab_adapter.py:319`), and SafetyFilter compares it to 0.20 m
(`safety.py:163`). At the stop, world Z was 0.199234918 m. An independent vertical
intersection with the regenerated mesh beneath the final root XY gave surface
Z = -0.114775188 m, hence vertical clearance = 0.314010106 m. There was no non-foot
contact; the filter reason was `height`. A falling terrain datum thus triggered
the absolute-world-height guard while local clearance remained above 0.20 m.

The same world-height assumption can weaken low-clearance protection on elevated
terrain. Before a formal scan, define and implement a terrain-relative safety
height, retain conservative behavior for missing/invalid surface measurements,
and verify flat parity and vertically shifted terrain cases. Do not lower or
disable the current threshold to obtain a downhill completion.

## Coverage limits

None of these four episodes triggered manager `illegal_contact` or
`bad_orientation_2`; `termination_terms` was empty and automatic pre-reset capture
was false in all reports. Out-of-bounds, SafetyFilter pause, and completion were
runner-level endings. Explicit reset passed, but ordinary manager-termination
automatic reset remains uncovered. The earlier injected-timeout test remains
separate evidence and does not close this gap.

Single deterministic L1 trials do not estimate success rates or qualify the
six-family Source distribution. Uphill lateral drift and the 0.932 m ascending
stair lateral offset remain motion-quality findings despite the latter's
completion. No thresholds, commands, geometry, or seeds were changed in response.

## Evidence and verification

Manifest SHA-256:
`dba281ad8a2fe31adae410c4df65947b51886ceb597c5472f3ca38bcfbaa6c54`.

Reports in `family_runtime_20261023/` retain full per-step action traces, cell
definitions, explicit reset checks, and source/frozen-module hashes:

| Report | SHA-256 |
| --- | --- |
| slope_L1_uphill.json | 2a3ba4b4e4789d97f0ffc41145c653edeee471b8938f884255b036e9fdda0c6b |
| slope_L1_downhill.json | bad00a3fe106e765c477f8ecd5080f3fb6fe84f9cdc69fa7f01ef07646bef728 |
| multi_step_stairs_L1_ascending.json | 28d79d087aa6c0955387b8320d9a35afc8e69720cfcfbd44b47213f617a79d25 |
| multi_step_stairs_L1_descending.json | 783ccd74bcadb14ebcc9d021ed4ff1d2c54033a4a2935264617fd741270b61b6 |

All four processes exited with code zero. Existing Source manifest, runner,
bridge and adapter tests passed (110 tests). Script compilation, output report
assertions, preparation-structure audit, and `git diff --check` passed. The
preparation audit still reports `execution_ready=false`.

Recommended next separately approved step: correct the safety height reference
and add focused regression checks, then repeat the same bounded downhill
diagnostic with preserved old results. A controlled normal-termination/reset
check remains necessary before claiming complete runtime qualification.
