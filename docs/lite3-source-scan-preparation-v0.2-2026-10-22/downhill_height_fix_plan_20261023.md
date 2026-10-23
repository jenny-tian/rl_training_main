# Downhill height-reference regression

Authorized scope: fix Source safety height, run focused regressions, then repeat
one headless downhill L1 diagnostic. No other cell, training, or formal scan.

The Source runner selects `base_scan` safety height: root Z minus the highest
of the nine hits in the existing 0.1 m square base scanner. All hit/root XYZ must
be finite and every hit must be within 0.08 m horizontally of the current root.
Missing or invalid measurements yield zero clearance, triggering the unchanged
0.20 m height guard. This is a simulator terrain measurement, not a qualified
real-robot estimator. Other adapter clients retain the explicit legacy default.

Repeat `slope_L1_downhill`, geometry seed 10070, state seed 20070, speed 0.3 m/s,
5 degree slope, 1600-step cap, normal termination and SafetyFilter enabled.
Keep prior reports immutable. Save the new report as
`family_runtime_20261023/slope_L1_downhill_height_fix.json`.
No extra seeds, harder conditions, or retries based on policy outcome.

Regression checks cover flat decision parity, vertical translation invariance,
low clearance on elevated terrain, the highest surface at an edge, and missing,
nonfinite, malformed or spatially inconsistent scans. Full qualification and
normal manager automatic-reset coverage remain separate outstanding work.

Diagnostic amendment after the first run: the strict XY guard stopped at step
811. A same-condition replay with raw scan logging reproduced that stop and
showed finite rays lagging the root by about 0.0115 m; maximum ray distance was
0.08029 m. This was a measurement-timing issue, not a policy-driven seed change.
Update the Source base scanner each physics step, retain the 0.08 m spatial
check and 0.20 m height threshold, and run one corrected verification to
`family_runtime_20261023/slope_L1_downhill_height_fix_current_scan.json`.
Both intermediate reports remain immutable and are not qualification trials.
