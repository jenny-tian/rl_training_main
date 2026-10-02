# Independent route-policy learning replication

Protocol: `docs/lite3_full_policy_route_replication_protocol_20261002.json`. New training seed269, frozen evaluation seed271. Same algorithm, physics, observation/reward configuration and512-update budget as seed257; original base actor initialized again with zero route-input columns, fresh critic and optimizer. No trained state is imported from seed257.

Training completed2,097,152 charged/confirmed controls, uncertain0. Final checkpoint SHA256 `add03048cecded12015efb915fe32ee002b214d7d11cb88d52d02dd528e042b8`; training report SHA256 `b82fa03f84036b924f651ddeeb0bc983195ef15bd199f45e0f6d9cdc5117cc7e`. Runtime source SHA256 `bb3bec46a80863a408cb1766b93c4636147f332fa19b7a708bf71a2068e17778` matches the first route run exactly. Final checkpoint differs from the first learned policy.

Full2,097,152-row trajectory audit found0 actor-history, critic-history, reset-history and stored-emitted-command mismatches. Training episodes:767 successes,372 failures,302 deadlines; not a frozen success estimate.

Frozen evaluation is running on101/107 with new paired starts,5 task and2 flat windows per arm/terrain, total cap48,000 controls. `docs/analysis/audit_route_replication.py` will verify configuration and seed independence, checkpoint/report binding, repeated paired gains and flat retention, and>=9/10 aggregate per terrain with each run>=4/5. It deliberately leaves goal_complete false pending new-development terrain and full gait evidence. Prior failures and budgets are preserved; registered new terrain membership has not yet been generated or evaluated.

## Final result: replication gate failed

Seed269 candidate completed5/5 on101 and4/5 on107; original base completed1/5 on each. Combined with seed257, candidate totals10/10 and9/10. However seed269 failed paired flat retention on both comparisons (meanXY approximately0.02765m/s vs base0.01939m/s). The joint replication audit is false: rough completion alone does not qualify this configuration. New-development tests remain unexecuted; do not replace this failed run with a favorable extra seed.

Evaluation consumed37,480 confirmed controls, uncertain0. Cumulative full-policy training/evaluation7,478,483 controls; including registered engineering smokes7,479,651. Both models, all trajectories and failed retention evidence are preserved. Next work is an explicitly separate flat-anchor training configuration, requiring new preregistration and independent validation.
