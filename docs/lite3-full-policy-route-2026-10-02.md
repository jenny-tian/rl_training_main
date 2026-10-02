# Lite3 route-observation PPO pilot

Protocol: `docs/lite3_full_policy_route_protocol_20261002.json`. Training seed257, frozen evaluation seed263. Actor48/critic51 with explicit simulator route localization; original fixed commands, physical course and evaluation criteria preserved. Fresh original-base actor initialization via zero input columns, fresh critic and optimizer. Mirror loss1, velocity tracking std.15, heading cost-1. This compound change includes additional sensing and does not establish equal-sensing superiority.

Training completed512 updates and2,097,152 charged/confirmed controls, uncertain0. Final checkpoint SHA256 `298ea36cfd907bf833e109b6cbfb2c96025b9ed0b1e719ab381380ee9be713a8`; report SHA256 `c48799ccdeb83631c9b844e6a5e5cd08cf81be43f10e1c9a2cffa8866b63ca41`. Source SHA256 `bb3bec46a80863a408cb1766b93c4636147f332fa19b7a708bf71a2068e17778`.

Full trajectory audit covered2,097,152 rows, with0 actor-history, critic-history, reset-history and stored-emitted-action mismatches. Training counted910 successes,309 failures and222 deadlines across the mixed flat/rough environments. These are training diagnostics, not frozen completion rates.

Frozen final-checkpoint comparison has launched: terrain101 candidate first, terrain107 original base first; each arm5 task trials up to2,000 steps and2 flat windows of1,000 steps. Candidate worker configuration declares `sim-localization-v1`; original base retains45D observations. Phase-local starts are paired and no online updates/exploration are permitted. Evaluation cap48,000 controls. Runtime source must stay fixed until both supervisors exit.

Prior full-policy training/evaluation3,209,837 controls plus this training totals5,306,989 before current evaluation. Route engineering648 and earlier environment smoke520 are separately preserved (including0-control failed frozen CLI launches). Stable gait qualification remains unproven and requires frozen task/retention success, independent training replication, new-development terrain and gait evidence.
