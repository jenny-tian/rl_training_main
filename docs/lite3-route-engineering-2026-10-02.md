# Route observation engineering validation

Training and frozen execution support `sim-localization-v1`: lateral offset/1.65 clipped to[-2,2], sine yaw and cosine yaw, inserted after velocity commands. Actor48/critic51; original base45 remains unchanged. The expanded first layer inserts zero columns, preserving the original mapping within float32 tolerance. No hand-coded steering is added.

59 targeted tests passed for route geometry, symmetry, original/expanded PPO updates, actor-only initialization, evaluation scheduling, CLI and managed trials. Vector smoke completed520 controls with no updates and no uncertain controls, including the forced deadline. Actual-state mean output parity, emitted-action histories, reset histories and unchanged parameters passed.

First frozen launch was rejected before session creation because retention-trials0 is invalid:0 controls. Logs and command record retained. The separately registered correction used the required retention schedule count1 but executed only evaluation. Candidate/base each completed64 frozen controls, no replay/optimizer writes, matched starts and initial action parity. Maximum subsequent action difference0.01811 was retained descriptively; trajectories are not asserted bitwise identical. Both checkpoint files and zero residual actors remained unchanged. Total route engineering cost648 controls.

Artifacts: `artifacts/policy_route_smoke_20261002/attempt01/report.json`, `frozen_execution.json`, `frozen_execution_v2.json`, `frozen_audit.json`. The next training protocol binds the vector and frozen audit hashes. These checks establish integration only; they do not establish learned gait or rough-terrain success.
