# Lite3 `model_1998.pt` offline parity audit

Date: 2026-10-22. This audit used CPU-only inference on fixed inputs. It did not launch Isaac Sim, start an environment, run episodes, evaluate the policy in simulation, or train a model.

## Result

The checkpoint Actor was reconstructed from its saved weights and compared with the existing TorchScript and ONNX exports on 351 identical 45D inputs. The TorchScript export matched exactly. The largest ONNX absolute error was `1.9073486328125e-6`, below the audit tolerance of `1e-5`.

| Comparison | Maximum absolute error | Mean absolute error | Result |
|---|---:|---:|---|
| TorchScript vs checkpoint Actor | 0 | 0 | Pass |
| ONNX vs checkpoint Actor | 1.9073486328125e-6 | 1.8918373712040193e-7 | Pass |
| ONNX vs TorchScript | 1.9073486328125e-6 | 1.8918373712040193e-7 | Pass |

The checkpoint is iteration 1998. Its Actor is a deterministic 45→512→256→128→12 MLP with ELU between linear layers and a separate 12-element Gaussian log-standard-deviation parameter. Agent configuration sets Actor observation normalization to false and empirical normalization to an empty mapping; the TorchScript graph uses an identity normalizer. The ONNX model accepts fixed batches of `[1, 45]`, returns `[1, 12]`, and has no embedded metadata.

The input corpus is reproducible from seed 1998: zero, ±1 and ±0.5 constants, 256 uniform samples in `[-1, 1]`, and ±45 identity-basis vectors. The corpus hash and full artifact hashes are in [the machine-readable report](lite3-checkpoint-parity-2026-10-22.json).

## Observation and action contract

The saved environment configuration gives the 45 Actor inputs in this order: base angular velocity (3, scale 0.25), projected gravity (3), velocity command (3), relative joint positions (12), joint velocities (12, scale 0.05), and previous action (12). Joint groups are leg-major in the order FL, FR, HL, HR, with HipX, HipY, Knee inside each group. The configuration enables observation corruption: angular velocity, gravity, joint position, and joint velocity receive noise. The original random noise samples were not saved, so this audit establishes output parity for identical inputs, not replay of a historical noisy episode.

The environment action term uses the same joint order, scales HipX by 0.125 and HipY/Knee by 0.25, and adds the robot's configured default joint pose `(0, -0.65, 1.3)` per leg. The shared action conversion helper uses those same values. The export helper had stale Lite3 metadata defaults `-0.8` and `1.6`; these were corrected to match the trained environment. The existing TorchScript and ONNX files carry no action metadata, and their neural-network outputs were not changed.

This is policy-output parity plus a source/configuration audit of the action transform. It does not establish full simulator-to-deployment parity, actuator behavior, SafetyFilter behavior, or locomotion performance. The fixed ONNX batch-size-one interface also means the parity tool intentionally evaluates rows separately.

As a separate exporter smoke check, the corrected metadata helper generated a temporary ONNX file outside the repository. Its 12 joint names, action scales, and default pose matched the saved environment configuration, and its outputs matched the existing TorchScript export on 34 fixed inputs with maximum absolute error `1.430511474609375e-6`. The checked-in and run-directory export files were not regenerated or changed.

## Reproduction

From the repository root, with the `env_isaaclab` Python environment:

```bash
python scripts/tools/audit_lite3_checkpoint_parity.py \
  --checkpoint logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt \
  --torchscript logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/exported/policy.pt \
  --onnx logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/exported/policy.onnx \
  --context-file logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/params/agent.yaml \
  --context-file logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/params/env.yaml \
  --context-file source/rl_training/rl_training/lite3_joint_conventions.py \
  --context-file scripts/tools/export_onnx_fast.py \
  --report docs/lite3-checkpoint-parity-2026-10-22.json
```

The command writes a JSON report only when the requested report path does not already exist. It reads the checkpoint and export files without modifying them.
