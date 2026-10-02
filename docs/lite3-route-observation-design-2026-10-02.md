# Next full-policy method: explicit route observations

Status: design only, not execution authorization or completed experiment. Finish and audit the running symmetry pilot before editing source-bound runtime code. Register the final protocol and budget before physics.

The symmetry pilot's candidate101 failed5/5 by lane departure. Terminal yaw in the first two trials was approximately-22 and-18degrees. Body-frame mean lateral velocity near zero does not imply following the world-aligned track. A memoryless actor with angular rate and gravity cannot directly distinguish different static yaw orientations. Both candidate flat windows completed, but meanXY error0.02969m/s exceeded the matched base mean0.01853 plus0.005 allowance.

Proposed method:

- Explicit simulator-localization observations: normalized track-relative lateral position, sine yaw, cosine yaw. Do not claim unchanged proprio-only inputs or hardware transfer.
- Preserve fixed(.3,0,0) commands, terrain, physical limits and evaluation criteria. Feed extra route observations to the candidate neural policy, without a hand-coded steering controller.
- Insert three features after the existing velocity command term. Actor45 becomes48; critic48 becomes51. Last12 action-history coordinates remain unchanged.
- Expand only the first actor weight matrix by three zero columns. Existing mean mapping must reproduce the original base exactly for arbitrary extra features before learning. Keep the original checkpoint file unchanged; fresh critic and optimizer.
- Extend sagittal reflection: lateral offset and sine yaw negate; cosine yaw is unchanged. Existing angular/polar vector and leg reflection rules remain in force.
- Retain a separately registered full-policy PPO budget; investigate narrower velocity tracking tolerance and an explicit heading penalty. Any combined changes must be reported as a compound method, not a causal ablation.
- Frozen candidate execution must declare the expanded observation schema in its immutable worker configuration. Base remains the original45D actor. Phase-local matched starts, no exploration, no optimizer/replay writes, and all existing task/retention thresholds remain required.
- Before launch: test feature geometry, reflection, actor-initialization parity, actual PPO update, managed command/schema propagation, and a separately charged simulation smoke covering both training and frozen inference. Freeze the source before training/evaluation.

Keep prior costs and failures. Success still requires independent learning runs, robust task completion, flat retention, new-development terrain confirmation and gait evidence.
