# Bounded family runtime diagnostic

Scope authorized by the user's agreement to the next small headless terrain check.
This is a diagnostic of the existing frozen policy, metrics, and reset wiring.
It does not authorize the 225-episode scan, training, or Sealed access.

Fixed order and budget, declared before execution:

| Cell | Geometry seed | Initial-state seed | Maximum control steps |
| --- | --- | --- | --- |
| slope_L1_uphill | 10060 | 20060 | 1600 |
| slope_L1_downhill | 10070 | 20070 | 1600 |
| multi_step_stairs_L1_ascending | 10180 | 20180 | 1600 |
| multi_step_stairs_L1_descending | 10190 | 20190 | 1600 |

Each cell runs once with the registered geometry and 0.3 m/s command. Slopes
are 5 degrees; stairs have three 0.02 m risers and 0.5 m treads. The existing
SafetyFilter and normal failure terms remain active. No timeout is injected;
the runner stops at completion, a normal failure/safety boundary, or the cap.
An explicit seeded reset follows each diagnostic without another control step.
No added seeds or harder levels are selected from the outcomes.

Inspect progress into the non-flat region, finite common/family metrics,
contact-conditioned slip, stair progress/collision/clearance proxies, the actual
termination terms, automatic pre-reset capture when triggered, and explicit
reset observations and episode length. An untriggered automatic reset is
uncovered, not a passing test. A safety stop is a diagnostic result and must
not be bypassed to obtain a full track traversal. Reports retain frozen-model,
checkpoint, source-code and manifest hashes. Existing outputs are not reused.

These deterministic L1 repeats must not be combined with the later formal scan
as independent terrain instances or used to claim a success rate.
