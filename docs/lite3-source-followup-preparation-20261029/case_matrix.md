# Remaining L1 case matrix

All rows are prepared and unrun. Geometry seed is **10001** for every row. Initial-state seeds use the first value in the historical registration. Difficulties below are exact Python/JSON float representations; execution reads `preparation.json`. No row authorizes a run.

| Bank column | Explicit case ID | State seed | Pinned difficulty |
| --- | --- | ---: | ---: |
| 1 | `multi_scale_rough_L1_forward_lane` | 20030 | `0.6294702331215974` |
| 2 | `slope_L1_uphill` | 20060 | `0.3384848514104325` |
| 3 | `slope_L1_downhill` | 20070 | `0.3636354823034158` |
| 4 | `slope_L1_cross_slope_left` | 20080 | `0.30318197031305205` |
| 5 | `slope_L1_cross_slope_right` | 20090 | `0.2751071764368547` |
| 6 | `multi_step_stairs_L1_ascending` | 20180 | `0.02844575462330101` |
| 7 | `multi_step_stairs_L1_descending` | 20190 | `0.9917839285729633` |
| 8 | `slope_plus_rough_L1_uphill` | 20240 | `0.6540093430223354` |
| 9 | `slope_plus_rough_L1_downhill` | 20250 | `0.7852916736578777` |
| 10 | `slope_plus_rough_L1_cross_slope_left` | 20260 | `0.9256251464646599` |
| 11 | `slope_plus_rough_L1_cross_slope_right` | 20270 | `0.4893918379161446` |
| 12 | `friction_variation_L1_left_right_bands` | 20360 | `0.8649016238617524` |
| 13 | `friction_variation_L1_front_back_bands` | 20370 | `0.7265883240717208` |
| 14 | `friction_variation_L1_alternating_regions` | 20380 | `0.5308660894597165` |

Column 0 Flat is excluded from new execution candidates. Its archived seed 10000 / difficulty 0.0 mesh exactly matches bank column 0 (bank difficulty 0.7148064231614649), as Flat does not vary here. This equivalence does not transfer a runtime qualification to shared-bank motion.

After the relevant phase approval, select exactly one prepared case, for example the first row:

```bash
PYTHONPATH=source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_policy_prequalification.py \
  --case-id multi_scale_rough_L1_forward_lane \
  --run-dir /tmp/lite3_source_policy_prequalification_multi_scale_rough_L1_forward_lane_bank_20261029
```

The output directory must be new. Failure stops the phase; it does not authorize another seed, a repeat, a longer budget or the next row.
