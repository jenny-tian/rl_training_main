# Stage-1 qualification

Design: qualify; declared final checkpoints only.

Selected seed: none.
No qualified seed: neither final checkpoint passes all of Q1-Q4.

| Policy | Q1 | Q2 | Q3 | Q4 | L2 completion | Total safety |
| --- | --- | --- | --- | --- | --- | --- |
| m1998 | FAIL | FAIL | PASS | PASS | 0.49583333333333335 | 24 |
| seed42 | FAIL | FAIL | FAIL | PASS | 0.9333333333333333 | 32 |
| seed43 | FAIL | FAIL | FAIL | FAIL | 0.9291666666666667 | 32 |

## m1998

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt`

SHA-256: `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`

| Level | Family | Completed/n | Rate | 95% CI | Safety | Step-weighted projection |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | slope | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L1 | multi_step_stairs | 63/64 | 0.9844 | 0.9160–0.9996 | 1 | 0.000000 |
| L1 | slope_plus_rough | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L1 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |
| L2 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | slope | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L2 | multi_step_stairs | 14/64 | 0.2188 | 0.1251–0.3397 | 23 | 0.000000 |
| L2 | slope_plus_rough | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L2 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |

| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3103 | 0.5425 | 1.3017 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3078 | 0.6755 | 1.0988 |
| L1 | slope_uphill | 0/32 | 0.0000–0.1089 | 0 | 4.5307 | 1.2422 | 4.9062 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3072 | 0.4771 | 1.1800 |
| L1 | slope_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 5.6590 | 1.3135 | 0.9870 |
| L1 | slope_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 4.5284 | 1.3161 | 0.8251 |
| L1 | stairs_ascending | 31/32 | 0.8378–0.9992 | 1 | 9.1037 | 0.8132 | 1.2901 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3079 | 0.5314 | 1.4028 |
| L1 | slope_rough_uphill | 0/32 | 0.0000–0.1089 | 0 | 4.7542 | 1.1243 | 4.7178 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3068 | 0.5134 | 1.2334 |
| L1 | slope_rough_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 5.5148 | 1.3187 | 0.6491 |
| L1 | slope_rough_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 4.4651 | 1.3107 | 1.1776 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3131 | 0.5133 | 1.4493 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3124 | 0.5053 | 1.4755 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3140 | 0.4472 | 1.4607 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3103 | 0.5425 | 1.3017 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3087 | 0.7301 | 1.2013 |
| L2 | slope_uphill | 0/32 | 0.0000–0.1089 | 0 | 2.4754 | 0.3417 | 10.2406 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3073 | 0.5374 | 0.7067 |
| L2 | slope_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 4.3969 | 1.2959 | 2.1487 |
| L2 | slope_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 3.7612 | 1.2981 | 1.9464 |
| L2 | stairs_ascending | 0/32 | 0.0000–0.1089 | 5 | 2.1567 | 0.2744 | 4.0421 |
| L2 | stairs_descending | 14/32 | 0.2636–0.6234 | 18 | 5.2122 | 0.3139 | 1.3782 |
| L2 | slope_rough_uphill | 0/32 | 0.0000–0.1089 | 0 | 2.7212 | 0.3135 | 7.6693 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3062 | 0.5948 | 0.8119 |
| L2 | slope_rough_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 4.2463 | 1.2968 | 2.0595 |
| L2 | slope_rough_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 3.7429 | 1.2869 | 2.7180 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3163 | 0.4237 | 1.6984 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3155 | 0.4438 | 1.7286 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3184 | 0.3591 | 1.6871 |

Q4 detail: `{"actual": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`

## seed42

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106/logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-10_16-00-10_full_seed42/model_3497.pt`

SHA-256: `47889ee59d07416287332164a38566a812d6ebbc0be637e13752829f8891d49c`

| Level | Family | Completed/n | Rate | 95% CI | Safety | Step-weighted projection |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | slope | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L1 | multi_step_stairs | 64/64 | 1.0000 | 0.9440–1.0000 | 0 | 0.000000 |
| L1 | slope_plus_rough | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L1 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |
| L2 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | slope | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L2 | multi_step_stairs | 32/64 | 0.5000 | 0.3723–0.6277 | 32 | 0.000000 |
| L2 | slope_plus_rough | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L2 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |

| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3180 | 0.1226 | 1.1627 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3177 | 0.1153 | 1.2441 |
| L1 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3142 | 0.1131 | 0.8228 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3167 | 0.2411 | 1.2917 |
| L1 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3148 | 0.5646 | 1.1299 |
| L1 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3147 | 0.4228 | 0.6283 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3177 | 0.1310 | 1.1753 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3180 | 0.1259 | 1.0846 |
| L1 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3160 | 0.1484 | 0.7981 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3171 | 0.2016 | 1.2314 |
| L1 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3179 | 0.6206 | 1.0924 |
| L1 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3168 | 0.4488 | 0.6227 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3179 | 0.0908 | 1.3619 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3184 | 0.1293 | 1.0990 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3179 | 0.0989 | 1.4634 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3180 | 0.1226 | 1.1627 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3184 | 0.1148 | 1.2608 |
| L2 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3150 | 0.1634 | 0.5259 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3160 | 0.2376 | 1.3426 |
| L2 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3154 | 0.7113 | 1.0981 |
| L2 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3146 | 0.5924 | 0.1488 |
| L2 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3190 | 0.1327 | 1.1934 |
| L2 | stairs_descending | 0/32 | 0.0000–0.1089 | 32 | 2.1152 | 0.0658 | 1.1472 |
| L2 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3165 | 0.2000 | 0.4110 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3168 | 0.1960 | 1.2524 |
| L2 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3183 | 0.7153 | 0.9430 |
| L2 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3162 | 0.5516 | 0.1695 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3173 | 0.0974 | 1.3540 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3186 | 0.1329 | 1.0805 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3182 | 0.1087 | 1.3386 |

Q4 detail: `{"actual": {"falls_per_episode": 0.0224609375, "rms_xy_error": 0.059836738559757324, "rms_yaw_error": 0.07388872473116274}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`

## seed43

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106/logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-10_17-11-25_full_seed43/model_3497.pt`

SHA-256: `32d0a5e30460880f274a1a4fbc39176c8aca001fad55a95df4fbf598217fc11d`

| Level | Family | Completed/n | Rate | 95% CI | Safety | Step-weighted projection |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | slope | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L1 | multi_step_stairs | 64/64 | 1.0000 | 0.9440–1.0000 | 0 | 0.000000 |
| L1 | slope_plus_rough | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L1 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |
| L2 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | slope | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L2 | multi_step_stairs | 30/64 | 0.4688 | 0.3428–0.5977 | 32 | 0.000000 |
| L2 | slope_plus_rough | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L2 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |

| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3213 | 0.0858 | 0.5840 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3215 | 0.1032 | 0.5614 |
| L1 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3192 | 0.2158 | 0.6701 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3200 | 0.2727 | 0.3785 |
| L1 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3203 | 0.4364 | 0.6818 |
| L1 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3186 | 0.4733 | 0.4932 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3227 | 0.1002 | 0.5708 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3199 | 0.0804 | 0.7371 |
| L1 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3188 | 0.2189 | 0.8300 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3201 | 0.2162 | 0.4576 |
| L1 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3234 | 0.4550 | 0.7096 |
| L1 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3204 | 0.5218 | 0.5190 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3223 | 0.0848 | 0.5983 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3223 | 0.1018 | 0.6493 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3216 | 0.0899 | 0.5582 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3213 | 0.0858 | 0.5840 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3216 | 0.1036 | 0.5531 |
| L2 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3193 | 0.1111 | 0.5071 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3205 | 0.2534 | 0.3303 |
| L2 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3196 | 0.5713 | 0.8005 |
| L2 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3180 | 0.6645 | 0.2002 |
| L2 | stairs_ascending | 30/32 | 0.7919–0.9923 | 0 | 8.9431 | 0.1309 | 0.9716 |
| L2 | stairs_descending | 0/32 | 0.0000–0.1089 | 32 | 2.0785 | 0.0663 | 0.6608 |
| L2 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3182 | 0.1082 | 0.7381 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3209 | 0.1951 | 0.4616 |
| L2 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3226 | 0.5201 | 0.8431 |
| L2 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3203 | 0.7024 | 0.1744 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3227 | 0.0915 | 0.5648 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3221 | 0.1063 | 0.6199 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3231 | 0.0971 | 0.5602 |

Q4 detail: `{"actual": {"falls_per_episode": 0.05105691056910569, "rms_xy_error": 0.06328891990518798, "rms_yaw_error": 0.11688959512778624}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`
