# Stage-1 qualification

Design: extend; declared final checkpoints only.

Selected seed: none.
No qualified seed: neither final checkpoint passes all of Q1-Q4.

| Policy | Q1 | Q2 | Q3 | Q4 | L2 completion | Total safety |
| --- | --- | --- | --- | --- | --- | --- |
| m1998 | FAIL | FAIL | FAIL | PASS | 0.48541666666666666 | 34 |
| seed42 | FAIL | FAIL | FAIL | FAIL | 0.9333333333333333 | 32 |
| seed43 | FAIL | FAIL | FAIL | FAIL | 0.9333333333333333 | 32 |

## m1998

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt`

SHA-256: `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`

| Level | Family | Completed/n | Rate | 95% CI | Safety | Step-weighted projection |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | slope | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L1 | multi_step_stairs | 64/64 | 1.0000 | 0.9440–1.0000 | 0 | 0.000000 |
| L1 | slope_plus_rough | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L1 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |
| L2 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | slope | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L2 | multi_step_stairs | 9/64 | 0.1406 | 0.0664–0.2502 | 34 | 0.000000 |
| L2 | slope_plus_rough | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L2 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |

| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3101 | 0.5376 | 1.2536 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3085 | 0.6805 | 1.0913 |
| L1 | slope_uphill | 0/32 | 0.0000–0.1089 | 0 | 4.4757 | 1.2470 | 4.6544 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3072 | 0.4834 | 1.1453 |
| L1 | slope_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 5.6197 | 1.3133 | 0.9986 |
| L1 | slope_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 4.5089 | 1.3152 | 0.8585 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3121 | 0.8313 | 1.2802 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3065 | 0.5372 | 1.3992 |
| L1 | slope_rough_uphill | 0/32 | 0.0000–0.1089 | 0 | 4.8023 | 1.1290 | 4.9179 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3076 | 0.5088 | 1.2202 |
| L1 | slope_rough_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 5.5285 | 1.3219 | 0.4268 |
| L1 | slope_rough_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 4.4237 | 1.3107 | 1.1792 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3127 | 0.5003 | 1.4399 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3127 | 0.5038 | 1.5054 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3138 | 0.4600 | 1.4935 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3101 | 0.5376 | 1.2536 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3078 | 0.7309 | 1.1522 |
| L2 | slope_uphill | 0/32 | 0.0000–0.1089 | 0 | 2.3478 | 0.3627 | 10.1242 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3074 | 0.5353 | 0.7260 |
| L2 | slope_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 4.3727 | 1.2949 | 2.2001 |
| L2 | slope_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 3.7397 | 1.2965 | 2.0478 |
| L2 | stairs_ascending | 0/32 | 0.0000–0.1089 | 11 | 2.1632 | 0.3192 | 4.1055 |
| L2 | stairs_descending | 9/32 | 0.1375–0.4675 | 23 | 4.0809 | 0.2736 | 1.3024 |
| L2 | slope_rough_uphill | 0/32 | 0.0000–0.1089 | 0 | 2.7546 | 0.3205 | 7.8389 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3065 | 0.5909 | 0.7677 |
| L2 | slope_rough_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 4.2468 | 1.2972 | 2.0659 |
| L2 | slope_rough_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 3.7159 | 1.2862 | 2.7485 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3168 | 0.4152 | 1.6213 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3166 | 0.4379 | 1.7301 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3184 | 0.3649 | 1.7272 |

Q4 detail: `{"actual": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`

## seed42

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106/logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-10_22-43-50_extend_seed42/model_4496.pt`

SHA-256: `e619ba0b2717c9d67c7aa7611a8c1db031f89c3dc33bea5a11365d58cb9c9792`

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
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3054 | 0.1407 | 1.8935 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3057 | 0.1303 | 1.8305 |
| L1 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3050 | 0.1425 | 1.8619 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3039 | 0.5815 | 2.6902 |
| L1 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3042 | 0.6730 | 1.9549 |
| L1 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3047 | 0.1321 | 1.9265 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3065 | 0.1035 | 1.8913 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3055 | 0.1380 | 1.8133 |
| L1 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3044 | 0.1169 | 1.6639 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3036 | 0.5784 | 2.5698 |
| L1 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3073 | 0.6445 | 1.8286 |
| L1 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3048 | 0.1432 | 1.8867 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3056 | 0.1145 | 1.7976 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3055 | 0.1284 | 1.9104 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3066 | 0.1032 | 1.7317 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3054 | 0.1407 | 1.8935 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3050 | 0.1456 | 1.8427 |
| L2 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3041 | 0.2957 | 2.0117 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3027 | 0.7689 | 3.1413 |
| L2 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3032 | 0.7024 | 1.7445 |
| L2 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3043 | 0.2642 | 1.9527 |
| L2 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3063 | 0.1108 | 1.8775 |
| L2 | stairs_descending | 0/32 | 0.0000–0.1089 | 32 | 2.1381 | 0.0844 | 2.5933 |
| L2 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3034 | 0.2585 | 1.8339 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3037 | 0.7443 | 3.1106 |
| L2 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3065 | 0.6799 | 1.6156 |
| L2 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3052 | 0.2703 | 1.9266 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3057 | 0.1107 | 1.7759 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3059 | 0.1331 | 1.9782 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3054 | 0.0964 | 1.6378 |

Q4 detail: `{"actual": {"falls_per_episode": 0.024739583333333332, "rms_xy_error": 0.0586544676436561, "rms_yaw_error": 0.09873350702399247}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`

## seed43

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106/logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-10_22-43-55_extend_seed43/model_4496.pt`

SHA-256: `a1608ed9982982bd26a7d551f357b3d3d9f8f9cc986ffcbee2e3c734a76ddf2e`

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
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3117 | 0.1195 | 0.7431 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3108 | 0.1153 | 0.5817 |
| L1 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3087 | 0.7347 | 0.5024 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3106 | 0.2324 | 1.3295 |
| L1 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3089 | 0.5715 | 0.2462 |
| L1 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3088 | 0.6288 | 0.2647 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3139 | 0.1023 | 0.7903 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3128 | 0.0952 | 0.8365 |
| L1 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3093 | 0.7240 | 0.2390 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3120 | 0.3087 | 1.6079 |
| L1 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3120 | 0.5844 | 0.2154 |
| L1 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3104 | 0.6815 | 0.2774 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3124 | 0.1099 | 0.7580 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3128 | 0.1156 | 0.7051 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3126 | 0.1044 | 0.6536 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3117 | 0.1195 | 0.7431 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3107 | 0.1143 | 0.6188 |
| L2 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3093 | 0.6654 | 0.3835 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3121 | 0.3744 | 1.8878 |
| L2 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3098 | 0.8150 | 0.4158 |
| L2 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3084 | 0.7084 | 0.4808 |
| L2 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3131 | 0.1440 | 0.7713 |
| L2 | stairs_descending | 0/32 | 0.0000–0.1089 | 32 | 2.3475 | 0.0895 | 2.6349 |
| L2 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3099 | 0.7469 | 0.2669 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3121 | 0.4816 | 2.1047 |
| L2 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3131 | 0.7585 | 0.5498 |
| L2 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3096 | 0.7125 | 0.4572 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3127 | 0.1162 | 0.6780 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3137 | 0.1198 | 0.7002 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3119 | 0.0973 | 0.7379 |

Q4 detail: `{"actual": {"falls_per_episode": 0.010416666666666666, "rms_xy_error": 0.060993693987441214, "rms_yaw_error": 0.09310829351774132}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`
