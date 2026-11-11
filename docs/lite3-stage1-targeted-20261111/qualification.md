# Stage-1 qualification

Design: targeted; declared final checkpoints only.

Selected seed: none.
No qualified seed: neither final checkpoint passes all of Q1-Q4.

| Policy | Q1 | Q2 | Q3 | Q4 | L2 completion | Total safety |
| --- | --- | --- | --- | --- | --- | --- |
| m1998 | FAIL | FAIL | FAIL | PASS | 0.50625 | 27 |
| starting | FAIL | FAIL | FAIL | PASS | 0.9333333333333333 | 32 |
| seed42 | PASS | PASS | PASS | FAIL | 0.9875 | 5 |
| seed43 | FAIL | FAIL | PASS | FAIL | 0.9645833333333333 | 7 |

## m1998

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt`

SHA-256: `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`

| Level | Family | Completed/n | Rate | 95% CI | Safety | Step-weighted projection |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L1 | slope | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L1 | multi_step_stairs | 62/64 | 0.9688 | 0.8916–0.9962 | 2 | 0.000000 |
| L1 | slope_plus_rough | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L1 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |
| L2 | flat | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | multi_scale_rough | 32/32 | 1.0000 | 0.8911–1.0000 | 0 | 0.000000 |
| L2 | slope | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L2 | multi_step_stairs | 19/64 | 0.2969 | 0.1891–0.4242 | 25 | 0.000000 |
| L2 | slope_plus_rough | 32/128 | 0.2500 | 0.1777–0.3342 | 0 | 0.000000 |
| L2 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |

| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3105 | 0.5342 | 1.3301 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3079 | 0.6642 | 1.0521 |
| L1 | slope_uphill | 0/32 | 0.0000–0.1089 | 0 | 4.5071 | 1.2571 | 4.8993 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3068 | 0.4882 | 1.1854 |
| L1 | slope_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 5.6396 | 1.3137 | 0.9943 |
| L1 | slope_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 4.5203 | 1.3150 | 0.9098 |
| L1 | stairs_ascending | 30/32 | 0.7919–0.9923 | 2 | 8.8956 | 0.7726 | 1.4119 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3076 | 0.5293 | 1.4238 |
| L1 | slope_rough_uphill | 0/32 | 0.0000–0.1089 | 0 | 4.7945 | 1.0945 | 4.5025 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3073 | 0.5280 | 1.2421 |
| L1 | slope_rough_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 5.5284 | 1.3215 | 0.4570 |
| L1 | slope_rough_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 4.4908 | 1.3097 | 1.2301 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3126 | 0.4869 | 1.4846 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3131 | 0.5038 | 1.4670 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3145 | 0.4223 | 1.3748 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3105 | 0.5342 | 1.3301 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3079 | 0.7140 | 1.1398 |
| L2 | slope_uphill | 0/32 | 0.0000–0.1089 | 0 | 2.4720 | 0.3618 | 10.0972 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3073 | 0.5486 | 0.6792 |
| L2 | slope_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 4.3834 | 1.2967 | 2.0603 |
| L2 | slope_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 3.7570 | 1.2968 | 2.0193 |
| L2 | stairs_ascending | 0/32 | 0.0000–0.1089 | 12 | 2.1628 | 0.3057 | 4.6662 |
| L2 | stairs_descending | 19/32 | 0.4064–0.7630 | 13 | 6.3912 | 0.3742 | 1.2770 |
| L2 | slope_rough_uphill | 0/32 | 0.0000–0.1089 | 0 | 2.7291 | 0.3106 | 7.3910 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3062 | 0.6062 | 0.8963 |
| L2 | slope_rough_cross_slope_left | 0/32 | 0.0000–0.1089 | 0 | 4.2519 | 1.2982 | 1.9599 |
| L2 | slope_rough_cross_slope_right | 0/32 | 0.0000–0.1089 | 0 | 3.7435 | 1.2854 | 2.7819 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3165 | 0.3979 | 1.6617 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3162 | 0.4340 | 1.7351 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3187 | 0.3444 | 1.6758 |

Q4 detail: `{"actual": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`

## starting

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
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3174 | 0.1267 | 1.1403 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3183 | 0.1226 | 1.2555 |
| L1 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3162 | 0.1166 | 0.8629 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3165 | 0.2288 | 1.2815 |
| L1 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3142 | 0.5742 | 1.1748 |
| L1 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3154 | 0.4187 | 0.6298 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3186 | 0.1222 | 1.1735 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3167 | 0.1381 | 1.0580 |
| L1 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3162 | 0.1466 | 0.7743 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3170 | 0.2016 | 1.2136 |
| L1 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3193 | 0.6307 | 1.0815 |
| L1 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3169 | 0.4430 | 0.6073 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3177 | 0.1002 | 1.3672 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3174 | 0.1270 | 1.1029 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3182 | 0.1292 | 1.3219 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3174 | 0.1267 | 1.1403 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3170 | 0.1206 | 1.2278 |
| L2 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3157 | 0.1738 | 0.5100 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3164 | 0.2223 | 1.3415 |
| L2 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3147 | 0.7219 | 1.0463 |
| L2 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3154 | 0.5838 | 0.1515 |
| L2 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3192 | 0.1254 | 1.1554 |
| L2 | stairs_descending | 0/32 | 0.0000–0.1089 | 32 | 2.3009 | 0.0709 | 1.2349 |
| L2 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3160 | 0.2180 | 0.4120 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3176 | 0.2034 | 1.2594 |
| L2 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3190 | 0.7075 | 0.9345 |
| L2 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3160 | 0.5546 | 0.2230 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3179 | 0.1114 | 1.3159 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3172 | 0.1289 | 1.0963 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3182 | 0.1266 | 1.2946 |

Q4 detail: `{"actual": {"falls_per_episode": 0.0224609375, "rms_xy_error": 0.059836738559757324, "rms_yaw_error": 0.07388872473116274}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`

## seed42

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106/logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-11_09-18-09_targeted_seed42/model_4496.pt`

SHA-256: `1e28d7b6c6206fa591867943acc76485f3ab75328f1774d014b4be3e26404177`

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
| L2 | multi_step_stairs | 58/64 | 0.9062 | 0.8070–0.9648 | 5 | 0.000000 |
| L2 | slope_plus_rough | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L2 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |

| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3069 | 0.1245 | 3.8141 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3072 | 0.1556 | 3.6835 |
| L1 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3033 | 0.1212 | 4.1072 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3030 | 0.2547 | 2.9530 |
| L1 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3007 | 0.6424 | 3.6193 |
| L1 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3046 | 0.3042 | 4.2246 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.3086 | 0.1320 | 3.8674 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.3053 | 0.1064 | 3.8482 |
| L1 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3036 | 0.1363 | 4.2161 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3040 | 0.3314 | 3.0063 |
| L1 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3077 | 0.6448 | 3.8231 |
| L1 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3058 | 0.2737 | 4.3003 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3068 | 0.1011 | 3.9221 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3072 | 0.1127 | 3.9764 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3066 | 0.1422 | 3.6853 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.3069 | 0.1245 | 3.8141 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.3069 | 0.1490 | 3.6800 |
| L2 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3035 | 0.1440 | 4.2709 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3020 | 1.0876 | 0.6996 |
| L2 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3006 | 0.7560 | 3.7011 |
| L2 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3034 | 0.2807 | 4.4259 |
| L2 | stairs_ascending | 31/32 | 0.8378–0.9992 | 0 | 9.1323 | 0.1145 | 3.8846 |
| L2 | stairs_descending | 27/32 | 0.6721–0.9472 | 5 | 8.3090 | 0.2615 | 3.3527 |
| L2 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.3037 | 0.1535 | 4.4859 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.3023 | 1.0237 | 0.9109 |
| L2 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.3074 | 0.7341 | 3.8464 |
| L2 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.3054 | 0.2595 | 4.4927 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3064 | 0.0972 | 3.9481 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.3061 | 0.1185 | 4.0151 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.3061 | 0.1348 | 3.8068 |

Q4 detail: `{"actual": {"falls_per_episode": 0.020182291666666668, "rms_xy_error": 0.06235626404071395, "rms_yaw_error": 0.1385053055283686}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`

## seed43

Actor: `/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106/logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-11_09-18-10_targeted_seed43/model_4496.pt`

SHA-256: `fc45e19eaf49a4b3a3dfb2af931699f9a0dfc086be23bff9fdc3defbd877c12c`

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
| L2 | multi_step_stairs | 47/64 | 0.7344 | 0.6091–0.8370 | 7 | 0.000000 |
| L2 | slope_plus_rough | 128/128 | 1.0000 | 0.9716–1.0000 | 0 | 0.000000 |
| L2 | friction_variation | 96/96 | 1.0000 | 0.9623–1.0000 | 0 | 0.000000 |

| Level | Condition | Completed/n | 95% CI | Safety | Progress m | Peak lateral m | Abs yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.2737 | 0.6510 | 1.3178 |
| L1 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.2739 | 0.6131 | 1.2563 |
| L1 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.2731 | 0.6069 | 1.0204 |
| L1 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.2750 | 0.2354 | 0.3305 |
| L1 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.2737 | 0.8530 | 0.7579 |
| L1 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.2735 | 0.2570 | 0.9717 |
| L1 | stairs_ascending | 32/32 | 0.8911–1.0000 | 0 | 9.2752 | 0.6357 | 1.2687 |
| L1 | stairs_descending | 32/32 | 0.8911–1.0000 | 0 | 9.2729 | 0.5918 | 1.4117 |
| L1 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.2731 | 0.6577 | 1.0728 |
| L1 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.2746 | 0.2219 | 0.2874 |
| L1 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.2762 | 0.8588 | 0.5905 |
| L1 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.2745 | 0.2981 | 0.7812 |
| L1 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.2742 | 0.6558 | 1.2768 |
| L1 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.2751 | 0.6443 | 1.2300 |
| L1 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.2748 | 0.6703 | 1.3427 |
| L2 | flat | 32/32 | 0.8911–1.0000 | 0 | 9.2737 | 0.6510 | 1.3178 |
| L2 | multi_scale_rough | 32/32 | 0.8911–1.0000 | 0 | 9.2749 | 0.5886 | 1.2850 |
| L2 | slope_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.2743 | 0.4172 | 0.4459 |
| L2 | slope_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.2755 | 0.2568 | 0.7385 |
| L2 | slope_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.2753 | 0.8182 | 0.3374 |
| L2 | slope_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.2744 | 0.6152 | 0.3649 |
| L2 | stairs_ascending | 18/32 | 0.3766–0.7364 | 4 | 6.5486 | 0.4734 | 7.2228 |
| L2 | stairs_descending | 29/32 | 0.7498–0.9802 | 3 | 8.6238 | 0.5037 | 1.2717 |
| L2 | slope_rough_uphill | 32/32 | 0.8911–1.0000 | 0 | 9.2731 | 0.4417 | 0.3994 |
| L2 | slope_rough_downhill | 32/32 | 0.8911–1.0000 | 0 | 9.2734 | 0.2354 | 0.7237 |
| L2 | slope_rough_cross_slope_left | 32/32 | 0.8911–1.0000 | 0 | 9.2769 | 0.8684 | 0.5586 |
| L2 | slope_rough_cross_slope_right | 32/32 | 0.8911–1.0000 | 0 | 9.2746 | 0.6432 | 0.3068 |
| L2 | friction_left_right_bands | 32/32 | 0.8911–1.0000 | 0 | 9.2743 | 0.6535 | 1.2873 |
| L2 | friction_front_back_bands | 32/32 | 0.8911–1.0000 | 0 | 9.2758 | 0.6392 | 1.1565 |
| L2 | friction_alternating_regions | 32/32 | 0.8911–1.0000 | 0 | 9.2759 | 0.6785 | 1.3924 |

Q4 detail: `{"actual": {"falls_per_episode": 0.009114583333333334, "rms_xy_error": 0.0632532757862564, "rms_yaw_error": 0.10347367043417838}, "baseline": {"falls_per_episode": 0.029612756264236904, "rms_xy_error": 0.05669851781912998, "rms_yaw_error": 0.07779961096267259}, "limits": {"falls_per_episode": 0.0396127562642369, "rms_xy_error": 0.062368369601042986, "rms_yaw_error": 0.08557957205893985}}`
