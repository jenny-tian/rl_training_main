from docs.analysis.summarize_full_policy_gait import summarize


def test_motion_summary_retains_empty_windows_and_excludes_gapped_stride_estimates():
    assert summarize([], .02)=={'samples':0}
    rows=[dict(step=s,foot_contact=c,foot_lift_from_contact_m=[.04]*4) for s,c in
          [(1,[0,0,0,0]),(2,[1,0,0,1]),(10,[0,1,1,0]),(11,[1,0,0,1])]]
    result=summarize(rows,.02)
    assert result['stance_onset_intervals'][0]['count']==0
    assert result['all_feet_airborne_fraction']==.25
    assert result['diagonal_contact_fraction']==.75
    assert result['swing_lift_mean_m']==[.04]*4


def test_transfer_gait_summary_includes_both_models_and_controls_on_all_terrain():
    from docs.analysis.summarize_full_policy_gait import session_specs
    specs=session_specs({'schema':'lite3_route_transfer_v1','terrain_seeds':[1,2,3,4],
                         'policies':[{'learning_seed':283},{'learning_seed':307}]})
    assert len(specs)==16 and len({x[3] for x in specs})==16
    assert (4,'base',307,'base-307-4') in specs
