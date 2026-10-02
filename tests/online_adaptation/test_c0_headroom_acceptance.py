from docs.analysis.audit_c0_headroom import headroom


def rows(progress,success=None,falls=None):
    return [dict(progress_m=p,completion=s,fell=f,safety_violation=False,valid_complete_episode=True)
            for p,s,f in zip(progress,success or [False]*3,falls or [False]*3)]


def test_extra_completion_requires_preserving_other_start_progress():
    base=rows([5,6,7])
    assert headroom(rows([8,6,7],[True,False,False]),base,[1,0])['passed']
    alternative=headroom(rows([8,5,7],[True,False,False]),base,[1,0])
    assert not alternative['extra_completion_no_other_progress_loss']
    assert alternative['passed']  # independent median-progress branch
    assert not headroom(rows([8,5,6],[True,False,False]),base,[1,0])['passed']


def test_progress_branch_cannot_hide_more_falls():
    base=rows([5,5,5])
    assert headroom(rows([5.6,5.6,5.6]),base,[0,1])['passed']
    assert not headroom(rows([5.6,5.6,5.6],falls=[True,False,False]),base,[0,1])['passed']


def test_zero_code_incomplete_or_empty_progress_cannot_qualify():
    c,b=rows([8,8,8]),rows([5,5,5])
    assert not headroom(c,b,[0,0])['passed']
    c[0]['valid_complete_episode']=False
    assert not headroom(c,b,[1,0])['passed']
    assert not headroom(rows([0,0,0]),rows([0,0,0]),[1,0])['passed']
    assert not headroom([],b,[1,0])['passed']
