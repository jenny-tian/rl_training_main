import torch
from rl_training.policy_course import course_state,terminal_event_value


def test_course_coordinates_follow_each_cloned_origin_and_check_feet():
    origins=torch.tensor([[0.,0.,0.],[10.,20.,0.],[20.,40.,0.]])
    roots=origins+torch.tensor([[8.,0.,.375],[8.,0.,.375],[7.,0.,.375]])
    feet=origins[:,None,:]+torch.zeros(3,4,3)
    feet[1,0,1]+=1.66
    progress,lateral,success,failure=course_state(roots,feet,origins)
    assert progress.tolist()==[8.,8.,7.]
    assert success.tolist()==[True,False,False]
    assert failure.tolist()==[False,True,False]


def test_terminal_priority_preserves_failure_and_success_at_deadline():
    value=terminal_event_value(torch.tensor([True,True,False,False]),torch.tensor([True,False,False,False]),torch.tensor([True,True,True,False]))
    assert value.tolist()==[-2.,5.,-2.,0.]
