"""恢复约束不直接作用于前进、主动转向、平稳站立或严重倾倒。"""
import pytest
import torch

from reference_low_stairs_recovery_gate import recovery_mask,recovery_loss


def inputs():
    command=torch.tensor([[0.,0.,0.],[.32,0.,0.],[0.,0.,.2],[0.,0.,-.8],
        [0.,0.,0.],[0.,0.,0.],[.0001,0.,0.]])
    tilt=torch.deg2rad(torch.tensor([20.,20.,20.,20.,3.,40.,20.]))
    return command,tilt


def test_gate_excludes_other_tasks_and_out_of_range_tilt():
    command,tilt=inputs()
    assert recovery_mask(command,tilt).tolist()==[True,False,False,False,False,False,False]


def test_only_selected_output_has_gradient_and_teacher_detached():
    command,tilt=inputs();prediction=torch.ones(7,12,requires_grad=True)
    teacher=torch.zeros_like(prediction,requires_grad=True)
    recovery_loss(prediction,teacher,command,tilt).backward()
    assert prediction.grad[0].gt(0.).all() and not prediction.grad[1:].any()
    assert teacher.grad is None


def test_empty_group_produces_zero_loss_and_zero_gradient():
    command,tilt=inputs();prediction=torch.ones(7,12,requires_grad=True);tilt[:]=0.
    loss=recovery_loss(prediction,torch.zeros_like(prediction),command,tilt)
    loss.backward();assert loss.item()==0. and not prediction.grad.any()


@pytest.mark.parametrize('which',['command','tilt','prediction','teacher'])
def test_nonfinite_rejected(which):
    command,tilt=inputs();prediction=torch.ones(7,12);teacher=torch.zeros(7,12)
    values=dict(command=command,tilt=tilt,prediction=prediction,teacher=teacher)
    values[which].view(-1)[0]=float('nan')
    with pytest.raises(ValueError):recovery_loss(prediction,teacher,command,tilt)
