"""Vectorized course terms for full-policy training, separate from evaluation reward."""
import torch


def course_state(root_pos,foot_pos,origins,*,distance=8.,lateral_limit=1.65):
    progress=root_pos[:,0]-origins[:,0]
    body=(root_pos[:,1]-origins[:,1]).abs()
    feet=(foot_pos[:,:,1]-origins[:,None,1]).abs().amax(dim=1)
    lateral=torch.maximum(body,feet)
    outside=lateral>lateral_limit
    return progress,lateral,(progress>=distance)&~outside,outside


def _state(env,asset_cfg):
    asset=env.scene[asset_cfg.name]
    return course_state(asset.data.root_pos_w,asset.data.body_pos_w[:,asset_cfg.body_ids],env.scene.env_origins)


def course_success(env,asset_cfg):
    return _state(env,asset_cfg)[2]


def course_lateral_failure(env,asset_cfg):
    return _state(env,asset_cfg)[3]


def course_lateral_cost(env,asset_cfg):
    return _state(env,asset_cfg)[1].square()


def terminal_event_value(success,failure,deadline,*,success_value=5.,failure_value=-2.):
    # Physical/orientation/lateral failure outranks crossing the finish line;
    # reaching it at the deadline still succeeds if no failure also occurs.
    return torch.where(failure,torch.full_like(success,dtype=torch.float32,fill_value=failure_value),
        torch.where(success,torch.full_like(success,dtype=torch.float32,fill_value=success_value),
                    torch.where(deadline,torch.full_like(success,dtype=torch.float32,fill_value=failure_value),
                                torch.zeros_like(success,dtype=torch.float32))))


def course_terminal_reward(env):
    manager=env.termination_manager
    failure=manager.get_term('illegal_contact')|manager.get_term('bad_orientation_2')|manager.get_term('course_lateral')
    value=terminal_event_value(manager.get_term('course_success'),failure,manager.get_term('time_out'))
    return value/env.step_dt  # RewardManager multiplies terms by dt once.
