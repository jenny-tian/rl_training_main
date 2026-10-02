import torch
from tensordict import TensorDict

from rl_training.policy_symmetry import mirror_action, mirror_observation, augment_lite3_mirror
from rl_training.lite3_joint_conventions import clip_lite3_policy_action, physical_targets_from_action


def test_reflection_involution_and_physical_targets():
    torch.manual_seed(17)
    action = torch.randn(64, 12) * 5
    torch.testing.assert_close(mirror_action(mirror_action(action)), action, atol=0, rtol=0)
    torch.testing.assert_close(mirror_action(clip_lite3_policy_action(action)), clip_lite3_policy_action(mirror_action(action)))
    torch.testing.assert_close(physical_targets_from_action(mirror_action(action)), mirror_action(physical_targets_from_action(action)))
    for width in (45, 48):
        obs = torch.randn(64, width)
        torch.testing.assert_close(mirror_observation(mirror_observation(obs)), obs, atol=0, rtol=0)
    # Reflection reverses lateral polar velocity, but reverses roll/yaw axial velocity.
    critic = torch.ones(1, 48)
    torch.testing.assert_close(mirror_observation(critic)[0, :9], torch.tensor([1.,-1,1,-1,1,-1,1,-1,1]))


def test_actual_rsl_mirror_loss_update():
    from rsl_rl.models import MLPModel
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    torch.manual_seed(23)
    obs = TensorDict({'policy': torch.randn(8,45), 'critic': torch.randn(8,48)}, batch_size=[8])
    saved = obs.clone()
    augmented, actions = augment_lite3_mirror(obs, torch.randn(8,12))
    assert augmented.batch_size == torch.Size([16]) and actions.shape == (16,12)
    torch.testing.assert_close(obs['policy'], saved['policy'])
    groups = {'actor':['policy'], 'critic':['critic']}
    actor = MLPModel(obs,groups,'actor',12,hidden_dims=[32],distribution_cfg={'class_name':'GaussianDistribution','init_std':.2,'std_type':'log'})
    critic = MLPModel(obs,groups,'critic',1,hidden_dims=[32])
    storage = RolloutStorage('rl',8,4,obs,[12])
    ppo = PPO(actor,critic,storage,num_learning_epochs=1,num_mini_batches=1,schedule='fixed',
              symmetry_cfg={'use_data_augmentation':False,'use_mirror_loss':True,'mirror_loss_coeff':1.,
                            'data_augmentation_func':augment_lite3_mirror,'_env':None})
    original = {k:v.clone() for k,v in actor.state_dict().items()}
    with torch.inference_mode():
        for _ in range(4):
            ppo.act(obs)
            ppo.process_env_step(obs,torch.randn(8),torch.zeros(8,dtype=torch.bool),{})
        ppo.compute_returns(obs)
    losses = ppo.update()
    assert losses['symmetry'] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert any(not torch.equal(v,actor.state_dict()[k]) for k,v in original.items())
