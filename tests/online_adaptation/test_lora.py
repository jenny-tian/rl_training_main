import copy

import torch
from torch import nn

from rl_training.online_adaptation.lora import (
    LoRALinear,
    inject_lora,
    lora_parameter_count,
    mark_only_lora_trainable,
    merge_lora,
    unmerge_lora,
)


def test_zero_initialized_adapter_matches_source_on_real_inputs_and_freezes_base():
    torch.manual_seed(4)
    source = nn.Sequential(nn.Linear(3, 5), nn.ELU(), nn.Linear(5, 2))
    inputs = torch.randn(7, 3)
    original = copy.deepcopy(source)
    adapters = inject_lora(source, rank=2)
    torch.testing.assert_close(source(inputs), original(inputs), rtol=0, atol=0)
    torch.testing.assert_close(adapters[0](inputs), adapters[0].base(inputs), rtol=0, atol=0)
    hidden_inputs = source[0](inputs)
    torch.testing.assert_close(adapters[1](hidden_inputs), adapters[1].base(hidden_inputs), rtol=0, atol=0)
    for adapter in adapters:
        assert not adapter.base.weight.requires_grad
        assert not adapter.base.bias.requires_grad


def test_only_adapter_receives_gradient_and_merge_round_trip_is_equivalent():
    torch.manual_seed(8)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    inject_lora(model, rank=2)
    mark_only_lora_trainable(model)
    x = torch.randn(9, 3)
    for adapter in [m for m in model.modules() if isinstance(m, LoRALinear)]:
        adapter.lora_B.data.normal_(std=0.1)
    before = model(x).detach()
    model.zero_grad()
    model(x).square().mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    merge_lora(model)
    torch.testing.assert_close(model(x), before, rtol=1e-5, atol=1e-6)
    unmerge_lora(model)
    torch.testing.assert_close(model(x), before, rtol=1e-5, atol=1e-6)


def test_state_dict_round_trip_preserves_merged_state_and_parameter_count():
    model = nn.Sequential(nn.Linear(6, 3))
    inject_lora(model, rank=3)
    assert lora_parameter_count(model) == 3 * (6 + 3)
    next(iter(m for m in model.modules() if isinstance(m, LoRALinear))).lora_B.data.fill_(0.02)
    merge_lora(model)
    state = copy.deepcopy(model.state_dict())
    restored = nn.Sequential(nn.Linear(6, 3))
    inject_lora(restored, rank=3)
    restored.load_state_dict(state)
    assert next(m for m in restored.modules() if isinstance(m, LoRALinear)).merged
    inputs = torch.randn(4, 6)
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)


def test_injected_model_loads_original_linear_state_dict_strictly():
    torch.manual_seed(12)
    original = nn.Sequential(nn.Linear(4, 3), nn.ELU(), nn.Linear(3, 2))
    adapted = copy.deepcopy(original)
    inject_lora(adapted, rank=2)
    adapted.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(5, 4)
    torch.testing.assert_close(adapted(inputs), original(inputs), rtol=0, atol=0)
