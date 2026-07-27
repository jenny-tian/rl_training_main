import json
import torch

from rl_training.online_adaptation.runtime import as_torch_device, write_failure_record


def test_as_torch_device_normalizes_isaaclab_string_device():
    device = as_torch_device("cuda:0")

    assert isinstance(device, torch.device)
    assert device.type == "cuda"
    assert device.index == 0


def test_write_failure_record_preserves_exception_details(tmp_path):
    try:
        raise AttributeError("device has no attribute type")
    except AttributeError as error:
        path = write_failure_record(tmp_path, error)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["error_type"] == "AttributeError"
    assert record["message"] == "device has no attribute type"
    assert "raise AttributeError" in record["traceback"]
