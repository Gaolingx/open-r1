from types import SimpleNamespace

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning_grpo.callbacks.core import CheckpointCallback
from lightning_grpo.utils.modeling_common import export_configured_model


@pytest.mark.parametrize("use_reference", [False, True])
def test_periodic_grpo_weights_match_final_policy(tmp_path, monkeypatch, use_reference):
    module = L.LightningModule()
    module.policy = torch.nn.Linear(2, 3)
    module.reference_model = torch.nn.Linear(2, 3) if use_reference else None
    trainer = SimpleNamespace(lightning_module=module, is_global_zero=True)
    native_saves = []
    monkeypatch.setattr(ModelCheckpoint, "_save_checkpoint", lambda self, trainer, path: native_saves.append(path))
    path = str(tmp_path / "step.ckpt")
    CheckpointCallback()._save_checkpoint(trainer, path)
    assert native_saves == [path]
    config = SimpleNamespace(save_pth_format=True, save_safetensors_format=False)
    export_configured_model(module.policy, config, tmp_path / "final")
    periodic = torch.load(tmp_path / "pt_checkpoint/step.pth", weights_only=True)
    final = torch.load(tmp_path / "final/pt_checkpoint/pretrain_model.pth", weights_only=True)
    torch.nn.Linear(2, 3).load_state_dict(periodic, strict=True)
    torch.testing.assert_close(periodic, final)
