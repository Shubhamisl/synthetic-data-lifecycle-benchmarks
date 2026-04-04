from __future__ import annotations

import sys
import types

import numpy as np
import torch


def test_try_attach_privacy_rebinds_optimizer_when_validator_replaces_module(monkeypatch):
    from dp_triangle.dp_ctgan import DPCTGANSynthesizer

    synthesizer = DPCTGANSynthesizer(epsilon=1.0, epochs=1, batch_size=2, noise_dim=8)
    synthesizer.data_dim = 3
    synthesizer._build_models()

    encoded = np.zeros((4, 3), dtype=np.float32)
    loader = synthesizer._make_loader(encoded, batch_size=2)
    optimizer = torch.optim.Adam(synthesizer.discriminator.parameters(), lr=2e-4)

    class FakePrivacyEngine:
        def make_private_with_epsilon(
            self,
            module,
            optimizer,
            data_loader,
            epochs,
            target_epsilon,
            target_delta,
            max_grad_norm,
        ):
            optimizer_param_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            module_param_ids = {id(parameter) for parameter in module.parameters()}
            if optimizer_param_ids != module_param_ids:
                raise ValueError("Module parameters are different than optimizer Parameters")
            return module, optimizer, data_loader

    class FakeModuleValidator:
        @staticmethod
        def fix(module):
            replacement = type(module)(3)
            replacement.load_state_dict(module.state_dict())
            return replacement

    fake_opacus = types.ModuleType("opacus")
    fake_opacus.PrivacyEngine = FakePrivacyEngine
    fake_validators = types.ModuleType("opacus.validators")
    fake_validators.ModuleValidator = FakeModuleValidator

    monkeypatch.setitem(sys.modules, "opacus", fake_opacus)
    monkeypatch.setitem(sys.modules, "opacus.validators", fake_validators)

    private_module, private_optimizer, private_loader, privacy_engine = synthesizer._try_attach_privacy(
        optimizer,
        loader,
    )

    assert private_module is synthesizer.discriminator
    assert private_loader is loader
    assert privacy_engine is not None

    optimizer_param_ids = {
        id(parameter)
        for group in private_optimizer.param_groups
        for parameter in group["params"]
    }
    module_param_ids = {id(parameter) for parameter in private_module.parameters()}
    assert optimizer_param_ids == module_param_ids
