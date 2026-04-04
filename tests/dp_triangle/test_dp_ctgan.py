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
            poisson_sampling,
        ):
            assert poisson_sampling is False
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


def test_generator_step_disables_and_restores_discriminator_hooks():
    from dp_triangle.dp_ctgan import DPCTGANSynthesizer

    class HookedDiscriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)
            self.disable_calls = 0
            self.enable_calls = 0

        def disable_hooks(self):
            self.disable_calls += 1

        def enable_hooks(self):
            self.enable_calls += 1

        def forward(self, batch):
            return self.linear(batch)

    synthesizer = DPCTGANSynthesizer(epsilon=1.0, epochs=1, batch_size=2, noise_dim=4)
    synthesizer.generator = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Tanh()).to(synthesizer.device)
    synthesizer.discriminator = HookedDiscriminator().to(synthesizer.device)

    optimizer_g = torch.optim.Adam(synthesizer.generator.parameters(), lr=2e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    real_labels = torch.ones((2, 1), device=synthesizer.device)

    synthesizer._prepare_discriminator_for_generator_step()
    try:
        noise = torch.randn(2, synthesizer.noise_dim, device=synthesizer.device)
        generated_batch = synthesizer.generator(noise)
        generated_logits = synthesizer.discriminator(generated_batch)
        g_loss = criterion(generated_logits, real_labels)
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
    finally:
        synthesizer._restore_discriminator_after_generator_step()

    assert synthesizer.discriminator.disable_calls == 1
    assert synthesizer.discriminator.enable_calls == 1
    assert all(parameter.requires_grad for parameter in synthesizer.discriminator.parameters())
