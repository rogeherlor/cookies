"""
model_factory.py — matches external/tlio/src/network/model_factory.py interface.

Credit: https://github.com/CathIAS/TLIO (MIT License)

Loaded via importlib in tlio_runner.py / train_tlio.py to bypass
external/tlio/src/network/__init__.py's transitive imports.
"""

import importlib.util
from pathlib import Path

_resnet_path = Path(__file__).resolve().parent / 'model_resnet.py'
_spec = importlib.util.spec_from_file_location('_tlio_model_resnet', _resnet_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BasicBlock1D = _mod.BasicBlock1D
ResNet1D     = _mod.ResNet1D


def get_model(arch: str, net_config: dict, input_dim: int = 6, output_dim: int = 3):
    if arch == 'resnet':
        return ResNet1D(
            BasicBlock1D, input_dim, output_dim,
            [2, 2, 2, 2],
            net_config['in_dim'],
        )
    raise ValueError(f"Unknown architecture: {arch!r}. Supported: 'resnet'")
