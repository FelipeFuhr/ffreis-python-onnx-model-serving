"""Build paired PyTorch and TensorFlow smoke models with matching linear behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Self

import numpy as np
import tensorflow as tf
import torch


def _build_pytorch_model(path: Path) -> None:
    """Write TorchScript model computing y = x @ 1."""

    class SumModel(torch.nn.Module):
        def __init__(self: Self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.float())

    model = SumModel().eval()
    scripted = torch.jit.script(model)
    scripted.save(str(path))


def _build_tensorflow_model(path: Path) -> None:
    """Write Keras model computing y = x @ 1."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(3,)),
            tf.keras.layers.Dense(1, use_bias=False),
        ]
    )
    model.layers[0].set_weights([np.ones((3, 1), dtype=np.float32)])
    model.save(path)


def main() -> None:
    """Write PyTorch and TensorFlow smoke model artifacts."""
    output_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/onnx-runner-comparison")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pytorch_path = output_dir / "model.pt"
    tensorflow_path = output_dir / "model.keras"

    _build_pytorch_model(pytorch_path)
    _build_tensorflow_model(tensorflow_path)
    print(f"wrote PyTorch model to {pytorch_path}")
    print(f"wrote TensorFlow model to {tensorflow_path}")


if __name__ == "__main__":
    main()
