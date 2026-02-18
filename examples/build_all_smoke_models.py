"""Build ONNX, sklearn, PyTorch, and TensorFlow smoke models into one folder."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from build_pytorch_and_tensorflow_smoke_models import (
    _build_pytorch_model,
    _build_tensorflow_model,
)
from build_sklearn_and_onnx_smoke_models import _train_model
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main() -> None:
    """Write model artifacts required for all benchmark services."""
    output_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/onnx-runner-comparison")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sklearn_model = _train_model()
    joblib.dump(sklearn_model, output_dir / "model.joblib")
    onnx_model = convert_sklearn(
        sklearn_model,
        initial_types=[("x", FloatTensorType([None, 3]))],
        target_opset=14,
    )
    (output_dir / "model.onnx").write_bytes(onnx_model.SerializeToString())

    _build_pytorch_model(output_dir / "model.pt")
    _build_tensorflow_model(output_dir / "model.keras")
    print(f"wrote smoke model artifacts to {output_dir}")


if __name__ == "__main__":
    main()
