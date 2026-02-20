"""Base adapter contracts and adapter factory."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from config import Settings
from parsed_types import ParsedInput
from value_types import PredictionValue


class BaseAdapter(ABC):
    """Abstract contract for inference adapters."""

    @abstractmethod
    def is_ready(self: BaseAdapter) -> bool:
        """Return whether the adapter is ready to serve predictions.

        Returns
        -------
        bool
            ``True`` when the adapter is fully initialized.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self: BaseAdapter, parsed_input: ParsedInput) -> PredictionValue:
        """Run inference for a parsed input payload.

        Parameters
        ----------
        parsed_input : ParsedInput
            Parsed model input.

        Returns
        -------
        PredictionValue
            Model prediction output.
        """
        raise NotImplementedError


def load_adapter(settings: Settings) -> BaseAdapter:
    """Instantiate the appropriate adapter for current settings.

    Parameters
    ----------
    settings : Settings
        Runtime configuration.

    Returns
    -------
    BaseAdapter
        Instantiated inference adapter.
    """
    from onnx_adapter import OnnxAdapter
    from pytorch_adapter import PytorchAdapter
    from sklearn_adapter import SklearnAdapter
    from tensorflow_adapter import TensorflowAdapter

    model_filename = settings.model_filename.strip()

    onnx_default = os.path.join(settings.model_dir, "model.onnx")
    sklearn_default = os.path.join(settings.model_dir, "model.joblib")
    pickle_default = os.path.join(settings.model_dir, "model.pkl")
    pytorch_default = os.path.join(settings.model_dir, "model.pt")
    tensorflow_default = os.path.join(settings.model_dir, "model.keras")
    tensorflow_saved_model_default = os.path.join(settings.model_dir, "saved_model")

    if settings.model_type == "onnx":
        return OnnxAdapter(settings)
    if settings.model_type == "sklearn":
        return SklearnAdapter(settings)
    if settings.model_type == "pytorch":
        return PytorchAdapter(settings)
    if settings.model_type == "tensorflow":
        return TensorflowAdapter(settings)

    if settings.model_type and settings.model_type not in {
        "onnx",
        "sklearn",
        "pytorch",
        "tensorflow",
    }:
        raise RuntimeError(
            f"MODEL_TYPE={settings.model_type} is not implemented in this package"
        )

    if model_filename:
        model_path = os.path.join(settings.model_dir, model_filename)
        lowered = model_filename.lower()
        if lowered.endswith(".onnx") and os.path.exists(model_path):
            return OnnxAdapter(settings)
        if lowered.endswith((".joblib", ".pkl")) and (os.path.exists(model_path)):
            return SklearnAdapter(settings)
        if lowered.endswith((".pt", ".pth", ".jit", ".torchscript")) and (
            os.path.exists(model_path)
        ):
            return PytorchAdapter(settings)
        if lowered.endswith((".keras", ".h5", ".hdf5")) and os.path.exists(model_path):
            return TensorflowAdapter(settings)
        if os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "saved_model.pb")
        ):
            return TensorflowAdapter(settings)
    else:
        if os.path.exists(onnx_default):
            return OnnxAdapter(settings)
        if os.path.exists(sklearn_default) or os.path.exists(pickle_default):
            return SklearnAdapter(settings)
        if os.path.exists(pytorch_default):
            return PytorchAdapter(settings)
        if os.path.exists(tensorflow_default) or os.path.exists(
            tensorflow_saved_model_default
        ):
            return TensorflowAdapter(settings)

    raise RuntimeError(
        "Set MODEL_TYPE=onnx|sklearn|pytorch|tensorflow or place a known model "
        "artifact under SM_MODEL_DIR"
    )
