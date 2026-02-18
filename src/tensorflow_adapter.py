"""TensorFlow adapter implementation."""

from __future__ import annotations

import importlib
import os
from typing import Protocol, Self, cast

import numpy as np

from base_adapter import BaseAdapter
from config import Settings
from parsed_types import ParsedInput


class _TensorflowModel(Protocol):
    def __call__(self: Self, features: np.ndarray, training: bool = False) -> object:
        """Run callable inference path."""

    def predict(self: Self, features: np.ndarray, verbose: int = 0) -> object:
        """Run predict-style inference path."""


class _KerasModels(Protocol):
    def load_model(self: Self, path: str) -> _TensorflowModel:
        """Load TensorFlow/Keras model."""


class _KerasModule(Protocol):
    models: _KerasModels


class _TensorflowModule(Protocol):
    keras: _KerasModule


class TensorflowAdapter(BaseAdapter):
    """Inference adapter backed by a TensorFlow/Keras model."""

    def __init__(self: Self, settings: Settings) -> None:
        """Load tensorflow model from disk."""
        self.settings = settings
        model_filename = settings.model_filename.strip() or "model.keras"
        model_path = os.path.join(settings.model_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TensorFlow model not found at: {model_path}")

        self._tensorflow = cast(
            _TensorflowModule, importlib.import_module("tensorflow")
        )
        keras_models = self._tensorflow.keras.models
        self.model = keras_models.load_model(model_path)

    def is_ready(self: Self) -> bool:
        """Report whether model is loaded."""
        return self.model is not None

    def _extract_features(self: Self, parsed_input: ParsedInput) -> np.ndarray:
        """Extract model input matrix from parsed input."""
        if parsed_input.X is not None:
            features = parsed_input.X
        elif parsed_input.tensors:
            features = next(iter(parsed_input.tensors.values()))
        else:
            raise ValueError(
                "TensorFlow adapter requires ParsedInput.X or ParsedInput.tensors"
            )
        array = np.asarray(features)
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array

    def _to_python(self: Self, value: object) -> object:
        """Recursively convert framework outputs to JSON-serializable shapes."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return [self._to_python(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_python(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self._to_python(v) for k, v in value.items()}
        if hasattr(value, "numpy"):
            numpy_value = value.numpy()
            if isinstance(numpy_value, np.ndarray):
                return numpy_value.tolist()
            return numpy_value
        if isinstance(value, np.generic):
            return value.item()
        return value

    def predict(self: Self, parsed_input: object) -> object:
        """Run prediction with tensorflow model."""
        if not isinstance(parsed_input, ParsedInput):
            raise TypeError("TensorFlow adapter expects ParsedInput")

        features = self._extract_features(parsed_input).astype(np.float32, copy=False)

        if callable(self.model):
            output = self.model(features, training=False)
        else:
            output = self.model.predict(features, verbose=0)
        return self._to_python(output)
