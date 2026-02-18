"""Unit tests for tensorflow adapter."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from config import Settings
from parsed_types import ParsedInput
from tensorflow_adapter import TensorflowAdapter

pytestmark = pytest.mark.unit


class _FakeTensorflowModel:
    """Fake keras model for adapter tests."""

    def __call__(
        self: Self, features: np.ndarray, training: bool = False
    ) -> np.ndarray:
        _ = training
        return np.sum(features, axis=1, keepdims=True).astype(np.float32)


def _fake_tensorflow_module() -> object:
    """Build fake tensorflow module with keras load_model."""
    keras_models = types.SimpleNamespace(
        load_model=lambda _path: _FakeTensorflowModel()
    )
    keras = types.SimpleNamespace(models=keras_models)
    return types.SimpleNamespace(keras=keras)


class TestTensorflowAdapter:
    """Test suite for tensorflow adapter behavior."""

    def test_predicts_from_tabular_input(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Predict from ParsedInput.X with fake tensorflow module."""
        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "tensorflow":
                return _fake_tensorflow_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        adapter = TensorflowAdapter(Settings())

        result = adapter.predict(
            ParsedInput(X=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        )
        assert result == [[6.0], [15.0]]

    def test_predict_raises_for_empty_payload(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise validation error when neither X nor tensors are provided."""
        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "tensorflow":
                return _fake_tensorflow_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "tensorflow")
        adapter = TensorflowAdapter(Settings())

        with pytest.raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())
