"""Unit tests for sklearn adapter."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import pytest

from config import Settings
from parsed_types import ParsedInput
from sklearn_adapter import SklearnAdapter

pytestmark = pytest.mark.unit


class _FakeModel:
    """Simple fake estimator exposing ``predict``."""

    def predict(self: Self, features: np.ndarray) -> np.ndarray:
        return np.sum(features, axis=1).astype(np.float64)


class TestSklearnAdapter:
    """Test suite for sklearn adapter behavior."""

    def test_loads_joblib_model(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Load model through joblib when dependency is available."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types.SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")

        adapter = SklearnAdapter(Settings())
        assert adapter.is_ready() is True

    def test_predicts_from_tabular_input(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Predict from ParsedInput.X with deterministic fake model."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types.SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())

        result = adapter.predict(
            ParsedInput(X=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        )
        assert result == [6.0, 15.0]

    def test_predict_raises_for_empty_payload(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise validation error when neither X nor tensors are provided."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types.SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())

        with pytest.raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())

    def test_predict_rejects_invalid_parsed_input(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise type error when parsed input object is invalid."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"serialized")

        def _fake_import_module(name: str) -> object:
            if name == "joblib":
                return types.SimpleNamespace(load=lambda _path: _FakeModel())
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "sklearn")
        adapter = SklearnAdapter(Settings())

        with pytest.raises(TypeError, match="Sklearn adapter expects ParsedInput"):
            adapter.predict(cast(Any, object()))
