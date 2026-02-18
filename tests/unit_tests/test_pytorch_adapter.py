"""Unit tests for pytorch adapter."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from config import Settings
from parsed_types import ParsedInput
from pytorch_adapter import PytorchAdapter

pytestmark = pytest.mark.unit


class _NoGrad:
    """Minimal context manager used by fake torch module."""

    def __enter__(self: Self) -> _NoGrad:
        return self

    def __exit__(
        self: Self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object | None,
    ) -> bool:
        return False


class _FakeTensor:
    """Small tensor-like wrapper for adapter conversion tests."""

    def __init__(self: Self, array: np.ndarray) -> None:
        self._array = array
        self.dtype = types.SimpleNamespace(
            is_floating_point=bool(np.issubdtype(array.dtype, np.floating))
        )

    def to(self: Self, dtype: object) -> _FakeTensor:
        _ = dtype
        return _FakeTensor(self._array.astype(np.float32, copy=False))

    def detach(self: Self) -> _FakeTensor:
        return self

    def cpu(self: Self) -> _FakeTensor:
        return self

    def numpy(self: Self) -> np.ndarray:
        return self._array


class _FakeModel:
    """Callable model returning sum over features."""

    def eval(self: Self) -> None:
        return None

    def __call__(self: Self, tensor: _FakeTensor) -> _FakeTensor:
        arr = tensor.numpy().astype(np.float32)
        summed = arr.sum(axis=1, keepdims=True)
        return _FakeTensor(summed)


def _fake_torch_module() -> object:
    """Build a fake torch module object."""
    return types.SimpleNamespace(
        Tensor=_FakeTensor,
        float32="float32",
        jit=types.SimpleNamespace(load=lambda _path, map_location=None: _FakeModel()),
        load=lambda _path, map_location=None: _FakeModel(),
        as_tensor=lambda array: _FakeTensor(np.asarray(array)),
        no_grad=lambda: _NoGrad(),
    )


class TestPytorchAdapter:
    """Test suite for pytorch adapter behavior."""

    def test_predicts_from_tabular_input(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Predict from ParsedInput.X with fake torch module."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "torch":
                return _fake_torch_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        adapter = PytorchAdapter(Settings())

        result = adapter.predict(
            ParsedInput(X=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        )
        assert result == [[6.0], [15.0]]

    def test_predict_raises_for_empty_payload(
        self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raise validation error when neither X nor tensors are provided."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        def _fake_import_module(name: str) -> object:
            if name == "torch":
                return _fake_torch_module()
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("importlib.import_module", _fake_import_module)
        monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("MODEL_TYPE", "pytorch")
        adapter = PytorchAdapter(Settings())

        with pytest.raises(
            ValueError, match="requires ParsedInput.X or ParsedInput.tensors"
        ):
            adapter.predict(ParsedInput())
