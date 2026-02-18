"""gRPC server exposing ONNX inference endpoints."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol, cast

import grpc

from base_adapter import BaseAdapter, load_adapter
from config import Settings
from input_output import format_output, parse_payload
from parsed_types import batch_size as _batch_size

log = logging.getLogger("byoc.grpc")


@dataclass
class _FallbackStatusReply:
    """Fallback status reply used when generated stubs are unavailable."""

    ok: bool = False
    status: str = ""


@dataclass
class _FallbackPredictReply:
    """Fallback predict reply used when generated stubs are unavailable."""

    body: bytes = b""
    content_type: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def _load_grpc_stub_module(module_name: str) -> ModuleType | None:
    """Load grpc generated module by name when present."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


_INFERENCE_PB2 = _load_grpc_stub_module("onnx_serving_grpc.inference_pb2")
_INFERENCE_PB2_GRPC = _load_grpc_stub_module("onnx_serving_grpc.inference_pb2_grpc")


def _status_reply(*, ok: bool, status: str) -> object:
    """Create status reply from generated stubs or fallback type."""
    if _INFERENCE_PB2 is None:
        return _FallbackStatusReply(ok=ok, status=status)
    return _INFERENCE_PB2.StatusReply(ok=ok, status=status)


def _predict_reply(
    *,
    body: bytes = b"",
    content_type: str = "",
    metadata: dict[str, str] | None = None,
) -> object:
    """Create predict reply from generated stubs or fallback type."""
    if _INFERENCE_PB2 is None:
        return _FallbackPredictReply(
            body=body,
            content_type=content_type,
            metadata=metadata or {},
        )
    return _INFERENCE_PB2.PredictReply(
        body=body,
        content_type=content_type,
        metadata=metadata or {},
    )


def _require_grpc_stubs_module() -> ModuleType:
    """Return generated grpc service module or raise actionable error."""
    if _INFERENCE_PB2_GRPC is None:
        raise RuntimeError(
            "gRPC stubs are missing. Run ./scripts/generate_grpc_stubs.sh first."
        )
    return _INFERENCE_PB2_GRPC


class _PredictRequestLike(Protocol):
    """Protocol for predict request object shape."""

    payload: bytes
    content_type: str
    accept: str


def _set_grpc_error(
    context: grpc.ServicerContext | grpc.aio.ServicerContext | None,
    code: grpc.StatusCode,
    details: str,
) -> None:
    """Safely set error code/details when context is available."""
    if context is None:
        return
    context.set_code(code)
    context.set_details(details)


class InferenceGrpcService:
    """gRPC inference service implementation."""

    def __init__(self: InferenceGrpcService, settings: Settings) -> None:
        """Initialize service and eagerly load model adapter.

        Parameters
        ----------
        settings : Settings
            Runtime configuration.
        """
        self.settings = settings
        self.adapter: BaseAdapter = load_adapter(settings)

    async def live(
        self: InferenceGrpcService,
        request: object,
        context: grpc.ServicerContext | grpc.aio.ServicerContext,
    ) -> object:
        """Report process liveness."""
        _ = (request, context)
        return _status_reply(ok=True, status="live")

    async def ready(
        self: InferenceGrpcService,
        request: object,
        context: grpc.ServicerContext | grpc.aio.ServicerContext,
    ) -> object:
        """Report model readiness."""
        _ = (request, context)
        is_ready = bool(self.adapter.is_ready())
        return _status_reply(
            ok=is_ready,
            status="ready" if is_ready else "not_ready",
        )

    async def predict(
        self: InferenceGrpcService,
        request: object,
        context: grpc.ServicerContext | grpc.aio.ServicerContext | None,
    ) -> object:
        """Run model prediction for a payload."""
        predict_request = cast(_PredictRequestLike, request)
        content_type = (
            predict_request.content_type or self.settings.default_content_type
        )
        accept = predict_request.accept or self.settings.default_accept
        payload = bytes(predict_request.payload)

        try:
            parsed_input = parse_payload(
                payload,
                content_type=content_type,
                settings=self.settings,
            )
            batch_size = _batch_size(parsed_input)
            if batch_size > self.settings.max_records:
                raise ValueError(
                    f"too_many_records: {batch_size} > {self.settings.max_records}"
                )
            predictions = self.adapter.predict(parsed_input)
            body, output_content_type = format_output(
                predictions,
                accept=accept,
                settings=self.settings,
            )
            body_bytes = body if isinstance(body, bytes) else body.encode("utf-8")
            return _predict_reply(
                body=body_bytes,
                content_type=output_content_type,
                metadata={"batch_size": str(batch_size)},
            )
        except ValueError as exc:
            _set_grpc_error(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return _predict_reply()
        except Exception as exc:  # pragma: no cover - unexpected adapter/runtime errors
            log.exception("Predict RPC failed")
            _set_grpc_error(context, grpc.StatusCode.INTERNAL, str(exc))
            return _predict_reply()

    # gRPC generated service wiring expects these exact method names.
    Live = live
    Ready = ready
    Predict = predict


def create_server(
    settings: Settings,
    *,
    host: str,
    port: int,
    max_workers: int = 16,
) -> grpc.aio.Server:
    """Create configured gRPC server instance.

    Parameters
    ----------
    settings : Settings
        Runtime settings.
    host : str
        Bind host.
    port : int
        Bind TCP port.
    max_workers : int, default=16
        Maximum concurrent RPCs.

    Returns
    -------
    grpc.aio.Server
        Configured server.
    """
    server = grpc.aio.server(maximum_concurrent_rpcs=max_workers)
    grpc_stubs = _require_grpc_stubs_module()
    grpc_stubs.add_InferenceServiceServicer_to_server(
        InferenceGrpcService(settings),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    return server


async def _serve(
    *,
    settings: Settings,
    host: str,
    port: int,
    max_workers: int,
) -> None:
    """Start async gRPC server and block until termination."""
    server = create_server(
        settings,
        host=host,
        port=port,
        max_workers=max_workers,
    )
    await server.start()
    log.info("gRPC inference server listening on %s:%s", host, port)
    await server.wait_for_termination()


def main() -> None:
    """Run gRPC inference server entrypoint."""
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="ONNX model serving gRPC endpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    asyncio.run(
        _serve(
            settings=settings,
            host=cast(str, args.host),
            port=cast(int, args.port),
            max_workers=cast(int, args.max_workers),
        )
    )


if __name__ == "__main__":
    main()
