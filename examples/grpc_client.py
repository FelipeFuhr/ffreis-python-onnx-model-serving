"""Minimal gRPC client for ONNX serving Predict endpoint."""

from __future__ import annotations

import json
import os

import grpc

from onnx_serving_grpc import inference_pb2, inference_pb2_grpc


def main() -> None:
    """Run one health and predict request against gRPC server."""
    target = os.getenv("GRPC_TARGET", "127.0.0.1:50052")
    channel = grpc.insecure_channel(target)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    try:
        live = stub.Live(inference_pb2.LiveRequest())
        ready = stub.Ready(inference_pb2.ReadyRequest())
        print(f"live={live.ok} ready={ready.ok}")

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0, 4.0]]}).encode("utf-8")
        result = stub.Predict(
            inference_pb2.PredictRequest(
                payload=payload,
                content_type="application/json",
                accept="application/json",
            )
        )
        print(result.body.decode("utf-8"))
    finally:
        channel.close()


if __name__ == "__main__":
    main()
