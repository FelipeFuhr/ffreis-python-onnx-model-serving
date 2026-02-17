# gRPC Interface

The serving package now exposes gRPC endpoints in parallel to HTTP.

HTTP health endpoints are standardized as:

- `GET /healthz` (alias of `/live`)
- `GET /readyz` (alias of `/ready`)

## Install

```bash
uv sync --extra grpc
```

## Run server

```bash
uv run --extra grpc onnx-model-serving-grpc --host 0.0.0.0 --port 50052
```

## API surface

- `Live` -> process liveness
- `Ready` -> model readiness
- `Predict` -> payload inference

`Predict` fields:

- `payload` (`bytes`)
- `content_type` (defaults to configured HTTP default)
- `accept` (defaults to configured HTTP default)

Proto contract:

- `proto/onnx_serving_grpc/inference.proto`

Generated modules:

- `src/onnx_serving_grpc/inference_pb2.py`
- `src/onnx_serving_grpc/inference_pb2_grpc.py`

## Sync checks

Regenerate:

```bash
make grpc-generate
```

Verify generated files are in sync with proto:

```bash
make grpc-check
```

Run API + gRPC compose smoke:

```bash
make smoke-api-grpc
```
