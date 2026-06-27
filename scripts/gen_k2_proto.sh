#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT/backend/asr/k2"

uv run --extra dev python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PROTO_DIR" \
  --grpc_python_out="$PROTO_DIR" \
  "$PROTO_DIR/asr.proto"

python - "$PROTO_DIR/asr_pb2_grpc.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text = text.replace("import asr_pb2 as asr__pb2", "from . import asr_pb2 as asr__pb2")
path.write_text(text, encoding="utf-8")
PY
