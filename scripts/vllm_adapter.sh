#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
PATCH_DIR="${PATCH_DIR:-${REPO_ROOT}/moss_ocr/static/vllm_patches/v${VLLM_VERSION}/vllm}"

if [[ ! -d "${PATCH_DIR}" ]]; then
  echo "Patch directory not found: ${PATCH_DIR}" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo " Please creating virtualenv by `uv venv .venv` first" >&2
  exit 1
fi

PY="${VENV_DIR}/bin/python"
"${PY}" -m pip install --upgrade pip

PIP="${VENV_DIR}/bin/pip"
# "${PIP}" install "vllm==${VLLM_VERSION}"

PATCH_DIR="${PATCH_DIR}" "${PY}" - <<'PY'
import importlib.util
import os
import shutil
from pathlib import Path

patch_dir = Path(os.environ["PATCH_DIR"]).resolve()

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Cannot find installed vllm package")

target_root = Path(spec.submodule_search_locations[0]).resolve()

for source_file in patch_dir.rglob("*"):
    if not source_file.is_file():
        continue
    target_file = target_root / source_file.relative_to(patch_dir)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target_file)
    print(f"patched {target_file}")
PY

"${PY}" - <<'PY'
from vllm.transformers_utils.configs.moss_v1d5 import MOSSv1d5Config
from vllm.transformers_utils.configs.moss_v1d6 import MOSSv1d6Config
from vllm.model_executor.models import moss_v1d5, moss_v1d6

print("MOSS patch import check ok:", MOSSv1d5Config.__name__, MOSSv1d6Config.__name__)
print("MOSS model modules ok:", moss_v1d5.__name__, moss_v1d6.__name__)
PY

echo
echo "Done."