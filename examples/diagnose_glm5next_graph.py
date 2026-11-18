# SPDX-License-Identifier: Apache-2.0
"""Read serving bindings and optionally dump explicitly selected worker PIDs."""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import shutil
import subprocess
from pathlib import Path


def _binding(value):
    try:
        source = inspect.getfile(value)
    except (TypeError, OSError):
        source = None
    return {"module": getattr(value, "__module__", None), "name": getattr(value, "__name__", None), "source": source}


def _git(root):
    result = {}
    for key, args in (("HEAD", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        proc = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, timeout=15)
        result[key] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    return result


def collect(pids):
    import vllm

    import vllm_ascend

    root = Path(vllm_ascend.__file__).resolve().parent.parent
    bindings, errors = {}, {}
    # Reports from deployed containers also use a reorganized ops/layers tree.
    # Describe the installed modules instead of assuming that HEAD identifies
    # an unmodified runtime, or that one failed optional import selects GLM's op.
    for name in (
        "vllm_ascend.models.glm5next.kda",
        "vllm_ascend.ops.layers.attention.kda",
        "vllm_ascend.models.glm5next.ops.causal_conv1d",
        "vllm_ascend.ops.causal_conv1d",
    ):
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            lines = path.read_text(encoding="utf-8").splitlines()
            bindings[name] = {
                "source": str(path),
                "callables": {
                    attr: _binding(getattr(module, attr))
                    for attr in ("causal_conv1d", "causal_conv1d_fn", "causal_conv1d_update")
                    if hasattr(module, attr)
                },
                "convolution_source_lines": [
                    f"{i}: {line}" for i, line in enumerate(lines, 1) if "causal_conv1d" in line
                ],
            }
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    versions = {}
    for package in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend", "flash-linear-attention-npu-a2"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "versions": versions,
        "vllm_root": str(Path(vllm.__file__).resolve().parent.parent),
        "ascend_root": str(root),
        "git": _git(root),
        "bindings": bindings,
        "binding_probe_errors": errors,
        "note": (
            "Fresh-interpreter source/binding inspection: no model loaded or graph captured; "
            "live Worker stacks are separate."
        ),
        "workers": {},
    }
    spy = shutil.which("py-spy")
    for pid in pids:
        worker = {}
        for name in ("comm", "wchan", "stack"):
            try:
                worker[name] = Path(f"/proc/{pid}/{name}").read_text().strip()
            except OSError as error:
                worker[name] = str(error)
        if spy:
            try:
                proc = subprocess.run([spy, "dump", "--pid", str(pid)], text=True, capture_output=True, timeout=20)
                worker["py_spy"] = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
            except subprocess.TimeoutExpired:
                worker["py_spy"] = "Timed out after 20 seconds"
        else:
            worker["py_spy"] = "py-spy is not installed; /proc/wchan is not a Python stack"
        result["workers"][str(pid)] = worker
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pids", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(pid <= 0 for pid in args.pids):
        parser.error("PIDs must be positive and belong to the user's serving container")
    result = collect(args.pids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
