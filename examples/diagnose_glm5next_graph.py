# SPDX-License-Identifier: Apache-2.0
"""Read serving bindings and optionally dump explicitly selected worker PIDs."""

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path


def _source_report(path):
    """Read Python text only; findings are candidates, not executed call paths."""
    data = path.read_bytes()
    source = data.decode("utf-8-sig")
    tree = ast.parse(source)
    selected = {
        "npu_causal_conv1d_update",
        "causal_conv1d_update",
        "npu_recurrent_kda",
        "_get_direct_op",
        "_get_stable_op",
        "_make_raw_wrapper",
        "_current_stream_ptr",
        "_causal_conv1d_metadata_values_for_validation",
    }
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in selected:
            functions[node.name] = {"line": node.lineno, "source": ast.get_source_segment(source, node)}
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "functions": functions,
        "possible_host_reads": [
            f"{i}: {line.strip()}"
            for i, line in enumerate(source.splitlines(), 1)
            if any(token in line for token in (".cpu(", ".item(", ".tolist(", "synchronize("))
        ],
    }


def collect_fla_sources():
    """Inspect installed wheel metadata without importing torch or fla_npu."""
    result = {}
    for name in (
        "flash-linear-attention-npu",
        "flash-linear-attention-npu-a2",
        "flash-linear-attention-npu-a3",
        "flash-linear-attention-npu-a5",
    ):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        sources = {}
        for entry in dist.files or ():
            relative = str(entry).replace("\\", "/")
            if "fla_npu/ops/ascendc/" not in relative or not relative.endswith(".py"):
                continue
            path = Path(dist.locate_file(entry))
            try:
                sources[relative] = _source_report(path)
            except (OSError, SyntaxError, UnicodeError) as error:
                sources[relative] = {"error": f"{type(error).__name__}: {error}"}
        result[name] = {"version": dist.version, "sources": sources}
    return result


def _binding(value):
    try:
        source = inspect.getfile(value)
    except (TypeError, OSError):
        source = None
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = None
    return {
        "module": getattr(value, "__module__", None),
        "name": getattr(value, "__name__", None),
        "source": source,
        "signature": signature,
    }


def _git(root):
    result = {}
    for key, args in (("HEAD", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        proc = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, timeout=15)
        result[key] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    return result


def collect(pids, *, source_only=False):
    root = Path(__file__).resolve().parents[1]
    bindings, errors = {}, {}
    vllm_root = None
    if not source_only:
        import vllm

        import vllm_ascend

        vllm_root = str(Path(vllm.__file__).resolve().parent.parent)
        root = Path(vllm_ascend.__file__).resolve().parent.parent
    # The corrected container tree uses models/glm5next. Older reports of a
    # reorganized tree were withdrawn; optional alternate bindings are kept
    # for users running a different build, not assumed to be GLM's fallback.
    modules = (
        ()
        if source_only
        else (
            "vllm_ascend.models.glm5next.kda",
            "vllm_ascend.ops.layers.attention.kda",
            "vllm_ascend.models.glm5next.ops.causal_conv1d",
            "vllm_ascend.ops.causal_conv1d",
        )
    )
    for name in modules:
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
        "vllm_root": vllm_root,
        "ascend_root": str(root),
        "git": _git(root),
        "bindings": bindings,
        "binding_probe_errors": errors,
        "source_only": source_only,
        "fla_sources": collect_fla_sources(),
        "fla_backend_env_in_this_process": {
            name: os.environ.get(name)
            for name in ("FLA_NPU_STABLE_ABI", "FLA_NPU_STABLE_VALIDATE", "FLA_NPU_STABLE_TRACE")
        },
        "note": (
            "Fresh-interpreter source/binding inspection: no model loaded or graph captured; "
            "live Worker stacks and their environment are separate. Source-only mode imports no device libraries. "
            "A possible_host_reads match does not prove a device read or a call during capture."
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
    parser.add_argument(
        "--source-only", action="store_true", help="Read installed FLA Python sources without device imports"
    )
    args = parser.parse_args()
    if any(pid <= 0 for pid in args.pids):
        parser.error("PIDs must be positive and belong to the user's serving container")
    result = collect(args.pids, source_only=args.source_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
