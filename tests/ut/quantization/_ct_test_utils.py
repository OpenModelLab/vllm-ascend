# SPDX-License-Identifier: Apache-2.0
"""Execute selected source definitions while isolating unavailable runtimes."""

import ast
import sys
import types


def execute_nodes(path, scope, *, classes=None, functions=()):
    selected = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            base, methods = classes[node.name]
            node.decorator_list = []
            node.bases = [ast.Name(id=base, ctx=ast.Load())] if base else []
            if methods is not None:
                node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
            selected.append(node)
    if len(selected) != len(functions) + len(classes or {}):
        raise AssertionError(f"Expected source definitions not found in {path}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), scope)


def module(name, **attributes):
    value = types.ModuleType(name)
    value.__dict__.update(attributes)
    sys.modules[name] = value
    return value
