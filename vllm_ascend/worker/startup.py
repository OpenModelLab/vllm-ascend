# SPDX-License-Identifier: Apache-2.0
"""Host-only phase timing for worker startup; no device synchronization."""

import os
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from vllm.logger import logger

P = ParamSpec("P")
R = TypeVar("R")


def log_worker_startup(stage: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Identify the worker phase that has not returned when readiness stalls."""

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            worker = args[0]
            config = getattr(worker, "vllm_config", None)
            compilation = getattr(config, "compilation_config", None)
            graph_mode = getattr(compilation, "cudagraph_mode", None)
            rank = getattr(worker, "rank", "unknown")
            started = time.perf_counter()
            logger.info(
                "Ascend startup rank=%s pid=%d stage=%s begin graph=%s",
                rank,
                os.getpid(),
                stage,
                graph_mode,
            )
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception("Ascend startup rank=%s stage=%s failed", rank, stage)
                raise
            logger.info("Ascend startup rank=%s stage=%s end elapsed=%.3fs", rank, stage, time.perf_counter() - started)
            return result

        return wrapped

    return decorate
