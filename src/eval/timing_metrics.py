"""Inference timing, from the counters Ollama returns with every reply.

Ollama reports ``total_duration``, ``load_duration``, ``prompt_eval_duration``
and ``eval_duration`` in NANOSECONDS, plus ``prompt_eval_count`` /
``eval_count`` token counts. :meth:`src.extract.client.ModelReply.meta` keeps
them on every extraction result (``model_meta``), cached results included, so
a cache hit still carries the timing of the inference that produced it.

Medians, not means: a cold model load (216s on the reference machine) lands in
``load_duration`` of one call and would drag a mean far from the typical call.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable

NS = 1e9


def _seconds(meta: dict[str, Any], key: str) -> float | None:
    v = meta.get(key)
    return float(v) / NS if isinstance(v, (int, float)) else None


def _rate(meta: dict[str, Any], count_key: str, dur_key: str) -> float | None:
    n, d = meta.get(count_key), meta.get(dur_key)
    if isinstance(n, (int, float)) and isinstance(d, (int, float)) and d > 0:
        return float(n) / (float(d) / NS)
    return None


def _med(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return median(vals) if vals else None


def summarise_timing(results: Iterable[Any]) -> dict[str, Any]:
    """Median timings over extraction results that carry Ollama counters."""
    results = list(results)
    metas = [r.model_meta for r in results if getattr(r, "model_meta", None)]
    n_cached = sum(1 for r in results if getattr(r, "from_cache", False))
    return {
        "n_calls": len(metas),
        "n_from_cache": n_cached,
        "median_total_s": _med([_seconds(m, "total_duration") for m in metas]),
        "median_load_s": _med([_seconds(m, "load_duration") for m in metas]),
        "median_prompt_eval_s": _med(
            [_seconds(m, "prompt_eval_duration") for m in metas]),
        "median_eval_s": _med([_seconds(m, "eval_duration") for m in metas]),
        "median_prompt_tokens": _med(
            [float(m["prompt_eval_count"]) for m in metas
             if isinstance(m.get("prompt_eval_count"), (int, float))]),
        "median_eval_tokens": _med(
            [float(m["eval_count"]) for m in metas
             if isinstance(m.get("eval_count"), (int, float))]),
        "median_prompt_tok_per_s": _med(
            [_rate(m, "prompt_eval_count", "prompt_eval_duration") for m in metas]),
        "median_eval_tok_per_s": _med(
            [_rate(m, "eval_count", "eval_duration") for m in metas]),
        "max_total_s": max(
            (v for v in (_seconds(m, "total_duration") for m in metas)
             if v is not None), default=None),
    }
