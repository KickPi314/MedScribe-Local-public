"""Evaluation harness (spec section 8).

Phase 1 status: skeleton only. Fixture format, loader and CLI are real and
working; the metrics implemented are *preprocessing-quality proxies* computed
from the real scans. Per-field accuracy, CER and flag precision/recall are NOT
implemented -- they require extraction (phase 3+) and are reported as pending,
never as a fabricated number.
"""

from .fixtures import Fixture, FixtureError, fixture_path_for, load_fixtures, scaffold_fixture
from .metrics import PageProxyMetrics, compute_proxy_metrics

__all__ = [
    "Fixture",
    "FixtureError",
    "fixture_path_for",
    "load_fixtures",
    "scaffold_fixture",
    "PageProxyMetrics",
    "compute_proxy_metrics",
]
