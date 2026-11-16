"""Region-segmentation proxy metrics (Phase 2).

Same honesty contract as ``src.eval.metrics``: everything here is a *proxy*
computed on real scans. There is **no ground truth for region boxes**, so this
module computes and prints no localisation accuracy, no IoU, and no field
accuracy. What it can measure, and does:

1. Anchor match confidence per page (correlation peak and peak sharpness), and
   whether each anchor cleared the configured floors.
2. Anchor-frame accuracy as a *geometric residual*: printed landmarks that the
   frame did not use, measured against the position the frame predicts for them.
   This is a real accuracy number, in pixels, about geometry only.
3. Geometric sanity per region: non-degenerate, inside the page, how much was
   lost to the page edge.
4. Ink density per region against bands measured across the corpus -- the same
   idea as Phase 1's ``fg_ratio`` sanity band. A vitals crop with no ink means
   the box missed; a crop that is nearly all ink means it caught the letterhead
   or a scan artefact.
5. Prescription line-split shape: band count, band heights, and whether the
   split was trustworthy.

NOT implemented, and why (spec section 8 asks for these; they need things that
do not exist yet):
  - region localisation accuracy / IoU  -- needs hand-drawn ground-truth boxes
  - region field-containment rate       -- needs transcribed field values
  - prescription line-count accuracy    -- needs transcribed line counts
  - everything in ``src.eval.metrics.PENDING_METRICS`` -- needs extraction
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..segment.config import ANCHOR_NAMES, REGION_NAMES
from ..segment.pipeline import PRESCRIPTION_REGION, SegmentResult

# Phase-2 metrics that need ground truth this project does not have. Listed so
# the harness states what it is NOT measuring, rather than letting the proxy
# tables read as an evaluation of correctness.
PENDING_SEGMENT_METRICS = {
    "region_localisation_iou": "needs hand-drawn ground-truth region boxes",
    "region_field_containment_rate": "needs transcribed field values (phase 3+)",
    "prescription_line_count_accuracy": "needs transcribed line counts (phase 3+)",
}


@dataclass
class RegionProxy:
    name: str
    enabled: bool
    geometry_ok: bool
    width_px: int | None
    height_px: int | None
    ink_ratio: float | None
    ink_in_band: bool | None
    ink_band: list[float] | None
    clamped: bool
    clamped_frac: float | None
    clip_source: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageSegmentProxy:
    page_id: str
    source_shape_hw: list[int]
    effective_dpi: list[float] | None
    duration_s: float
    crop_source_stage: str

    # anchoring
    anchored: bool = False
    frame_source: str | None = None
    frame_reason: str | None = None
    anchor_span_px: float | None = None
    frame_scale: float | None = None
    anchor_y_residual_px: float | None = None
    anchor_scores: dict[str, float | None] = field(default_factory=dict)
    anchor_peak_ratios: dict[str, float | None] = field(default_factory=dict)
    anchor_found: dict[str, bool] = field(default_factory=dict)
    is_template_source: bool = False

    # independent frame check
    check_residual_px: dict[str, float | None] = field(default_factory=dict)
    check_residual_units: dict[str, float | None] = field(default_factory=dict)
    check_scores: dict[str, float | None] = field(default_factory=dict)
    check_found: dict[str, bool] = field(default_factory=dict)
    worst_check_residual_units: float | None = None

    # regions
    regions: list[RegionProxy] = field(default_factory=list)
    regions_geometry_ok: int = 0
    regions_total: int = 0
    regions_ink_in_band: int = 0

    # prescription line split
    line_split_ok: bool = False
    line_count: int | None = None
    line_height_min_px: int | None = None
    line_height_median_px: int | None = None
    line_height_max_px: int | None = None
    line_split_reason: str | None = None

    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["regions"] = [r.as_dict() for r in self.regions]
        return d


def _ink_band(cfg, name: str) -> list[float]:
    """Per-region ink-density band, falling back to the global band.

    Explicit fallback rather than a hidden default: the global band is written in
    config/segment.toml too, so nothing here invents a number.
    """
    bands = cfg.section("proxy").get("ink_band", {})
    if name in bands:
        return [float(v) for v in bands[name]]
    return [
        float(cfg.get("proxy", "min_region_ink")),
        float(cfg.get("proxy", "max_region_ink")),
    ]


def compute_segment_proxy(res: SegmentResult, cfg) -> PageSegmentProxy:
    """Proxy metrics for one Phase 2 page result. Nothing here re-runs the stage."""
    m = PageSegmentProxy(
        page_id=res.page_id,
        source_shape_hw=res.source_shape_hw,
        effective_dpi=res.effective_dpi,
        duration_s=round(res.duration_s, 3),
        crop_source_stage=res.crop_source_stage,
        warnings=list(res.warnings),
    )

    for name in ANCHOR_NAMES:
        a = res.anchors.get(name)
        m.anchor_scores[name] = None if a is None else a.score
        m.anchor_peak_ratios[name] = None if a is None else a.peak_ratio
        m.anchor_found[name] = bool(a is not None and a.found)
        if a is not None and a.is_template_source:
            m.is_template_source = True

    m.anchored = res.anchored
    if res.frame is not None:
        m.frame_source = res.frame.source
        m.frame_reason = res.frame.reason
        m.anchor_span_px = (
            None if res.frame.span_px is None else round(res.frame.span_px, 1)
        )
        m.frame_scale = res.frame.scale
        m.anchor_y_residual_px = res.frame.anchor_y_residual_px

    worst = None
    for c in res.checks:
        m.check_scores[c.name] = c.score
        m.check_found[c.name] = bool(c.found)
        m.check_residual_px[c.name] = c.residual_px
        m.check_residual_units[c.name] = c.residual_units
        if c.residual_units is not None:
            worst = c.residual_units if worst is None else max(worst, c.residual_units)
    m.worst_check_residual_units = worst

    for r in res.regions:
        band = _ink_band(cfg, r.name)
        in_band = (
            None
            if r.ink_ratio is None
            else bool(band[0] <= r.ink_ratio <= band[1])
        )
        m.regions.append(
            RegionProxy(
                name=r.name,
                enabled=r.enabled,
                geometry_ok=r.geometry_ok,
                width_px=None if r.shape_hw is None else r.shape_hw[1],
                height_px=None if r.shape_hw is None else r.shape_hw[0],
                ink_ratio=r.ink_ratio,
                ink_in_band=in_band,
                ink_band=band,
                clamped=r.clamped,
                clamped_frac=r.clamped_frac,
                clip_source=r.clip_source,
            )
        )
        if in_band is False:
            m.warnings.append(
                f"region {r.name!r} ink density {r.ink_ratio:.4f} outside the "
                f"measured band [{band[0]}, {band[1]}]: the box may have missed "
                f"its content or caught printed matter"
            )
    m.regions_total = len([r for r in res.regions if r.enabled])
    m.regions_geometry_ok = len([r for r in res.regions if r.geometry_ok])
    m.regions_ink_in_band = len([r for r in m.regions if r.ink_in_band])

    m.line_split_ok = bool(res.line_info.get("split_ok"))
    m.line_split_reason = res.line_info.get("reason")
    if res.lines:
        heights = sorted(b.height for b in res.lines)
        m.line_count = len(res.lines)
        m.line_height_min_px = heights[0]
        m.line_height_median_px = heights[len(heights) // 2]
        m.line_height_max_px = heights[-1]
    else:
        m.line_count = 0
    return m


def region_names_for_table() -> tuple[str, ...]:
    return REGION_NAMES


def prescription_region_name() -> str:
    return PRESCRIPTION_REGION
