"""Phase 2 orchestration: ingest -> Phase 1 pipeline -> anchor -> crop -> split.

Dump layout mirrors Phase 1's ``data/dumps/<page_id>/``::

    data/segments/<page_id>/header_strip.png
                           /vitals_column.png
                           /...
                           /prescription_block.png
                           /prescription_block_line_0.png
                           /overlay.png
                           /manifest.json

PHI: directories and manifests are keyed on ``page_id`` (a content hash) only.
No source filename, no patient name, ever -- same rule as Phase 1, and
``data/segments/`` is gitignored for the same reason ``data/dumps/`` is.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..preprocess.config import Config, ConfigError, PROJECT_ROOT
from ..preprocess.config import STEP_NAMES
from ..preprocess.ingest import PageImage
from ..preprocess.pipeline import PipelineResult, _write_png
from .anchors import (
    AnchorFrame,
    AnchorMatch,
    LandmarkCheck,
    build_frame,
    check_landmarks,
    deskewed_source_gray,
    load_templates,
    match_anchors,
)
from .config import ANCHOR_NAMES
from .lines import LineBand, split_lines
from .regions import RegionCrop, crop_regions

DEFAULT_SEGMENT_ROOT = PROJECT_ROOT / "data" / "segments"

PRESCRIPTION_REGION = "prescription_block"


@dataclass
class SegmentResult:
    page_id: str
    source_shape_hw: list[int]
    effective_dpi: list[float] | None
    crop_source_stage: str
    anchors: dict[str, AnchorMatch] = field(default_factory=dict)
    frame: AnchorFrame | None = None
    checks: list[LandmarkCheck] = field(default_factory=list)
    regions: list[RegionCrop] = field(default_factory=list)
    lines: list[LineBand] = field(default_factory=list)
    line_info: dict[str, Any] = field(default_factory=dict)
    line_dumps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def anchored(self) -> bool:
        return self.frame is not None and self.frame.ok

    def region(self, name: str) -> RegionCrop | None:
        for r in self.regions:
            if r.name == name:
                return r
        return None


# ---------------------------------------------------------------------------
# choosing the image the crops come from
# ---------------------------------------------------------------------------


def _crop_source(presult: PipelineResult, cfg: Config) -> tuple[np.ndarray, str, list[str]]:
    """Pick the Phase 1 stage image that region crops are cut from.

    It must be a stage that runs *before* Phase 1's ``crop`` step: ``crop``
    changes the pixel frame (it removes the header band), so cutting anchor-frame
    boxes out of the Phase 1 *final* image would shift every one of them by the
    header crop. Everything from ``deskew`` onwards shares one frame -- the
    deskewed page at full size -- which is the frame the anchors are matched in.
    """
    want = str(cfg.get("source", "stage"))
    order = [s.name for s in presult.stages]
    if want not in STEP_NAMES:
        raise ConfigError(
            f"[source].stage {want!r} is not a Phase 1 step; choose one of "
            f"{[s for s in STEP_NAMES if s != 'crop']}"
        )
    if want == "crop":
        raise ConfigError(
            "[source].stage may not be 'crop': the crop step changes the pixel "
            "frame the anchors were matched in"
        )
    if "crop" in order and order.index(want) > order.index("crop"):
        raise ConfigError(
            f"[source].stage {want!r} runs after Phase 1's 'crop' step in "
            f"[steps].order, so its pixel frame no longer matches the anchors"
        )

    notes: list[str] = []
    stage = presult.stage(want)
    if stage is not None and stage.enabled and stage.image is not None:
        return stage.image, want, notes

    # Configured stage is switched off: fall back to the last enabled stage
    # before 'crop' and say so, rather than silently using a different image.
    fallback = None
    for s in presult.stages:
        if s.name == "crop":
            break
        if s.image is not None:
            fallback = s
    if fallback is None:
        raise ConfigError(
            f"[source].stage {want!r} produced no image and no earlier Phase 1 "
            f"stage did either"
        )
    notes.append(
        f"[source].stage {want!r} is disabled in the preprocess config; region "
        f"crops were cut from {fallback.name!r} instead"
    )
    return fallback.image, fallback.name, notes


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


def segment_page(
    page: PageImage,
    presult: PipelineResult,
    cfg: Config,
    dump_dir: Path | None = None,
    templates: dict[str, np.ndarray] | None = None,
    keep_images: bool = True,
) -> SegmentResult:
    """Anchor, crop and line-split one page.

    ``presult`` is the Phase 1 :func:`~src.preprocess.pipeline.run_pipeline`
    result for the same page, run with ``keep_stage_images=True``.
    """
    t0 = time.perf_counter()
    src_img, src_stage, notes = _crop_source(presult, cfg)

    res = SegmentResult(
        page_id=page.page_id,
        source_shape_hw=list(presult.source_shape),
        effective_dpi=(
            [round(page.effective_dpi[0], 1), round(page.effective_dpi[1], 1)]
            if page.effective_dpi
            else None
        ),
        crop_source_stage=src_stage,
        warnings=list(notes),
    )

    gray = deskewed_source_gray(page, presult)
    if gray.shape[:2] != src_img.shape[:2]:
        # Would mean a Phase 1 step changed the frame before 'crop'. Refuse
        # rather than cut boxes out of a mismatched image.
        res.warnings.append(
            f"anchor frame {gray.shape[:2]} and crop source {src_img.shape[:2]} "
            f"have different shapes; refusing to segment this page"
        )
        res.duration_s = time.perf_counter() - t0
        return res

    res.anchors = match_anchors(
        gray, cfg, page.effective_dpi, templates=templates, page_id=page.page_id
    )
    res.frame = build_frame(res.anchors, cfg)

    for name in ANCHOR_NAMES:
        m = res.anchors[name]
        if not m.found:
            res.warnings.append(f"anchor {name!r} not located: {m.reason}")
    if not res.frame.ok:
        res.warnings.append(f"page not anchored: {res.frame.reason}")
        res.duration_s = time.perf_counter() - t0
        if dump_dir is not None:
            _dump(dump_dir, page, cfg, res, src_img, gray)
        return res
    if res.frame.source.startswith("single_anchor"):
        res.warnings.append(f"degraded frame: {res.frame.reason}")

    # Independent check: printed landmarks the frame did not use. Reported,
    # never fed back to adjust the frame (same rule as Phase 1's crosscheck).
    res.checks = check_landmarks(
        gray, res.frame, cfg, page.effective_dpi, templates=templates
    )
    for c in res.checks:
        if not c.found:
            res.warnings.append(
                f"frame-check landmark {c.name!r} not located, so the frame is "
                f"unverified there: {c.reason}"
            )
        elif c.within_tolerance is False:
            res.warnings.append(
                f"frame-check landmark {c.name!r} sits {c.residual_px:.0f}px "
                f"({c.residual_units:.3f} anchor units, tolerance "
                f"{c.tolerance_units:.3f}) from where the frame predicts it; "
                f"region boxes near it are off by about that much"
            )

    res.regions = crop_regions(
        src_img, res.frame, cfg, keep_images=True, checks=res.checks
    )

    presc = res.region(PRESCRIPTION_REGION)
    if presc is not None and presc.image is not None and presc.geometry_ok:
        res.lines, res.line_info = split_lines(
            presc.image,
            cfg,
            res.frame.span_px,
            block_x0_px=presc.box_px[0],
            origin_x=res.frame.origin_x,
        )
        if not res.line_info.get("split_ok"):
            res.warnings.append(
                f"prescription line split not trustworthy: "
                f"{res.line_info.get('reason', 'unknown')}"
            )

    if dump_dir is not None:
        _dump(dump_dir, page, cfg, res, src_img, gray)

    if not keep_images:
        for r in res.regions:
            r.image = None

    res.duration_s = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# dumping
# ---------------------------------------------------------------------------


def _dump(
    dump_dir: Path,
    page: PageImage,
    cfg: Config,
    res: SegmentResult,
    src_img: np.ndarray,
    gray: np.ndarray,
) -> None:
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_max = int(cfg.get("dump", "max_long_edge"))

    for r in res.regions:
        if r.image is None:
            continue
        name = f"{r.name}.png"
        _write_png(dump_dir / name, r.image, dump_max)
        r.dump = name

    presc = res.region(PRESCRIPTION_REGION)
    # Only write per-line crops when the split is trustworthy. An untrustworthy
    # split still gets its bands recorded in the manifest with the reason, so the
    # operator can see what happened -- but a directory full of line_N.png files
    # would read as a finished answer.
    if presc is not None and presc.image is not None and res.line_info.get("split_ok"):
        for band in res.lines:
            fname = f"{PRESCRIPTION_REGION}_line_{band.index}.png"
            _write_png(dump_dir / fname, presc.image[band.y0 : band.y1], dump_max)
            res.line_dumps.append(fname)

    if bool(cfg.get("dump", "overlay")):
        ov = _overlay(gray, res, cfg)
        if ov is not None:
            _write_png(dump_dir / "overlay.png", ov, int(cfg.get("dump", "overlay_max_long_edge")))

    (dump_dir / "manifest.json").write_text(
        json.dumps(_manifest(page, cfg, res), indent=2), encoding="utf-8"
    )


def _ref_span(cfg: Config) -> float:
    from .anchors import reference_span

    return reference_span(cfg)


_OVERLAY_COLOURS = {
    "header_strip": (0, 160, 255),
    "date_row": (255, 200, 0),
    "vitals_column": (0, 200, 0),
    "complaints_block": (255, 120, 0),
    "diagnosis_line": (200, 0, 200),
    "investigations_margin": (0, 200, 200),
    "prescription_block": (255, 0, 0),
}


def _overlay(gray: np.ndarray, res: SegmentResult, cfg: Config) -> np.ndarray | None:
    """Anchor and region boxes drawn on the deskewed page, for eyeball checking.

    This is the artefact that makes the region numbers in config/segment.toml
    reviewable: the operator edits a box, re-runs, and looks at the overlay.
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for name in ANCHOR_NAMES:
        m = res.anchors.get(name)
        if m is None or m.x is None:
            continue
        colour = (0, 255, 0) if m.found else (0, 0, 255)
        cv2.rectangle(bgr, (m.x, m.y), (m.x + m.w, m.y + m.h), colour, 3)
        cv2.putText(
            bgr,
            f"{name} {m.score:.3f}/{m.peak_ratio:.2f}{'' if m.found else ' REJECTED'}",
            (m.x, max(m.y - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colour,
            2,
            cv2.LINE_AA,
        )
    # Frame-check landmarks: predicted position (dashed-looking thin box) vs
    # measured position. A visible gap between the two IS the frame error.
    for c in res.checks:
        if c.predicted_x is None:
            continue
        spec = cfg.section("frame_check").get(c.name)
        if not isinstance(spec, dict):
            continue
        rx0, rx1, ry0, ry1 = spec["reference_box_px"]
        pw = int(round((float(rx1) - float(rx0)) * (res.frame.span_px / _ref_span(cfg))))
        ph = int(round((float(ry1) - float(ry0)) * (res.frame.span_px / _ref_span(cfg))))
        px, py = int(round(c.predicted_x)), int(round(c.predicted_y))
        cv2.rectangle(bgr, (px, py), (px + pw, py + ph), (140, 140, 140), 2)
        if c.found and c.measured_x is not None:
            cv2.rectangle(
                bgr,
                (c.measured_x, c.measured_y),
                (c.measured_x + pw, c.measured_y + ph),
                (0, 140, 255),
                2,
            )
            cv2.putText(
                bgr,
                f"check {c.name} resid={c.residual_px:.0f}px",
                (c.measured_x, max(c.measured_y - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
    for r in res.regions:
        if r.box_px is None:
            continue
        x0, x1, y0, y1 = r.box_px
        colour = _OVERLAY_COLOURS.get(r.name, (128, 128, 128))
        cv2.rectangle(bgr, (x0, y0), (x1, y1), colour, 3)
        cv2.putText(
            bgr,
            r.name,
            (x0 + 6, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colour,
            2,
            cv2.LINE_AA,
        )
    presc = res.region(PRESCRIPTION_REGION)
    if presc is not None and presc.box_px is not None and res.line_info.get("split_ok"):
        px0, px1, py0, _py1 = presc.box_px
        for band in res.lines:
            cv2.line(bgr, (px0, py0 + band.y0), (px1, py0 + band.y0), (255, 255, 0), 2)
            cv2.line(bgr, (px0, py0 + band.y1), (px1, py0 + band.y1), (255, 255, 0), 2)
    return bgr


def _manifest(page: PageImage, cfg: Config, res: SegmentResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": 2,
        "stage": "region_segmentation",
        # Deliberately no source filename: filenames contain patient names.
        "page_id": res.page_id,
        "source_page_index": page.page_index,
        "source_shape_hw": res.source_shape_hw,
        "effective_dpi": res.effective_dpi,
        "crop_source_stage": res.crop_source_stage,
        "anchors": {k: v.as_dict() for k, v in res.anchors.items()},
        "frame": None if res.frame is None else res.frame.as_dict(),
        "frame_check": [c.as_dict() for c in res.checks],
        "regions": [r.as_dict() for r in res.regions],
        "prescription_lines": {
            "info": res.line_info,
            "bands": [b.as_dict() for b in res.lines],
            "dumps": res.line_dumps,
        },
        "warnings": res.warnings,
        "duration_s": round(res.duration_s, 3),
        "not_implemented": {
            "gujarati_marginalia_crop": (
                "spec section 4 requires the Gujarati marginalia to be cropped and "
                "attached as an image to each prescription line. Detecting the "
                "marginalia sub-region is not part of the spec section 3.4 region "
                "list and is not implemented in Phase 2."
            ),
            "region_box_accuracy": (
                "there is no ground truth for region boxes, so no accuracy number "
                "is computed anywhere in Phase 2. See python -m src.eval segment "
                "for the proxy signals that are measured instead."
            ),
        },
        "config": cfg.as_dict(),
    }


def segment_dir_for(page_id: str, root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_SEGMENT_ROOT) / page_id
