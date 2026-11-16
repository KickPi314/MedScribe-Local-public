"""Region geometry in the anchor-derived coordinate frame (spec section 3.4).

Region boxes live in ``config/segment.toml`` as ``[x0, x1, y0, y1]`` in *anchor
units*: 1 unit is the horizontal distance between the two letterhead anchors,
and the origin is the logo anchor's top-left corner. Expressing them this way
rather than as whole-page fractions is the whole point of Phase 2 -- the
letterhead drifts across the corpus (measured 2.6% of page height vertically,
5.5% of page width horizontally), so a page fraction addresses a different
physical spot on every scan while an anchor unit addresses the same one.

Nothing here decides *what a region says*; it only decides where to cut. Sanity
checks are reported as measurements, never used to nudge a box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..preprocess.config import Config
from ..preprocess.steps import ink_mask, to_gray
from .anchors import AnchorFrame
from .config import REGION_NAMES


def ink_fraction(img: np.ndarray) -> float:
    """Fraction of ink (dark) pixels.

    Handles both a binarised crop and a greyscale one, using the same rule as
    ``src.preprocess.steps._row_ink_profile`` so the number is comparable with
    Phase 1's ``foreground_ratio`` / ``edge_ink_*`` figures.
    """
    if img.size == 0:
        return 0.0
    gray = to_gray(img)
    uniq = np.unique(gray)
    if len(uniq) <= 2 and set(uniq.tolist()) <= {0, 255}:
        return float((gray < 128).mean())
    return float(ink_mask(gray, drop=40.0).mean())


@dataclass
class RegionCrop:
    """One named region of one page."""

    name: str
    enabled: bool
    box_units: list[float]
    box_px: list[int] | None = None          # [x0, x1, y0, y1] in page pixels
    shape_hw: list[int] | None = None
    image: np.ndarray | None = None
    ink_ratio: float | None = None
    # geometry sanity, reported per spec's "measure, do not fabricate" bar
    geometry_ok: bool = False
    clamped: bool = False                    # the box ran off the page and was cut
    clamped_frac: float | None = None        # fraction of the requested area lost
    # where the bottom edge came from, when the region declares
    # clip_bottom_to_landmark: the measured landmark, or the configured box
    clip_source: str | None = None
    notes: list[str] = field(default_factory=list)
    dump: str | None = None

    def as_dict(self, include_image: bool = False) -> dict[str, Any]:
        d = {
            "name": self.name,
            "enabled": self.enabled,
            "box_units": [round(float(v), 4) for v in self.box_units],
            "box_px": self.box_px,
            "shape_hw": self.shape_hw,
            "ink_ratio": self.ink_ratio,
            "geometry_ok": self.geometry_ok,
            "clamped": self.clamped,
            "clamped_frac": self.clamped_frac,
            "clip_source": self.clip_source,
            "notes": self.notes,
            "dump": self.dump,
        }
        if include_image and self.image is not None:
            d["shape_hw"] = list(self.image.shape[:2])
        return d


def crop_regions(
    img: np.ndarray,
    frame: AnchorFrame,
    cfg: Config,
    keep_images: bool = True,
    checks: list | None = None,
) -> list[RegionCrop]:
    """Cut every enabled region out of ``img`` using ``frame``.

    ``img`` must be in the same (deskewed, un-cropped) pixel frame the anchors
    were matched in -- i.e. a Phase 1 stage output from before the ``crop``
    step. Passing the Phase 1 *final* image would silently shift every box by
    the header crop.

    ``checks`` are the frame-check landmark results. A region may declare
    ``clip_bottom_to_landmark``, in which case its bottom edge is pulled up to
    that landmark's *measured* position on this page. That is a measurement, not
    a guess, and it is the only thing that resolves the printed-footer overlap
    Phase 1 recorded as unresolved (config/preprocess.toml, ``[crop]``): the
    printed footer's position relative to the anchors varies by 0.14 anchor
    units across the corpus, which is more than the gap between "clears the
    print" and "clips the handwriting", so no fixed box can do it.
    """
    if not frame.ok:
        raise ValueError("crop_regions needs a usable AnchorFrame")

    min_px = int(cfg.get("proxy", "min_region_px"))
    h, w = img.shape[:2]
    regions_cfg = cfg.section("regions")
    by_name = {c.name: c for c in (checks or [])}
    out: list[RegionCrop] = []

    for name in REGION_NAMES:
        spec = regions_cfg[name]
        box_units = [float(v) for v in spec["box_units"]]
        enabled = bool(spec.get("enabled", True))
        rc = RegionCrop(name=name, enabled=enabled, box_units=box_units)
        if not enabled:
            rc.notes.append("disabled in [regions.%s]" % name)
            out.append(rc)
            continue

        clip_to = spec.get("clip_bottom_to_landmark")
        if clip_to:
            chk = by_name.get(str(clip_to))
            margin = float(spec["clip_margin_units"])
            if chk is None:
                rc.notes.append(
                    f"clip_bottom_to_landmark names {clip_to!r}, which is not a "
                    f"[frame_check] landmark; bottom left at the configured box"
                )
            elif not chk.found or chk.measured_y is None:
                rc.notes.append(
                    f"landmark {clip_to!r} was not located on this page, so the "
                    f"bottom edge falls back to the configured box_units value "
                    f"({box_units[3]}) -- it may include printed matter"
                )
                rc.clip_source = "configured_box (landmark not found)"
            else:
                lm_units = (chk.measured_y - frame.origin_y) / frame.span_px - margin
                if lm_units < box_units[3]:
                    box_units = list(box_units)
                    box_units[3] = round(float(lm_units), 4)
                    rc.clip_source = f"landmark:{clip_to}"
                    rc.notes.append(
                        f"bottom pulled up to the measured {clip_to!r} position "
                        f"({lm_units:.3f} anchor units) instead of the configured "
                        f"{rc.box_units[3]}"
                    )
                else:
                    rc.clip_source = "configured_box (landmark sits below it)"
                rc.box_units = box_units

        # requested (unclamped) box, so the amount lost to clamping is visible
        rx0 = frame.origin_x + frame.span_px * box_units[0]
        rx1 = frame.origin_x + frame.span_px * box_units[1]
        ry0 = frame.origin_y + frame.span_px * box_units[2]
        ry1 = frame.origin_y + frame.span_px * box_units[3]
        req_area = max((rx1 - rx0) * (ry1 - ry0), 1e-6)

        x0, x1, y0, y1 = frame.to_page(box_units, (h, w))
        rc.box_px = [x0, x1, y0, y1]
        got_area = max((x1 - x0) * (y1 - y0), 0)
        lost = 1.0 - got_area / req_area
        if lost > 1e-3:
            rc.clamped = True
            rc.clamped_frac = round(float(lost), 4)
            rc.notes.append(
                f"box ran off the page and lost {lost * 100:.1f}% of its requested "
                f"area to the page edge"
            )

        if x1 - x0 < 2 or y1 - y0 < 2:
            rc.notes.append("degenerate box (zero or one pixel in an axis)")
            out.append(rc)
            continue
        if got_area < min_px:
            rc.notes.append(
                f"box area {got_area}px below [proxy].min_region_px {min_px}"
            )
            out.append(rc)
            continue

        crop = img[y0:y1, x0:x1]
        rc.shape_hw = [int(crop.shape[0]), int(crop.shape[1])]
        rc.ink_ratio = round(ink_fraction(crop), 5)
        rc.geometry_ok = True
        if keep_images:
            rc.image = crop
        out.append(rc)

    return out
