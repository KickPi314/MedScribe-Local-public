"""Prescription-line splitting by horizontal ink projection (spec section 3.4).

The row-projection machinery is *not* reimplemented here: ``_row_ink_profile``
and ``_empty_runs`` from ``src.preprocess.steps`` are imported directly. They
already encode two decisions that were measured in Phase 1 and that this stage
must not diverge from -- the binary-vs-greyscale ink rule, and treating a gutter
as a run of rows whose ink fraction is at or below a threshold. Duplicating them
would let the two copies drift apart silently.

They are underscore-named because they are Phase 1 internals; importing them is
a deliberate coupling to Phase 1's definition of "an empty row", not an
oversight, and Phase 1 is not modified to accommodate Phase 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..preprocess.config import Config
from ..preprocess.steps import _empty_runs, _row_ink_profile


@dataclass
class LineBand:
    """One prescription line inside the prescription-block crop.

    ``y0``/``y1`` are rows relative to the top of the block crop, already
    padded by ``[lines].pad_units``.
    """

    index: int
    y0: int
    y1: int
    ink_mean: float

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["height_px"] = self.height
        return d


def split_lines(
    block: np.ndarray,
    cfg: Config,
    span_px: float,
    block_x0_px: int = 0,
    origin_x: float = 0.0,
) -> tuple[list[LineBand], dict[str, Any]]:
    """Split a prescription-block crop into per-line bands.

    ``span_px`` is the anchor unit (:attr:`AnchorFrame.span_px`). Minimum gutter
    height and minimum line height are configured in anchor units so they mean
    the same *physical* size on a 204 dpi scan and a 243 dpi one, instead of
    silently getting stricter as resolution rises.

    The profile is taken over a *column window* (``[lines].profile_x_units``),
    not the full block width. Measured reason: the doctor's tick/check marks in
    the left margin of the prescription list are single strokes 0.10-0.35 anchor
    units from the origin that run diagonally *across* the gutter between two
    drug entries, and the Gujarati marginalia on the far right does the same. On
    the full width they bridged every gutter and the projection found 1-4 bands
    where the real corpus has 2-5 drug entries. Restricted to the drug-name and
    dash columns the same pages split cleanly. This is the same fix, for the same
    reason, as ``[deskew].printed_band_box`` restricting x to exclude the
    page-edge shadow in Phase 1.

    Returns ``(bands, info)``. ``info["split_ok"]`` is False when the profile
    produced an implausible number of bands; in that case the caller must not
    present the bands as the page's prescription lines (spec section 1.3 --
    flag, do not emit a plausible-looking answer).
    """
    row_thresh = float(cfg.get("lines", "row_ink_threshold"))
    min_gap = max(int(round(float(cfg.get("lines", "min_gap_units")) * span_px)), 1)
    min_h = max(int(round(float(cfg.get("lines", "min_line_height_units")) * span_px)), 1)
    pad = max(int(round(float(cfg.get("lines", "pad_units")) * span_px)), 0)
    max_lines = int(cfg.get("lines", "max_lines"))
    px0u, px1u = cfg.get("lines", "profile_x_units")

    h, w = block.shape[:2]
    # profile window, in columns of the block crop
    c0 = int(round(origin_x + span_px * float(px0u))) - int(block_x0_px)
    c1 = int(round(origin_x + span_px * float(px1u))) - int(block_x0_px)
    c0, c1 = max(c0, 0), min(c1, w)
    info: dict[str, Any] = {
        "row_ink_threshold": row_thresh,
        "min_gap_px": min_gap,
        "min_line_height_px": min_h,
        "pad_px": pad,
        "block_height_px": int(h),
        "profile_x_units": [float(px0u), float(px1u)],
        "profile_cols": [int(c0), int(c1)],
        "split_ok": False,
    }
    if c1 - c0 < 20:
        info["reason"] = (
            f"[lines].profile_x_units {[px0u, px1u]} leaves only {c1 - c0} columns "
            f"inside the prescription block on this page; not splitting"
        )
        return [], info
    if h < min_h * 2:
        info["reason"] = (
            f"prescription block is only {h}px tall, less than two minimum line "
            f"heights ({min_h}px); not splitting"
        )
        return [], info

    profile = _row_ink_profile(block[:, c0:c1])
    gutters = [
        g for g in _empty_runs(profile, 0, h, row_thresh) if g[1] - g[0] >= min_gap
    ]
    info["gutters_found"] = len(gutters)

    # Content bands = the complement of the accepted gutters.
    bands: list[tuple[int, int]] = []
    cursor = 0
    for g0, g1 in gutters:
        if g0 > cursor:
            bands.append((cursor, g0))
        cursor = g1
    if cursor < h:
        bands.append((cursor, h))

    info["bands_before_filter"] = len(bands)
    kept: list[LineBand] = []
    for y0, y1 in bands:
        if y1 - y0 < min_h:
            continue
        seg = profile[y0:y1]
        if seg.size == 0 or float(seg.mean()) <= row_thresh:
            continue
        py0 = max(y0 - pad, 0)
        py1 = min(y1 + pad, h)
        kept.append(
            LineBand(
                index=len(kept),
                y0=int(py0),
                y1=int(py1),
                ink_mean=round(float(seg.mean()), 5),
            )
        )

    info["lines"] = len(kept)
    if not kept:
        info["reason"] = (
            "the row-projection profile produced no band tall enough to be a "
            "prescription line; the block crop is probably empty or mis-placed"
        )
        return [], info
    if len(kept) > max_lines:
        info["reason"] = (
            f"row projection produced {len(kept)} bands, more than "
            f"[lines].max_lines {max_lines}; the split is not trustworthy on this "
            f"page, so the block is reported unsplit rather than as a "
            f"plausible-looking line list"
        )
        return kept, info

    info["split_ok"] = True
    return kept, info
