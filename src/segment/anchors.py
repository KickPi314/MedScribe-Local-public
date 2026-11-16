"""Template anchoring on the printed letterhead (spec section 3.4).

Why this exists
---------------
Phase 1 documented (config/preprocess.toml, ``[crop]``) that no single fixed
header fraction works on this document class: the printed header's bottom edge
and the first handwritten body row occupy *overlapping* ranges across the real
corpus. Re-measured here on the deskewed pages, the letterhead wordmark's own
top edge moves from 0.076 to 0.102 of page height -- 2.6% of page height, or
about 90 px -- so a whole-page fraction cannot address the same physical spot on
two different scans. Anchoring on printed matter can, because the printed matter
*is* the coordinate system.

What is matched
---------------
Two purely pre-printed features, chosen because they are present and identical
on every page of the template:

``logo``                -- the clinic wordmark, top left.
``registration_block``  -- the doctor-name / qualification / speciality lines,
                           top right.

The registration block template deliberately stops *above* the
"Additional Registration No." line: that line is present on 4 of the 7 real
scans and absent on 3, so including it would make the template mismatch 3 pages
by construction. (This is the same variability Phase 1 saw as a moving header
boundary.)

Scale normalisation
-------------------
``PageImage.effective_dpi`` gives each page's measured scan resolution
(204-243 dpi across the corpus). The template is rescaled by
``page_dpi / template_reference_dpi`` before matching, so the sweep only has to
absorb the *residual* magnification error rather than brute-forcing a wide
scale range. Measured residual on the real corpus is inside +/-3%, so the sweep
is narrow (see ``[anchors].scale_sweep_halfwidth`` in config/segment.toml).

Never guess
-----------
A match is only reported when both the normalised correlation peak and the
peak's sharpness (peak over the best response outside a neighbourhood of it)
clear configured floors, exactly as ``_angle_printed_band`` in
src/preprocess/steps.py refuses to return an angle from a flat sweep. Below the
floor the anchor is reported ``found=False`` with a reason, and no region box is
derived from it.

Matching input
--------------
Anchoring runs on the *deskewed but not channel-suppressed* grey page. Phase 1's
channel separation is tuned to suppress the letterhead's magenta ink -- i.e. to
destroy precisely the signal this module needs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from ..preprocess.config import Config, ConfigError
from ..preprocess.ingest import PageImage
from ..preprocess.pipeline import PipelineResult
from ..preprocess.steps import rotate_image, to_gray
from .config import ANCHOR_NAMES, check_specs, template_path, template_specs


# ---------------------------------------------------------------------------
# the image anchoring runs on
# ---------------------------------------------------------------------------


def deskewed_source_gray(page: PageImage, result: PipelineResult) -> np.ndarray:
    """The page's source luminance in the pipeline's deskewed frame.

    Reconstructed from ``result.ctx["applied_skew_deg"]`` rather than read out
    of the pipeline, because ``run_pipeline`` strips arrays from the returned
    ctx. Reconstructing is exact -- it is the same call the deskew step makes to
    produce ``ctx["source_gray_deskewed"]`` -- and it keeps Phase 2 from having
    to modify Phase 1 to expose an intermediate.

    When deskew did not rotate (angle below ``[deskew].min_angle_deg``, or no
    trustworthy estimate) there is nothing to undo and the untouched luminance
    already *is* the pipeline frame.
    """
    gray = to_gray(page.image)
    angle = result.ctx.get("applied_skew_deg")
    if angle is None:
        return gray
    return rotate_image(gray, float(angle))


# ---------------------------------------------------------------------------
# one anchor
# ---------------------------------------------------------------------------


@dataclass
class AnchorMatch:
    """Result of matching one letterhead template against one page."""

    name: str
    found: bool
    reason: str | None = None
    # top-left corner of the matched template, in deskewed page pixels
    x: int | None = None
    y: int | None = None
    w: int | None = None
    h: int | None = None
    # TM_CCOEFF_NORMED peak of the winning scale
    score: float | None = None
    # peak / best response outside an exclusion neighbourhood of the peak.
    # 1.0 means "the whole search window matches this well", i.e. no real peak.
    peak_ratio: float | None = None
    # total template scale actually applied = dpi_scale * sweep factor
    scale_x: float | None = None
    scale_y: float | None = None
    # the dpi-derived hypothesis alone, so the sweep's contribution is visible
    dpi_scale_x: float | None = None
    dpi_scale_y: float | None = None
    sweep_factor: float | None = None
    scales_tried: int = 0
    # True when this page is the page the template was cropped from, which makes
    # its score a self-match rather than evidence of generalisation.
    is_template_source: bool = False

    @property
    def center(self) -> tuple[float, float] | None:
        if self.x is None:
            return None
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _second_best(resp: np.ndarray, peak_loc: tuple[int, int], rx: int, ry: int) -> float:
    """Best response outside a +/-(rx, ry) neighbourhood of ``peak_loc``.

    A separate *shape* measure from the peak height: a template can correlate
    well and still be ambiguous (repeated printed text, a large flat blob), and
    that case must not be reported as a located anchor.
    """
    masked = resp.copy()
    px, py = peak_loc
    x0, x1 = max(px - rx, 0), min(px + rx + 1, masked.shape[1])
    y0, y1 = max(py - ry, 0), min(py + ry + 1, masked.shape[0])
    masked[y0:y1, x0:x1] = -1.0
    return float(masked.max())


def match_anchor(
    gray: np.ndarray,
    template: np.ndarray,
    cfg: Config,
    name: str,
    spec: dict[str, Any],
    dpi_scale: tuple[float, float],
    is_template_source: bool = False,
) -> AnchorMatch:
    """Locate one letterhead template in ``gray`` (deskewed page luminance).

    ``spec`` is the template's config table (``file`` / ``reference_box_px`` /
    ``search_box_frac``). It is passed in rather than looked up so the same
    matcher serves both the two frame anchors and the independent frame-check
    landmarks in ``[frame_check]``.
    """
    x0f, x1f, y0f, y1f = spec["search_box_frac"]
    half = float(cfg.get("anchors", "scale_sweep_halfwidth"))
    step = float(cfg.get("anchors", "scale_sweep_step"))
    min_score = float(cfg.get("anchors", "min_score"))
    min_peak_ratio = float(cfg.get("anchors", "min_peak_ratio"))
    excl = float(cfg.get("anchors", "peak_exclusion_frac"))
    blur_k = int(cfg.get("anchors", "match_blur_ksize"))

    h, w = gray.shape[:2]
    sx0 = max(int(round(w * float(x0f))), 0)
    sx1 = min(int(round(w * float(x1f))), w)
    sy0 = max(int(round(h * float(y0f))), 0)
    sy1 = min(int(round(h * float(y1f))), h)
    if sx1 - sx0 < 8 or sy1 - sy0 < 8:
        return AnchorMatch(
            name=name,
            found=False,
            reason=f"[templates.{name}].search_box_frac is degenerate for a "
            f"{w}x{h}px page",
            is_template_source=is_template_source,
        )
    window = gray[sy0:sy1, sx0:sx1]

    # Mild blur on both sides. These are ~220 dpi JPEG scans; the pixel-level
    # noise is uncorrelated between two scans of the same printed glyph, and
    # smoothing it lifted the measured cross-page correlation peak without
    # moving the peak location (see config/segment.toml, [anchors]).
    if blur_k > 1:
        k = blur_k if blur_k % 2 == 1 else blur_k + 1
        window = cv2.GaussianBlur(window, (k, k), 0)
        template = cv2.GaussianBlur(template, (k, k), 0)

    th, tw = template.shape[:2]
    dsx, dsy = float(dpi_scale[0]), float(dpi_scale[1])

    n_steps = int(round(half / step)) if step > 0 else 0
    factors = [1.0 + i * step for i in range(-n_steps, n_steps + 1)]

    best: dict[str, Any] | None = None
    tried = 0
    for f in factors:
        rw = int(round(tw * dsx * f))
        rh = int(round(th * dsy * f))
        if rw < 8 or rh < 8 or rw > window.shape[1] or rh > window.shape[0]:
            continue
        tpl = cv2.resize(template, (rw, rh), interpolation=cv2.INTER_AREA)
        resp = cv2.matchTemplate(window, tpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(resp)
        tried += 1
        if best is None or peak > best["score"]:
            rx = max(int(round(rw * excl)), 1)
            ry = max(int(round(rh * excl)), 1)
            second = _second_best(resp, loc, rx, ry)
            best = {
                "score": float(peak),
                "second": second,
                "loc": loc,
                "w": rw,
                "h": rh,
                "factor": f,
            }

    if best is None:
        return AnchorMatch(
            name=name,
            found=False,
            reason=(
                f"no template scale fits inside the {name!r} search window on a "
                f"{w}x{h}px page (window {sx1 - sx0}x{sy1 - sy0}px, template "
                f"{tw}x{th}px at dpi scale {dsx:.3f}x{dsy:.3f})"
            ),
            dpi_scale_x=round(dsx, 4),
            dpi_scale_y=round(dsy, 4),
            is_template_source=is_template_source,
        )

    ratio = best["score"] / max(best["second"], 1e-6)
    m = AnchorMatch(
        name=name,
        found=True,
        x=int(sx0 + best["loc"][0]),
        y=int(sy0 + best["loc"][1]),
        w=int(best["w"]),
        h=int(best["h"]),
        score=round(best["score"], 4),
        peak_ratio=round(float(ratio), 3),
        scale_x=round(dsx * best["factor"], 4),
        scale_y=round(dsy * best["factor"], 4),
        dpi_scale_x=round(dsx, 4),
        dpi_scale_y=round(dsy, 4),
        sweep_factor=round(best["factor"], 4),
        scales_tried=tried,
        is_template_source=is_template_source,
    )
    if m.score < min_score:
        m.found = False
        m.reason = (
            f"correlation peak {m.score:.3f} below [anchors].min_score "
            f"{min_score}; refusing to report an anchor position"
        )
    elif m.peak_ratio < min_peak_ratio:
        m.found = False
        m.reason = (
            f"correlation peak {m.score:.3f} is not sharp (peak/next "
            f"{m.peak_ratio:.2f} below [anchors].min_peak_ratio "
            f"{min_peak_ratio}); the match is ambiguous, refusing to report it"
        )
    return m


def load_templates(cfg: Config) -> dict[str, np.ndarray]:
    """Load every template PNG: the two frame anchors and the check landmarks.

    A missing asset is a loud error: silently skipping an anchor would silently
    downgrade every page on the run to a single-anchor frame.
    """
    out: dict[str, np.ndarray] = {}
    for name, spec in template_specs(cfg).items():
        p = template_path(cfg, spec["file"])
        if not p.is_file():
            raise ConfigError(
                f"template asset for {name!r} not found at {p}. Regenerate it "
                f"with: python -m src.segment make-templates samples/"
            )
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ConfigError(f"could not decode template asset {p}")
        out[name] = img
    return out


def dpi_scale_for(cfg: Config, page_dpi: tuple[float, float] | None) -> tuple[float, float]:
    ref_dpi = cfg.get("templates", "reference_dpi")
    if page_dpi is None:
        # No dpi metadata (raw image input). The sweep still runs, but from an
        # unnormalised starting point, which is recorded so the operator can see
        # why the sweep may have had to work harder.
        return (1.0, 1.0)
    return (
        float(page_dpi[0]) / float(ref_dpi[0]),
        float(page_dpi[1]) / float(ref_dpi[1]),
    )


def match_anchors(
    gray: np.ndarray,
    cfg: Config,
    page_dpi: tuple[float, float] | None,
    templates: dict[str, np.ndarray] | None = None,
    page_id: str | None = None,
) -> dict[str, AnchorMatch]:
    """Match the two frame anchors against one deskewed page."""
    templates = templates if templates is not None else load_templates(cfg)
    src_page_id = str(cfg.get("templates", "source_page_id"))
    dpi_scale = dpi_scale_for(cfg, page_dpi)
    specs = template_specs(cfg)
    return {
        name: match_anchor(
            gray,
            templates[name],
            cfg,
            name,
            specs[name],
            dpi_scale,
            is_template_source=(page_id is not None and page_id == src_page_id),
        )
        for name in ANCHOR_NAMES
    }


# ---------------------------------------------------------------------------
# the anchor-derived coordinate frame
# ---------------------------------------------------------------------------


@dataclass
class AnchorFrame:
    """Page coordinates expressed relative to the printed letterhead.

    ``origin`` is the top-left corner of the matched ``logo`` template (or of
    whichever single anchor was available). ``span_px`` is the horizontal
    distance between the two anchors' top-left corners, which is the stage's
    unit of length: region boxes in config/segment.toml are written in multiples
    of it, so they are independent of scan resolution and of how much blank
    scanner bed surrounds the page.
    """

    ok: bool
    source: str                       # how the frame was derived
    reason: str | None = None
    origin_x: float | None = None
    origin_y: float | None = None
    span_px: float | None = None
    scale: float | None = None        # span_px / reference span
    # (measured anchor dy) - (scale * reference anchor dy). Non-zero means the
    # two matches are vertically inconsistent, i.e. at least one is wrong.
    anchor_y_residual_px: float | None = None
    anchors_used: list[str] = field(default_factory=list)

    def to_page(self, box_units, page_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """``[x0, x1, y0, y1]`` in anchor units -> pixel box, clamped to page.

        Returns ``(x0, x1, y0, y1)``.
        """
        if not self.ok:
            raise ValueError("cannot map a box through an unusable AnchorFrame")
        h, w = page_shape
        ux0, ux1, uy0, uy1 = (float(v) for v in box_units)
        x0 = int(round(self.origin_x + self.span_px * ux0))
        x1 = int(round(self.origin_x + self.span_px * ux1))
        y0 = int(round(self.origin_y + self.span_px * uy0))
        y1 = int(round(self.origin_y + self.span_px * uy1))
        return (
            max(min(x0, w), 0),
            max(min(x1, w), 0),
            max(min(y0, h), 0),
            max(min(y1, h), 0),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reference_origins(cfg: Config) -> dict[str, tuple[float, float]]:
    tpl = cfg.section("templates")
    out = {}
    for name in ANCHOR_NAMES:
        bx0, _bx1, by0, _by1 = tpl[name]["reference_box_px"]
        out[name] = (float(bx0), float(by0))
    return out


def reference_span(cfg: Config) -> float:
    origins = _reference_origins(cfg)
    span = origins["registration_block"][0] - origins["logo"][0]
    if span <= 0:
        raise ConfigError(
            "[templates] reference boxes put the registration block left of the "
            "logo; the anchor span must be positive"
        )
    return span


def build_frame(matches: dict[str, AnchorMatch], cfg: Config) -> AnchorFrame:
    """Derive the coordinate frame from whatever anchors were confidently found."""
    origins = _reference_origins(cfg)
    ref_span = reference_span(cfg)
    ref_dy = origins["registration_block"][1] - origins["logo"][1]

    min_scale = float(cfg.get("frame", "min_scale"))
    max_scale = float(cfg.get("frame", "max_scale"))
    max_resid = float(cfg.get("frame", "max_anchor_y_residual_units"))
    allow_single = bool(cfg.get("frame", "allow_single_anchor"))

    found = [n for n in ANCHOR_NAMES if matches[n].found]

    if len(found) == 2:
        lg, rg = matches["logo"], matches["registration_block"]
        span = float(rg.x - lg.x)
        scale = span / ref_span
        resid = float((rg.y - lg.y) - scale * ref_dy)
        if not (min_scale <= scale <= max_scale):
            return AnchorFrame(
                ok=False,
                source="rejected",
                reason=(
                    f"anchor span implies a page scale of {scale:.3f}, outside "
                    f"[frame].min_scale/max_scale [{min_scale}, {max_scale}]; at "
                    f"least one anchor matched the wrong place, refusing to build "
                    f"a frame"
                ),
                anchors_used=found,
            )
        if abs(resid) > max_resid * span:
            return AnchorFrame(
                ok=False,
                source="rejected",
                reason=(
                    f"the two anchors disagree vertically by {resid:+.0f}px "
                    f"({abs(resid) / span:.3f} anchor units, limit "
                    f"{max_resid}); they cannot both be right, refusing to build "
                    f"a frame"
                ),
                anchor_y_residual_px=round(resid, 1),
                anchors_used=found,
            )
        return AnchorFrame(
            ok=True,
            source="two_anchor",
            origin_x=float(lg.x),
            origin_y=float(lg.y),
            span_px=span,
            scale=round(scale, 4),
            anchor_y_residual_px=round(resid, 1),
            anchors_used=found,
        )

    if len(found) == 1:
        name = found[0]
        m = matches[name]
        if not allow_single:
            return AnchorFrame(
                ok=False,
                source="rejected",
                reason=(
                    f"only the {name!r} anchor was found and "
                    f"[frame].allow_single_anchor is false"
                ),
                anchors_used=found,
            )
        # The matched template scale is itself a measurement (dpi normalisation
        # times the winning sweep factor), so this is a degraded frame, not a
        # guessed one. It is labelled as degraded everywhere it is reported.
        scale = float((m.scale_x + m.scale_y) / 2.0)
        if not (min_scale <= scale <= max_scale):
            return AnchorFrame(
                ok=False,
                source="rejected",
                reason=(
                    f"single-anchor frame from {name!r} implies page scale "
                    f"{scale:.3f}, outside [{min_scale}, {max_scale}]"
                ),
                anchors_used=found,
            )
        ox, oy = float(m.x), float(m.y)
        # Shift a registration-block-only origin back to where the logo anchor
        # would be, so region boxes stay written against one origin.
        if name == "registration_block":
            ox -= scale * ref_span
            oy -= scale * ref_dy
        return AnchorFrame(
            ok=True,
            source=f"single_anchor:{name}",
            reason=(
                f"only the {name!r} anchor cleared the confidence floors; the "
                f"frame's scale comes from that template's matched size, not from "
                f"a measured anchor separation, so region boxes are less reliable "
                f"on this page"
            ),
            origin_x=ox,
            origin_y=oy,
            span_px=scale * ref_span,
            scale=round(scale, 4),
            anchors_used=found,
        )

    reasons = "; ".join(
        f"{n}: {matches[n].reason}" for n in ANCHOR_NAMES if matches[n].reason
    )
    return AnchorFrame(
        ok=False,
        source="unanchored",
        reason=f"no letterhead anchor cleared the confidence floors ({reasons})",
        anchors_used=[],
    )


# ---------------------------------------------------------------------------
# independent check on the frame
# ---------------------------------------------------------------------------


@dataclass
class LandmarkCheck:
    """A printed landmark the frame did NOT use, matched to test the frame.

    The two anchors define the frame, so their own match positions cannot say
    whether the frame is *right* -- they fit by construction. A third and fourth
    printed feature can: if the letterhead really is rigid under a
    translate-and-scale frame, a landmark's measured position must agree with the
    position the frame predicts for it. The disagreement, in pixels, is the only
    honest accuracy figure Phase 2 can produce, and it is a geometric residual,
    not a field accuracy.

    Same discipline as Phase 1's ``crosscheck_method``: measured and reported,
    never fed back to adjust the frame.
    """

    name: str
    found: bool
    reason: str | None = None
    score: float | None = None
    peak_ratio: float | None = None
    measured_x: int | None = None
    measured_y: int | None = None
    predicted_x: float | None = None
    predicted_y: float | None = None
    residual_x_px: float | None = None
    residual_y_px: float | None = None
    residual_px: float | None = None
    residual_units: float | None = None
    tolerance_units: float | None = None
    within_tolerance: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_landmarks(
    gray: np.ndarray,
    frame: AnchorFrame,
    cfg: Config,
    page_dpi: tuple[float, float] | None,
    templates: dict[str, np.ndarray] | None = None,
) -> list[LandmarkCheck]:
    """Match every ``[frame_check.<name>]`` landmark and compare with the frame."""
    if not bool(cfg.get("frame_check", "enabled")):
        return []
    landmarks = check_specs(cfg)
    if not landmarks or not frame.ok:
        return []
    templates = templates if templates is not None else load_templates(cfg)
    origins = _reference_origins(cfg)
    ref_span = reference_span(cfg)
    lx, ly = origins["logo"]
    default_tol = float(cfg.get("frame_check", "max_residual_units"))
    dpi_scale = dpi_scale_for(cfg, page_dpi)

    out: list[LandmarkCheck] = []
    for name, spec in landmarks.items():
        m = match_anchor(gray, templates[name], cfg, name, spec, dpi_scale)
        chk = LandmarkCheck(
            name=name,
            found=m.found,
            reason=m.reason,
            score=m.score,
            peak_ratio=m.peak_ratio,
            measured_x=m.x,
            measured_y=m.y,
        )
        rx0, _rx1, ry0, _ry1 = spec["reference_box_px"]
        ux = (float(rx0) - lx) / ref_span
        uy = (float(ry0) - ly) / ref_span
        chk.predicted_x = round(frame.origin_x + frame.span_px * ux, 1)
        chk.predicted_y = round(frame.origin_y + frame.span_px * uy, 1)
        # A landmark near the anchors and one two units below them cannot share a
        # tolerance -- the measured residual grows with distance from the anchor
        # pair (see config/segment.toml, [frame_check]). Each landmark may
        # therefore set its own; the section-level value is the fallback.
        chk.tolerance_units = float(spec.get("max_residual_units", default_tol))
        if m.found and m.x is not None:
            dx = m.x - chk.predicted_x
            dy = m.y - chk.predicted_y
            chk.residual_x_px = round(float(dx), 1)
            chk.residual_y_px = round(float(dy), 1)
            chk.residual_px = round(float(np.hypot(dx, dy)), 1)
            chk.residual_units = round(chk.residual_px / frame.span_px, 4)
            chk.within_tolerance = bool(chk.residual_units <= chk.tolerance_units)
        out.append(chk)
    return out
