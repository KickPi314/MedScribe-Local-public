"""Preprocessing-quality proxy metrics (Phase 1).

What is implemented here is deliberately limited. These are *proxies* -- they
say whether each preprocessing stage did the thing it claims to do, measured on
real scans. They are not accuracy.

NOT implemented HERE, and where they now live (spec section 8):
  - per-field exact-match accuracy -- phase 3 (vitals) and phase 4 (every
    region): src/eval/vitals_metrics.py and src/eval/region_metrics.py
  - character error rate on free-text fields -- phase 4:
    src/eval/region_metrics.py
  - flag precision / recall, false-alarm rate, missed-error rate -- still
    nowhere. They need the `flagged` field state and the validation rules of
    spec section 4, which are phase 5. :data:`PENDING_METRICS` below is the
    list the Phase 1 and Phase 2 reports print; src/eval/__main__.py marks the
    two that have since been implemented rather than leaving the table stale.

Every number below is computed from real sample pixels. Nothing is stubbed with
a plausible-looking constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from ..preprocess.pipeline import PipelineResult
from ..preprocess.steps import estimate_skew, paper_level, to_gray

# Metrics that require extraction and therefore cannot exist in Phase 1. Two of
# them do exist now (phases 3 and 4); the names stay in this tuple because it is
# the checklist the Phase 1/2 reports print, and src/eval/__main__.py annotates
# each row with its real status rather than deleting the row.
PENDING_METRICS = (
    "field_exact_match_accuracy",
    "field_character_error_rate",
    "flag_precision",
    "flag_recall",
    "false_alarm_rate",
    "missed_error_rate",
)


# ---------------------------------------------------------------------------
# primitive image measures
# ---------------------------------------------------------------------------


def shannon_entropy(gray: np.ndarray) -> float:
    """Entropy of the 256-bin intensity histogram, in bits."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ink_contrast(gray: np.ndarray, ink_pct: float = 2.0) -> float:
    """Paper level minus the darkest ``ink_pct``% of pixels, in 8-bit units.

    A direct read on "how far off the paper is the ink" -- the quantity the
    whole preprocessing stage exists to increase.
    """
    return float(paper_level(gray) - np.percentile(gray, ink_pct))


def foreground_ratio(binary: np.ndarray) -> float:
    """Fraction of ink (dark) pixels in a binarised image."""
    g = to_gray(binary)
    return float((g < 128).mean())


def region_slice(shape_hw: tuple[int, int], box: list[float] | tuple[float, ...]):
    h, w = shape_hw
    x0, x1, y0, y1 = box
    return (
        slice(max(int(round(h * y0)), 0), min(int(round(h * y1)), h)),
        slice(max(int(round(w * x0)), 0), min(int(round(w * x1)), w)),
    )


def stroke_mask(gray_region: np.ndarray, drop: float) -> np.ndarray:
    """Strokes within a region: pixels ``drop`` levels below the region's paper."""
    return gray_region < (paper_level(gray_region) - drop)


def stroke_amplitude(
    channel: np.ndarray, mask: np.ndarray, sl, paper: float
) -> float | None:
    """How dark the masked strokes are, relative to global paper level.

    Larger = the method retained that content. Smaller = it suppressed it.
    """
    if mask.sum() < 50:
        return None
    vals = channel[sl][mask]
    return float(paper - np.median(vals))


def skew_in_band(gray: np.ndarray, cfg, band: list[float]) -> float | None:
    """Skew measured from a horizontal band of the page, via Hough.

    Used by the harness as an *independent* check on the deskew step. The band
    is set to the printed footer, which on this template is three or four
    perfectly horizontal printed text lines present on every page -- a far more
    trustworthy reference than the handwriting, which slopes as people write.

    The alternative verifier (row-projection variance over the whole page) was
    tried first and rejected: on one real page it reported 3.9 degrees of skew
    on a scan whose printed letterhead is demonstrably square, because it locked
    onto the long sloping underline strokes in the prescription block.
    """
    from ..preprocess.steps import _angle_hough

    h = gray.shape[0]
    y0, y1 = int(round(h * float(band[0]))), int(round(h * float(band[1])))
    y0, y1 = max(y0, 0), min(y1, h)
    if y1 - y0 < 20:
        return None
    angle, _ = _angle_hough(gray[y0:y1], cfg)
    return angle


def paper_noise(channel: np.ndarray) -> float:
    """Robust sigma of the paper background (MAD-based, JPEG-noise dominated)."""
    p = channel[channel >= np.percentile(channel, 70)]
    if p.size == 0:
        return 1.0
    med = np.median(p)
    mad = np.median(np.abs(p.astype(np.float32) - med))
    return max(float(mad * 1.4826), 1.0)


# ---------------------------------------------------------------------------
# per-page proxy metrics
# ---------------------------------------------------------------------------


@dataclass
class PageProxyMetrics:
    page_id: str
    source_shape_hw: list[int]
    final_shape_hw: list[int]
    effective_dpi: list[float] | None
    duration_s: float

    # deskew
    deskew_method: str | None = None
    deskew_estimated_deg: float | None = None
    deskew_applied: bool = False
    skew_before_deg: float | None = None   # independently re-measured
    skew_after_deg: float | None = None
    skew_reduction_deg: float | None = None
    # Second estimate from a different estimator, as recorded by the deskew step.
    # Reported, never merged into the primary estimate.
    deskew_crosscheck_method: str | None = None
    deskew_crosscheck_deg: float | None = None
    deskew_crosscheck_disagreement_deg: float | None = None
    # True when the pipeline estimated skew from the same page band this harness
    # verifies on, which makes the residual figure a closed-loop consistency
    # check rather than an independent one. Reported so the number is not read
    # as stronger evidence than it is.
    skew_check_closed_loop: bool = False

    # channel separation
    channel_method: str | None = None

    # contrast (CLAHE)
    entropy_before: float | None = None
    entropy_after: float | None = None
    std_before: float | None = None
    std_after: float | None = None
    ink_contrast_before: float | None = None
    ink_contrast_after: float | None = None

    # binarisation
    binarize_method: str | None = None
    fg_ratio: float | None = None
    fg_ratio_cropped: float | None = None

    # crop
    crop_applied: bool = False
    crop_top_frac: float | None = None
    crop_bottom_frac: float | None = None
    crop_retained_area_frac: float | None = None
    header_band_removed: bool | None = None
    footer_band_removed: bool | None = None
    # Ink density in the outermost rows of the cropped page. High values mean
    # content runs into the crop line and may have been clipped.
    edge_ink_top: float | None = None
    edge_ink_bottom: float | None = None

    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stage_image(result: PipelineResult, name: str) -> np.ndarray | None:
    return result.stage_image(name)


def _prev_enabled_image(result: PipelineResult, name: str) -> np.ndarray | None:
    """Output of the stage immediately before ``name`` that produced an image."""
    prev = None
    for s in result.stages:
        if s.name == name:
            return prev
        if s.image is not None:
            prev = s.image
    return prev


def compute_proxy_metrics(result: PipelineResult, cfg, eval_cfg) -> PageProxyMetrics:
    """Compute Phase 1 proxy metrics for one pipeline run.

    ``cfg``      : preprocessing Config (for re-measuring skew)
    ``eval_cfg`` : eval Config (sanity bands)
    """
    src = _stage_image(result, "source")
    m = PageProxyMetrics(
        page_id=result.page_id,
        source_shape_hw=list(result.source_shape),
        final_shape_hw=list(result.final.shape[:2]),
        effective_dpi=(
            [round(result.effective_dpi[0], 1), round(result.effective_dpi[1], 1)]
            if result.effective_dpi
            else None
        ),
        duration_s=round(result.total_duration_s, 3),
    )

    fg_min = float(eval_cfg.get("proxy", "fg_ratio_min"))
    fg_max = float(eval_cfg.get("proxy", "fg_ratio_max"))
    skew_min = float(eval_cfg.get("proxy", "skew_improvement_min_deg"))
    skew_warn = float(eval_cfg.get("proxy", "skew_residual_warn_deg"))
    contrast_warn = float(eval_cfg.get("proxy", "ink_contrast_warn"))

    # -- channel separation ---------------------------------------------
    cs = result.stage("channel_separation")
    if cs is not None and cs.enabled:
        m.channel_method = cs.info.get("method")

    # -- deskew ----------------------------------------------------------
    ds = result.stage("deskew")
    if ds is not None and ds.enabled:
        m.deskew_method = ds.info.get("method")
        m.deskew_estimated_deg = ds.info.get("estimated_angle_deg")
        m.deskew_applied = bool(ds.info.get("applied"))
        m.deskew_crosscheck_method = ds.info.get("crosscheck_method")
        m.deskew_crosscheck_deg = ds.info.get("crosscheck_angle_deg")
        m.deskew_crosscheck_disagreement_deg = ds.info.get(
            "crosscheck_disagreement_deg"
        )
        if ds.info.get("crosscheck_disagrees"):
            m.warnings.append(
                f"deskew estimators disagree by "
                f"{m.deskew_crosscheck_disagreement_deg:+.2f} deg "
                f"({m.deskew_method} vs {m.deskew_crosscheck_method}); the applied "
                f"angle came from {ds.info.get('angle_source', m.deskew_method)}"
            )
        # The harness verifies on the printed footer band. If the pipeline
        # estimated from that same band -- as its primary, via method="none"'s
        # measure_only_method, or via the fallback -- this stops being an
        # independent check and becomes a closed-loop consistency check.
        if "printed_band" in (
            str(ds.info.get("effective_method", "")),
            str(ds.info.get("angle_source", "")).split(":")[-1],
        ):
            m.skew_check_closed_loop = True
        # Re-measure from the printed footer band only, so the deskew step is
        # not grading its own homework: the pipeline estimates skew from the
        # whole page, the check uses a disjoint set of pixels that are known to
        # be printed and horizontal.
        before_img = _prev_enabled_image(result, "deskew")
        after_img = ds.image
        band = eval_cfg.get("proxy", "skew_reference_band")
        if before_img is not None and after_img is not None:
            b = skew_in_band(to_gray(before_img), cfg, band)
            a = skew_in_band(to_gray(after_img), cfg, band)
            m.skew_before_deg = None if b is None else round(b, 3)
            m.skew_after_deg = None if a is None else round(a, 3)
            if b is not None and a is not None:
                m.skew_reduction_deg = round(abs(b) - abs(a), 3)
                if m.skew_reduction_deg < skew_min and abs(b) > skew_warn:
                    m.warnings.append(
                        f"deskew did not reduce skew (|{b:+.2f}| -> |{a:+.2f}| deg)"
                    )
                if abs(a) > skew_warn:
                    m.warnings.append(f"residual skew {a:+.2f} deg above warn threshold")

    # -- CLAHE -----------------------------------------------------------
    cl = result.stage("clahe")
    if cl is not None and cl.enabled and cl.image is not None:
        before = _prev_enabled_image(result, "clahe")
        if before is not None:
            gb = to_gray(before)
            ga = to_gray(cl.image)
            m.entropy_before = round(shannon_entropy(gb), 3)
            m.entropy_after = round(shannon_entropy(ga), 3)
            m.std_before = round(float(gb.std()), 2)
            m.std_after = round(float(ga.std()), 2)
            m.ink_contrast_before = round(ink_contrast(gb), 1)
            m.ink_contrast_after = round(ink_contrast(ga), 1)
            if m.ink_contrast_before < contrast_warn:
                m.warnings.append(
                    f"faint ink: pre-CLAHE ink/paper contrast only "
                    f"{m.ink_contrast_before:.0f} levels"
                )

    # -- binarise --------------------------------------------------------
    bn = result.stage("binarize")
    if bn is not None and bn.enabled and bn.image is not None:
        m.binarize_method = bn.info.get("method")
        m.fg_ratio = round(foreground_ratio(bn.image), 5)
        if not (fg_min <= m.fg_ratio <= fg_max):
            m.warnings.append(
                f"foreground ratio {m.fg_ratio:.4f} outside sanity band "
                f"[{fg_min}, {fg_max}]"
            )

    # -- crop ------------------------------------------------------------
    cp = result.stage("crop")
    if cp is not None and cp.enabled:
        m.crop_applied = bool(cp.info.get("applied"))
        m.crop_top_frac = cp.info.get("top_frac")
        m.crop_bottom_frac = cp.info.get("bottom_frac")
        m.crop_retained_area_frac = cp.info.get("retained_area_frac")
        if cp.image is not None:
            m.fg_ratio_cropped = round(foreground_ratio(cp.image), 5)
            g = to_gray(cp.image)
            band = max(int(round(g.shape[0] * 0.01)), 2)
            m.edge_ink_top = round(float((g[:band] < 128).mean()), 4)
            m.edge_ink_bottom = round(float((g[-band:] < 128).mean()), 4)
            edge_warn = float(eval_cfg.get("proxy", "crop_edge_ink_warn"))
            if m.edge_ink_top > edge_warn:
                m.warnings.append(
                    f"content runs into the top crop line (edge ink "
                    f"{m.edge_ink_top:.3f}); a body row may be clipped"
                )
            if m.edge_ink_bottom > edge_warn:
                m.warnings.append(
                    f"content runs into the bottom crop line (edge ink "
                    f"{m.edge_ink_bottom:.3f}); handwriting may be clipped"
                )
        if not m.crop_applied:
            m.warnings.append(f"crop not applied: {cp.info.get('reason', 'unknown')}")
        else:
            # Did the crop actually clear the printed letterhead bands? Checked
            # against the *source* image geometry via the known band positions.
            hb, fb = _letterhead_bands_removed(src, cp.info, eval_cfg)
            m.header_band_removed = hb
            m.footer_band_removed = fb
            if hb is False:
                m.warnings.append(
                    f"crop top {m.crop_top_frac} does not clear the deepest printed "
                    f"header seen in the corpus; a residual registration line may "
                    f"remain (needs phase-2 template anchoring to resolve)"
                )
            if fb is False:
                m.warnings.append("printed footer band not fully removed by crop")

    return m


def _letterhead_bands_removed(src, crop_info, eval_cfg) -> tuple[bool | None, bool | None]:
    """Does the crop box provably clear the printed letterhead bands?

    Compared against the *worst-case* band positions measured across the whole
    real-scan corpus (``[proxy].printed_header_bottom_frac`` /
    ``printed_footer_top_frac``). This is intentionally strict: ``False`` means
    "cannot prove the band is gone on a page whose letterhead sits at the
    corpus-worst position", not "printed matter is definitely still present".
    Proving it per page needs the template anchoring of spec section 3.4
    (phase 2).
    """
    if src is None:
        return None, None
    top_px = crop_info.get("top_px")
    bottom_px = crop_info.get("bottom_px")
    if top_px is None or bottom_px is None:
        return None, None
    h = src.shape[0]
    header_bottom = float(eval_cfg.get("proxy", "printed_header_bottom_frac"))
    footer_top = float(eval_cfg.get("proxy", "printed_footer_top_frac"))
    # Compare in whole rows, not fractions: a fraction comparison fails by one
    # row of rounding error and reports a correct crop as a failure.
    return (
        top_px >= int(round(h * header_bottom)),
        bottom_px <= int(round(h * footer_top)),
    )


# ---------------------------------------------------------------------------
# channel-separation comparison
# ---------------------------------------------------------------------------


@dataclass
class ChannelScore:
    method: str
    keep_hand: float | None          # SNR of handwriting strokes (want HIGH)
    lose_logo: float | None          # SNR of pink logo print   (want LOW)
    lose_footer: float | None        # SNR of pink footer print (want LOW)
    lose_drname: float | None        # SNR of black doctor-name print (want LOW)
    selectivity: float | None        # keep_hand / mean(lose_*)
    usable: bool = True              # False when the method destroyed the ink
    # downstream (post CLAHE + binarise) ink coverage per region, only filled
    # in when the downstream probe is requested
    bin_cov_hand: float | None = None
    bin_cov_logo: float | None = None
    bin_cov_footer: float | None = None
    bin_cov_drname: float | None = None
    bin_selectivity: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_channel_methods(
    bgr: np.ndarray,
    cfg,
    eval_cfg,
    methods: list[str],
    downstream: bool = False,
) -> list[ChannelScore]:
    """Score channel-separation methods on one real page.

    Stroke masks are built once from the source luminance and reused for every
    method, so no method is scored on a mask it produced itself.

    Two scores are produced:

    ``selectivity``     -- grey-level amplitude ratio straight out of the
                           channel-separation step.
    ``bin_selectivity`` -- ink coverage ratio *after* CLAHE + binarisation
                           (``downstream=True``). This is the decision-relevant
                           one: it answers "does the letterhead actually
                           disappear from the binarised page while the
                           handwriting survives".

    A method whose handwriting SNR falls below
    ``[channel_probe].min_keep_hand_snr`` is marked ``usable=False``: it has
    destroyed the ink, which makes its selectivity ratio meaningless (a method
    that outputs blank paper would otherwise score infinitely well).
    """
    from ..preprocess.steps import CHANNEL_METHODS, binarize, clahe

    gray = to_gray(bgr)
    shape = gray.shape
    drop = float(eval_cfg.get("channel_probe", "stroke_drop"))
    min_keep = float(eval_cfg.get_opt("channel_probe", "min_keep_hand_snr", 0.0))

    boxes = {
        "hand": eval_cfg.get("channel_probe", "handwriting_body"),
        "logo": eval_cfg.get("channel_probe", "letterhead_logo"),
        "footer": eval_cfg.get("channel_probe", "letterhead_footer"),
        "drname": eval_cfg.get("channel_probe", "letterhead_drname"),
    }
    slices = {k: region_slice(shape, v) for k, v in boxes.items()}
    masks = {k: stroke_mask(gray[sl], drop) for k, sl in slices.items()}

    out: list[ChannelScore] = []
    for name in methods:
        if name not in CHANNEL_METHODS:
            raise ValueError(f"unknown channel method in eval config: {name!r}")
        ch = CHANNEL_METHODS[name](bgr, cfg)
        paper = paper_level(ch)
        sigma = paper_noise(ch)
        amps = {k: stroke_amplitude(ch, masks[k], slices[k], paper) for k in slices}
        snr = {k: (None if v is None else v / sigma) for k, v in amps.items()}
        losses = [snr[k] for k in ("logo", "footer", "drname") if snr[k] is not None]
        sel = None
        if snr["hand"] is not None and losses:
            denom = max(float(np.mean(losses)), 1e-6)
            sel = snr["hand"] / denom
        usable = snr["hand"] is not None and snr["hand"] >= min_keep

        score = ChannelScore(
            method=name,
            keep_hand=None if snr["hand"] is None else round(snr["hand"], 2),
            lose_logo=None if snr["logo"] is None else round(snr["logo"], 2),
            lose_footer=None if snr["footer"] is None else round(snr["footer"], 2),
            lose_drname=None if snr["drname"] is None else round(snr["drname"], 2),
            selectivity=None if sel is None else round(sel, 3),
            usable=bool(usable),
        )

        if downstream:
            ctx: dict[str, Any] = {"source_gray": gray}
            contrasted, _ = clahe(ch, cfg, ctx)
            binary, _ = binarize(contrasted, cfg, ctx)
            cov = {
                k: float((binary[sl] < 128).mean()) for k, sl in slices.items()
            }
            score.bin_cov_hand = round(cov["hand"], 4)
            score.bin_cov_logo = round(cov["logo"], 4)
            score.bin_cov_footer = round(cov["footer"], 4)
            score.bin_cov_drname = round(cov["drname"], 4)
            denom = max(
                float(np.mean([cov["logo"], cov["footer"], cov["drname"]])), 1e-6
            )
            score.bin_selectivity = round(cov["hand"] / denom, 3)

        out.append(score)
    return out
