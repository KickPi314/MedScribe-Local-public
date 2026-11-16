"""The five preprocessing steps from spec section 3.3.

Each step is a standalone function with the signature::

    step(image, cfg, ctx) -> (image_out, info_dict)

so that the pipeline can enable, disable and reorder them freely, and so each
one can be called directly from a notebook or the eval harness. No step reads a
tuned constant from anywhere except ``cfg``.

Convention: after ``channel_separation`` the working image is single-channel
uint8 with **ink dark, paper bright**. Every later step preserves that.
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def paper_level(gray: np.ndarray) -> float:
    """Robust estimate of the paper (background) grey level."""
    return float(np.percentile(gray, 90))


def ink_mask(gray: np.ndarray, drop: float = 40.0) -> np.ndarray:
    """Rough ink mask: pixels at least ``drop`` levels below paper."""
    return gray < (paper_level(gray) - drop)


# ---------------------------------------------------------------------------
# 1. channel separation
# ---------------------------------------------------------------------------


def _suppressed(
    luma: np.ndarray,
    chroma: np.ndarray,
    weight: float,
    deadzone: float,
    max_suppression: float,
) -> np.ndarray:
    """Subtract a chromatic (letterhead) signal from a luminance ink signal.

    ``luma``   : uint8, ink dark / paper bright.
    ``chroma`` : float32, 0 where neutral, larger where the letterhead's hue is
                 present.

    Both the "simple per-channel subtraction" and the LAB/HSV variants funnel
    through here so they differ only in how ``chroma`` is measured -- which is
    exactly the comparison the brief asks for.
    """
    ink = 255.0 - luma.astype(np.float32)
    chroma = np.maximum(chroma - float(deadzone), 0.0)
    penalty = np.minimum(chroma * float(weight), float(max_suppression))
    ink = np.clip(ink - penalty, 0.0, 255.0)
    return (255.0 - ink).astype(np.uint8)


def _channel_gray(bgr: np.ndarray, cfg) -> np.ndarray:
    return to_gray(bgr)


def _channel_lab_l(bgr: np.ndarray, cfg) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def _channel_lab_a(bgr: np.ndarray, cfg) -> np.ndarray:
    """a* rendered ink-dark: high a* (magenta) becomes dark.

    Included for completeness -- it is the *opposite* of what we want for a
    magenta letterhead, and the eval harness reports it as such.
    """
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.int16)
    return np.clip(255 - (a - 128) * 4, 0, 255).astype(np.uint8)


def _channel_lab_b(bgr: np.ndarray, cfg) -> np.ndarray:
    b = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 2].astype(np.int16)
    return np.clip(255 - (128 - b) * 4, 0, 255).astype(np.uint8)


def _channel_lab_a_suppress(bgr: np.ndarray, cfg) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    luma = lab[:, :, 0]
    magenta = np.maximum(lab[:, :, 1].astype(np.float32) - 128.0, 0.0)
    return _suppressed(
        luma,
        magenta,
        float(cfg.get("channel_separation", "suppress_weight")),
        float(cfg.get("channel_separation", "chroma_deadzone")),
        float(cfg.get("channel_separation", "max_suppression")),
    )


def _channel_hsv_v(bgr: np.ndarray, cfg) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]


def _channel_hsv_hue_suppress(bgr: np.ndarray, cfg) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2]
    center = float(cfg.get("channel_separation", "hue_center"))
    halfwidth = float(cfg.get("channel_separation", "hue_halfwidth"))
    min_sat = float(cfg.get("channel_separation", "min_saturation"))
    # circular hue distance in OpenCV's 0-179 space
    dh = np.abs(h - center)
    dh = np.minimum(dh, 180.0 - dh)
    in_window = np.clip(1.0 - dh / max(halfwidth, 1e-6), 0.0, 1.0)
    chroma = np.where(s >= min_sat, s * in_window, 0.0)
    return _suppressed(
        v,
        chroma,
        float(cfg.get("channel_separation", "suppress_weight")),
        float(cfg.get("channel_separation", "chroma_deadzone")),
        float(cfg.get("channel_separation", "max_suppression")),
    )


def _channel_red(bgr: np.ndarray, cfg) -> np.ndarray:
    """Red plane. Magenta/pink absorbs green, not red, so pink print is bright
    (i.e. suppressed) here while neutral or blue ink stays dark."""
    return bgr[:, :, 2]


def _channel_green(bgr: np.ndarray, cfg) -> np.ndarray:
    """Green plane. Magenta print is maximally *dark* here -- the worst case,
    kept as a control in the comparison."""
    return bgr[:, :, 1]


def _channel_sub_magenta_suppress(bgr: np.ndarray, cfg) -> np.ndarray:
    """Simple per-channel subtraction: magenta-ness measured as (R - G)."""
    b, g, r = cv2.split(bgr)
    luma = to_gray(bgr)
    magenta = np.maximum(r.astype(np.float32) - g.astype(np.float32), 0.0)
    return _suppressed(
        luma,
        magenta,
        float(cfg.get("channel_separation", "suppress_weight")),
        float(cfg.get("channel_separation", "chroma_deadzone")),
        float(cfg.get("channel_separation", "max_suppression")),
    )


CHANNEL_METHODS: dict[str, Callable[[np.ndarray, Any], np.ndarray]] = {
    "gray": _channel_gray,
    "lab_l": _channel_lab_l,
    "lab_a": _channel_lab_a,
    "lab_b": _channel_lab_b,
    "lab_a_suppress": _channel_lab_a_suppress,
    "hsv_v": _channel_hsv_v,
    "hsv_hue_suppress": _channel_hsv_hue_suppress,
    "red_channel": _channel_red,
    "green_channel": _channel_green,
    "sub_magenta_suppress": _channel_sub_magenta_suppress,
}


def channel_separation(img: np.ndarray, cfg, ctx: dict) -> tuple[np.ndarray, dict]:
    method = str(cfg.get("channel_separation", "method"))
    if method not in CHANNEL_METHODS:
        raise ValueError(
            f"unknown channel_separation method {method!r}; "
            f"available: {sorted(CHANNEL_METHODS)}"
        )
    if img.ndim == 2:
        return img, {
            "method": method,
            "skipped": True,
            "reason": "input already single-channel",
        }
    out = CHANNEL_METHODS[method](img, cfg)
    return out, {"method": method, "skipped": False}


# ---------------------------------------------------------------------------
# 2. deskew
# ---------------------------------------------------------------------------


def _binary_for_geometry(gray: np.ndarray) -> np.ndarray:
    """Coarse ink=255 binary used only for geometry estimation."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _line_accept_angle(cfg) -> float:
    """Angle limit for accepting a *single* detected line as a text baseline.

    Deliberately distinct from ``[deskew].max_angle_deg``, which is a
    *page-level* plausibility gate on the final estimate. Conflating the two was
    a real bug: at the page gate's 6.0 deg, the line filter also admitted steep
    handwriting strokes (descenders, prescription-list underlines, the ``for
    <name>:`` bracket) sitting at -2.5..-5.8 deg. On the real corpus those were
    20-30% of all accepted lines and they dragged the unweighted median toward
    zero, so every page was under-corrected and one page (fd1243a1178d_p0) kept
    a 0.49 deg residual. See config/preprocess.toml for the measurements.
    """
    return float(cfg.get("deskew", "hough_max_line_angle_deg"))


def _angle_hough(gray: np.ndarray, cfg) -> tuple[float | None, dict]:
    lo = int(cfg.get("deskew", "hough_canny_lo"))
    hi = int(cfg.get("deskew", "hough_canny_hi"))
    thresh = int(cfg.get("deskew", "hough_threshold"))
    min_frac = float(cfg.get("deskew", "hough_min_line_frac"))
    max_gap = int(cfg.get("deskew", "hough_max_line_gap"))
    max_angle = _line_accept_angle(cfg)

    edges = cv2.Canny(gray, lo, hi, apertureSize=3)
    min_len = max(int(gray.shape[1] * min_frac), 20)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=thresh,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if lines is None or len(lines) == 0:
        return None, {"lines_considered": 0, "lines_used": 0}
    # OpenCV 4.x returns (N, 1, 4); OpenCV 5.x returns (N, 4).
    segs = np.asarray(lines).reshape(-1, 4)
    angles = []
    rejected_steep = 0
    for x1, y1, x2, y2 in segs:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dx == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        # keep near-horizontal lines only
        if abs(ang) <= max_angle:
            angles.append(ang)
        else:
            rejected_steep += 1
    if not angles:
        return None, {
            "lines_considered": int(len(segs)),
            "lines_used": 0,
            "lines_rejected_steep": rejected_steep,
        }
    return float(np.median(angles)), {
        "lines_considered": int(len(segs)),
        "lines_used": len(angles),
        # How much stroke contamination the line filter threw out. A high count
        # here relative to lines_used is the operator's signal that this page's
        # handwriting is unusually slanted.
        "lines_rejected_steep": rejected_steep,
        "line_accept_angle_deg": max_angle,
        "angle_iqr_deg": round(
            float(np.percentile(angles, 75) - np.percentile(angles, 25)), 3
        ),
    }


def _angle_minarearect(gray: np.ndarray, cfg) -> tuple[float | None, dict]:
    kx, ky = cfg.get("deskew", "minarearect_dilate_kernel")
    min_area_frac = float(cfg.get("deskew", "minarearect_min_area_frac"))
    # Same separated limit as Hough: this is per-rect acceptance, not the
    # page-level plausibility gate. See _line_accept_angle.
    max_angle = _line_accept_angle(cfg)

    bw = _binary_for_geometry(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kx), int(ky)))
    joined = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = gray.shape[0] * gray.shape[1] * min_area_frac
    angles, weights = [], []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if w < 1 or h < 1:
            continue
        # Derive the angle from the longest box edge with atan2 rather than from
        # the returned ``angle`` field: OpenCV's angle convention changed between
        # 4.x and 5.x, and reading it directly produced sign-flipped estimates
        # that disagreed with Hough on every real page.
        box = cv2.boxPoints(rect)
        best_len, best_vec = -1.0, None
        for i in range(4):
            v = box[(i + 1) % 4] - box[i]
            length = float(np.hypot(v[0], v[1]))
            if length > best_len:
                best_len, best_vec = length, v
        ang = float(np.degrees(np.arctan2(best_vec[1], best_vec[0])))
        # normalise the undirected edge angle to (-90, 90]
        if ang > 90.0:
            ang -= 180.0
        elif ang <= -90.0:
            ang += 180.0
        if abs(ang) > max_angle:
            continue
        angles.append(ang)
        weights.append(best_len)
    if not angles:
        return None, {"contours": len(contours), "rects_used": 0}
    angles_a = np.array(angles)
    weights_a = np.array(weights, dtype=np.float64)
    order = np.argsort(angles_a)
    cw = np.cumsum(weights_a[order])
    median = float(angles_a[order][np.searchsorted(cw, cw[-1] / 2.0)])
    return median, {"contours": len(contours), "rects_used": len(angles)}


def _projection_score(bw: np.ndarray, angle: float) -> float:
    h, w = bw.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    proj = rot.sum(axis=1, dtype=np.float64)
    return float(np.var(proj))


def _angle_projection(gray: np.ndarray, cfg) -> tuple[float | None, dict]:
    rng = float(cfg.get("deskew", "projection_range_deg"))
    coarse = float(cfg.get("deskew", "projection_coarse_step_deg"))
    fine = float(cfg.get("deskew", "projection_fine_step_deg"))
    bw = _binary_for_geometry(gray)
    # downscale: skew estimation does not need full resolution
    scale = 1000.0 / max(bw.shape)
    if scale < 1.0:
        bw = cv2.resize(
            bw, (int(bw.shape[1] * scale), int(bw.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        bw = (bw > 0).astype(np.uint8) * 255
    if bw.sum() == 0:
        return None, {"reason": "no ink"}
    cands = np.arange(-rng, rng + 1e-9, coarse)
    scores = [_projection_score(bw, a) for a in cands]
    best = float(cands[int(np.argmax(scores))])
    cands2 = np.arange(best - coarse, best + coarse + 1e-9, fine)
    scores2 = [_projection_score(bw, a) for a in cands2]
    best2 = float(cands2[int(np.argmax(scores2))])
    # ``best2`` is the counter-clockwise rotation that best aligns the text rows
    # with the pixel grid, which is exactly the correction ``rotate_image``
    # applies -- the same sign convention the Hough and minAreaRect estimators
    # return. Do not negate it: the eval harness compares the three directly.
    return best2, {"sweep_evaluations": len(scores) + len(scores2)}


def _projection_peak(bw: np.ndarray, rng: float, coarse: float, fine: float
                     ) -> tuple[float, float]:
    """Coarse-then-fine row-projection-variance sweep over ``bw`` (ink=255).

    Returns ``(best_angle, peak_ratio)`` where ``peak_ratio`` is the winning
    score over the mean coarse score -- a measure of how *peaked*, and therefore
    how trustworthy, the response is.

    ``bw`` is zero-padded top and bottom first: without that, rotating a wide
    short strip pushes ink out of frame at the corners, which biases the variance
    toward whichever angle spills the least.
    """
    h, w = bw.shape
    pad = int(abs(np.tan(np.radians(rng))) * w / 2.0) + 4
    bwp = cv2.copyMakeBorder(bw, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    ph, pw = bwp.shape

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, 1.0)
        rot = cv2.warpAffine(bwp, m, (pw, ph), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1, dtype=np.float64)))

    coarse_cands = np.arange(-rng, rng + 1e-9, coarse)
    coarse_scores = np.array([score(a) for a in coarse_cands])
    best_coarse = float(coarse_cands[int(np.argmax(coarse_scores))])
    fine_cands = np.arange(best_coarse - coarse, best_coarse + coarse + 1e-9, fine)
    fine_scores = np.array([score(a) for a in fine_cands])
    best = float(fine_cands[int(np.argmax(fine_scores))])
    baseline = max(float(coarse_scores.mean()), 1e-9)
    return best, float(fine_scores.max() / baseline)


def _angle_printed_band(gray: np.ndarray, cfg) -> tuple[float | None, dict]:
    """Skew from the printed footer band -- the only reliable printed reference
    on this template.

    Why this band: the footer carries three or four full-width printed text rows
    that are horizontal *by construction* on every page of the template, so
    unlike handwriting they cannot legitimately slope. A row-projection-variance
    sweep over them peaks sharply (measured ratio 2.0-2.6 on all 7 real scans).

    Why the band is restricted in x as well as y: the scans have a page-edge
    shadow down the left margin that Otsu turns into a large solid blob. Left in,
    it biased the sweep by up to 0.28 deg between band definitions; excluded, the
    estimate is stable to <0.10 deg across four different band boxes.

    Why the printed *header* is not used instead: measured, and rejected. The
    header block has too few printed rows and is overlaid by the pink logo
    watermark plus the handwritten date/vitals column; its sweep peak ratio is
    only 1.1-1.4 (i.e. no real peak) and it disagreed with the footer by up to
    3.8 deg. There is no second usable printed reference on this template.

    Returns ``None`` (rather than a guess) when the peak is not sharp enough to
    trust -- spec section 1.3.
    """
    x0f, x1f, y0f, y1f = cfg.get("deskew", "printed_band_box")
    rng = float(cfg.get("deskew", "printed_band_range_deg"))
    coarse = float(cfg.get("deskew", "printed_band_coarse_step_deg"))
    fine = float(cfg.get("deskew", "printed_band_fine_step_deg"))
    min_peak = float(cfg.get("deskew", "printed_band_min_peak_ratio"))

    h, w = gray.shape
    y0, y1 = int(round(h * float(y0f))), int(round(h * float(y1f)))
    x0, x1 = int(round(w * float(x0f))), int(round(w * float(x1f)))
    y0, y1 = max(y0, 0), min(y1, h)
    x0, x1 = max(x0, 0), min(x1, w)
    if y1 - y0 < 20 or x1 - x0 < 100:
        return None, {"reason": "printed_band_box degenerate for this page size"}

    strip = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(strip, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.sum() == 0:
        return None, {"reason": "no ink in printed reference band"}

    best, peak = _projection_peak(bw, rng, coarse, fine)
    detail = {
        "band_box": [float(x0f), float(x1f), float(y0f), float(y1f)],
        "band_px": [int(y0), int(y1), int(x0), int(x1)],
        "peak_ratio": round(peak, 3),
        "min_peak_ratio": min_peak,
    }
    if peak < min_peak:
        detail["reason"] = (
            f"printed reference band peak ratio {peak:.2f} below "
            f"min_peak_ratio {min_peak}; refusing to guess an angle"
        )
        return None, detail
    return best, detail


ANGLE_ESTIMATORS = {
    "hough": _angle_hough,
    "minarearect": _angle_minarearect,
    "projection": _angle_projection,
    "printed_band": _angle_printed_band,
}


def estimate_skew(gray: np.ndarray, cfg, method: str | None = None) -> tuple[float | None, dict]:
    """Estimate skew in degrees (positive = content rotated clockwise)."""
    method = method or str(cfg.get("deskew", "method"))
    if method == "none":
        # method="none" means "measure and report, but do not rotate", so it
        # still needs an estimator. Which one is a config decision, not a
        # hardcoded one: this used to be pinned to "projection", the least
        # accurate of the four (it reports -3.90 deg on a page that is square to
        # within 0.05 deg), so the number the operator was shown in measure-only
        # mode was the worst available.
        method = str(cfg.get("deskew", "measure_only_method"))
        if method == "none":
            raise ValueError(
                "[deskew].measure_only_method must name a real estimator, not "
                "'none'; it is the estimator used when method='none'"
            )
    if method not in ANGLE_ESTIMATORS:
        raise ValueError(
            f"unknown deskew method {method!r}; available: "
            f"{sorted(ANGLE_ESTIMATORS)} or 'none'"
        )
    return ANGLE_ESTIMATORS[method](gray, cfg)


def rotate_image(img: np.ndarray, angle: float, border: int | None = None) -> np.ndarray:
    """Rotate by ``-angle`` (i.e. correct a skew of ``angle``), keeping size."""
    h, w = img.shape[:2]
    if border is None:
        border = 255  # paper
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    flags = cv2.INTER_CUBIC if img.dtype == np.uint8 else cv2.INTER_LINEAR
    bval = (border,) * (1 if img.ndim == 2 else 3)
    return cv2.warpAffine(
        img, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=bval
    )


def deskew(img: np.ndarray, cfg, ctx: dict) -> tuple[np.ndarray, dict]:
    method = str(cfg.get("deskew", "method"))
    estimate_on = str(cfg.get("deskew", "estimate_on"))
    max_angle = float(cfg.get("deskew", "max_angle_deg"))
    min_angle = float(cfg.get("deskew", "min_angle_deg"))

    if estimate_on == "source_gray":
        src = ctx.get("source_gray")
        if src is None:
            src = to_gray(img)
    elif estimate_on == "current":
        src = to_gray(img)
    else:
        raise ValueError(
            f"[deskew].estimate_on must be 'current' or 'source_gray', got {estimate_on!r}"
        )

    # The estimator that actually ran. For method="none" that is
    # measure_only_method, not the literal "none" -- the fallback and cross-check
    # comparisons below must test against the real estimator or they would
    # pointlessly re-run it against itself.
    effective_method = (
        method if method != "none" else str(cfg.get("deskew", "measure_only_method"))
    )

    angle, detail = estimate_skew(src, cfg, effective_method)
    info: dict[str, Any] = {
        "method": method,
        "effective_method": effective_method,
        "estimate_on": estimate_on,
        "estimated_angle_deg": None if angle is None else round(angle, 3),
        "applied": False,
        **detail,
    }

    # -- fallback ---------------------------------------------------------
    # When the primary estimator declines to answer (it returns None rather than
    # guessing), try the configured fallback before giving up on the page.
    fallback = str(cfg.get("deskew", "fallback_method"))
    if angle is None and fallback != "none" and fallback != effective_method:
        fb_angle, fb_detail = estimate_skew(src, cfg, fallback)
        info["fallback_method"] = fallback
        info["fallback_angle_deg"] = None if fb_angle is None else round(fb_angle, 3)
        info["fallback_detail"] = fb_detail
        if fb_angle is not None:
            angle = fb_angle
            info["estimated_angle_deg"] = round(angle, 3)
            info["angle_source"] = f"fallback:{fallback}"
    elif angle is not None:
        info["angle_source"] = f"primary:{effective_method}"

    # -- independent cross-check -----------------------------------------
    # Measured with a different estimator and recorded, never used to silently
    # replace the primary estimate. The point is that a disagreement reaches the
    # manifest and the eval table instead of being averaged away.
    xmethod = str(cfg.get("deskew", "crosscheck_method"))
    if xmethod != "none" and xmethod != effective_method and angle is not None:
        x_angle, _ = estimate_skew(src, cfg, xmethod)
        info["crosscheck_method"] = xmethod
        info["crosscheck_angle_deg"] = None if x_angle is None else round(x_angle, 3)
        if x_angle is not None:
            disagree = abs(x_angle - angle)
            info["crosscheck_disagreement_deg"] = round(disagree, 3)
            warn = float(cfg.get("deskew", "crosscheck_warn_deg"))
            info["crosscheck_warn_deg"] = warn
            info["crosscheck_disagrees"] = bool(disagree > warn)

    if angle is None:
        info["reason"] = "no reliable angle estimate"
        return img, info
    if method == "none":
        info["reason"] = "method='none' (measured, not applied)"
        return img, info
    if abs(angle) > max_angle:
        info["reason"] = f"|angle| > max_angle_deg ({max_angle})"
        return img, info
    if abs(angle) < min_angle:
        info["reason"] = f"|angle| < min_angle_deg ({min_angle})"
        return img, info

    out = rotate_image(img, angle)
    info["applied"] = True
    ctx["applied_skew_deg"] = angle
    if ctx.get("source_gray") is not None:
        ctx["source_gray_deskewed"] = rotate_image(ctx["source_gray"], angle)
    return out, info


# ---------------------------------------------------------------------------
# 3. adaptive contrast (CLAHE)
# ---------------------------------------------------------------------------


def clahe(img: np.ndarray, cfg, ctx: dict) -> tuple[np.ndarray, dict]:
    gray = to_gray(img)
    prefilter = str(cfg.get("clahe", "prefilter"))
    ksize = _odd(int(cfg.get("clahe", "prefilter_ksize")))
    if prefilter == "median":
        gray = cv2.medianBlur(gray, ksize)
    elif prefilter == "bilateral":
        gray = cv2.bilateralFilter(gray, ksize, 40, 40)
    elif prefilter != "none":
        raise ValueError(f"unknown [clahe].prefilter {prefilter!r}")

    clip = float(cfg.get("clahe", "clip_limit"))
    tx, ty = cfg.get("clahe", "tile_grid")
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(int(tx), int(ty)))
    out = op.apply(gray)
    return out, {
        "clip_limit": clip,
        "tile_grid": [int(tx), int(ty)],
        "prefilter": prefilter,
    }


# ---------------------------------------------------------------------------
# 4. binarise
# ---------------------------------------------------------------------------


def sauvola_threshold(gray: np.ndarray, window: int, k: float, r: float) -> np.ndarray:
    """Sauvola local threshold via box filters (no skimage dependency)."""
    win = _odd(window)
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, cv2.CV_32F, (win, win), normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    sqmean = cv2.boxFilter(g * g, cv2.CV_32F, (win, win), normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))
    return mean * (1.0 + k * (std / r - 1.0))


def niblack_threshold(gray: np.ndarray, window: int, k: float) -> np.ndarray:
    win = _odd(window)
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, cv2.CV_32F, (win, win), normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    sqmean = cv2.boxFilter(g * g, cv2.CV_32F, (win, win), normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))
    return mean + k * std


def _despeckle(bw_inv: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    """Drop ink components smaller than ``min_area``. ``bw_inv``: ink=255."""
    if min_area <= 0:
        return bw_inv, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw_inv, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[0] = False
    removed = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[i] = True
        else:
            removed += 1
    out = np.where(keep[labels], 255, 0).astype(np.uint8)
    return out, removed


def binarize(img: np.ndarray, cfg, ctx: dict) -> tuple[np.ndarray, dict]:
    gray = to_gray(img)
    method = str(cfg.get("binarize", "method"))
    window = int(cfg.get("binarize", "window"))
    k = float(cfg.get("binarize", "k"))

    if method == "sauvola":
        thr = sauvola_threshold(gray, window, k, float(cfg.get("binarize", "r")))
        bw = np.where(gray.astype(np.float32) > thr, 255, 0).astype(np.uint8)
    elif method == "niblack":
        thr = niblack_threshold(gray, window, k)
        bw = np.where(gray.astype(np.float32) > thr, 255, 0).astype(np.uint8)
    elif method in ("adaptive_gaussian", "adaptive_mean"):
        block = _odd(int(cfg.get("binarize", "adaptive_block")))
        c = float(cfg.get("binarize", "adaptive_c"))
        atype = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "adaptive_gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        bw = cv2.adaptiveThreshold(gray, 255, atype, cv2.THRESH_BINARY, block, c)
    else:
        raise ValueError(
            f"unknown [binarize].method {method!r}; available: sauvola, niblack, "
            f"adaptive_gaussian, adaptive_mean (global thresholds are not offered)"
        )

    ink = cv2.bitwise_not(bw)
    min_area = int(cfg.get("binarize", "min_component_area"))
    ink, removed = _despeckle(ink, min_area)
    out = cv2.bitwise_not(ink)
    fg_ratio = float((ink > 0).mean())
    return out, {
        "method": method,
        "window": window,
        "k": k,
        "components_removed": removed,
        "foreground_ratio": round(fg_ratio, 5),
    }


# ---------------------------------------------------------------------------
# 5. crop header / footer bands
# ---------------------------------------------------------------------------


def _row_ink_profile(img: np.ndarray) -> np.ndarray:
    """Fraction of ink pixels per row, ink = dark."""
    gray = to_gray(img)
    if set(np.unique(gray)[:3].tolist()) <= {0, 255} and len(np.unique(gray)) <= 2:
        ink = gray < 128
    else:
        ink = ink_mask(gray, drop=40.0)
    return ink.mean(axis=1)


def _empty_runs(profile: np.ndarray, lo: int, hi: int, row_thresh: float) -> list[tuple[int, int]]:
    """Runs of near-empty rows inside ``[lo, hi)``, as absolute row indices."""
    band = profile[lo:hi] <= row_thresh
    out: list[tuple[int, int]] = []
    start = None
    for i, empty in enumerate(band):
        if empty and start is None:
            start = i
        elif not empty and start is not None:
            out.append((lo + start, lo + i))
            start = None
    if start is not None:
        out.append((lo + start, lo + len(band)))
    return out


def crop(img: np.ndarray, cfg, ctx: dict) -> tuple[np.ndarray, dict]:
    """Remove the printed header and footer bands.

    Header and footer use separate methods on purpose: measurements on the real
    scans showed the header boundary moves by ~1.5% of page height between
    pages (the letterhead carries a variable number of registration lines) and
    is worth detecting, while the footer boundary is stable to ~1% and the
    body content sometimes runs right up to it, so detecting it there only adds
    a way to clip handwriting.

    Safety rule for both: the configured fixed fraction is a **floor**, and
    detection may only ever crop *more* letterhead away, never less. That way a
    detection failure degrades to the fixed fraction instead of leaving printed
    matter in, and can never eat body content that the fixed fraction kept.
    """
    h, w = img.shape[:2]
    header_method = str(cfg.get("crop", "header_method"))
    footer_method = str(cfg.get("crop", "footer_method"))
    hf = float(cfg.get("crop", "header_frac"))
    hmax = float(cfg.get("crop", "header_max_frac"))
    ff = float(cfg.get("crop", "footer_frac"))
    lf = float(cfg.get("crop", "left_frac"))
    rf = float(cfg.get("crop", "right_frac"))
    margin = float(cfg.get("crop", "header_safety_margin_frac"))

    for name, value in (("header_method", header_method), ("footer_method", footer_method)):
        if value not in ("projection", "fixed_fraction", "none"):
            raise ValueError(
                f"unknown [crop].{name} {value!r}; available: projection, "
                f"fixed_fraction, none"
            )

    top_fixed = int(round(h * hf))
    bottom_fixed = int(round(h * (1.0 - ff)))
    left = int(round(w * lf))
    right = int(round(w * (1.0 - rf)))

    info: dict[str, Any] = {
        "header_method": header_method,
        "footer_method": footer_method,
    }

    if header_method == "none" and footer_method == "none":
        info["applied"] = False
        info["reason"] = "both header_method and footer_method are 'none'"
        return img, info

    top = 0 if header_method == "none" else top_fixed
    bottom = h if footer_method == "none" else bottom_fixed

    if header_method == "projection" or footer_method == "projection":
        profile = _row_ink_profile(img)
        min_gutter = max(int(round(h * float(cfg.get("crop", "min_gutter_frac")))), 2)
        row_thresh = float(cfg.get("crop", "row_ink_threshold"))

        if header_method == "projection":
            hs = cfg.get("crop", "header_search")
            runs = [
                r
                for r in _empty_runs(profile, int(h * float(hs[0])), int(h * float(hs[1])),
                                     row_thresh)
                if r[1] - r[0] >= min_gutter
            ]
            # First sufficient gutter from the top = the boundary below the
            # printed header block. (The *widest* gutter is not it: on these
            # scans the widest gap in the search window is often below the
            # first handwritten line.)
            if runs:
                cut = runs[0][1] - int(round(h * margin))
                capped = min(cut, int(round(h * hmax)))
                top = max(capped, top_fixed)  # fixed fraction is a floor
                info["header_gutter_found"] = True
                info["header_gutter_frac"] = [
                    round(runs[0][0] / h, 4), round(runs[0][1] / h, 4)
                ]
                info["header_cut_source"] = (
                    "projection" if top > top_fixed else "fixed_floor"
                )
            else:
                info["header_gutter_found"] = False
                info["header_cut_source"] = "fixed_fraction (no gutter found)"

        if footer_method == "projection":
            fs = cfg.get("crop", "footer_search")
            runs = [
                r
                for r in _empty_runs(profile, int(h * float(fs[0])), int(h * float(fs[1])),
                                     row_thresh)
                if r[1] - r[0] >= min_gutter
            ]
            if runs:
                # Last gutter in the window = the gap immediately above the
                # printed footer block.
                bottom = min(runs[-1][0], bottom_fixed)  # fixed fraction is a floor
                info["footer_gutter_found"] = True
                info["footer_cut_source"] = (
                    "projection" if bottom < bottom_fixed else "fixed_floor"
                )
            else:
                info["footer_gutter_found"] = False
                info["footer_cut_source"] = "fixed_fraction (no gutter found)"

    if bottom - top < h * 0.25 or right - left < w * 0.25:
        info["applied"] = False
        info["reason"] = "crop would discard >75% of the page; refusing"
        return img, info

    out = img[top:bottom, left:right]
    info.update(
        applied=True,
        top_px=int(top),
        bottom_px=int(bottom),
        left_px=int(left),
        right_px=int(right),
        top_frac=round(top / h, 4),
        bottom_frac=round(bottom / h, 4),
        retained_area_frac=round(((bottom - top) * (right - left)) / (h * w), 4),
    )
    ctx["crop_box"] = (top, bottom, left, right)
    return out, info


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

STEP_FUNCS: dict[str, Callable[[np.ndarray, Any, dict], tuple[np.ndarray, dict]]] = {
    "channel_separation": channel_separation,
    "deskew": deskew,
    "clahe": clahe,
    "binarize": binarize,
    "crop": crop,
}
