"""Evaluation harness CLI (Phases 1-3).

    python -m src.eval init-fixtures samples/     # scaffold empty fixtures
    python -m src.eval run                        # Phase 1 preprocessing proxies
    python -m src.eval channels                   # rank channel-separation methods
    python -m src.eval segment                    # Phase 2 segmentation proxies
    python -m src.eval vitals                     # Phase 3 REAL field accuracy

Every subcommand except ``vitals`` is offline in the strict sense: nothing in it
opens a socket. ``vitals`` drives VLM extraction, which talks to a local Ollama
over loopback (see src/extract/client.py for why that still satisfies spec
section 1.1) -- and only when a page has no cached result, or with ``--refresh``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..preprocess.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    ConfigError,
    apply_cli_overrides,
    load_config,
)
from ..preprocess.ingest import discover_inputs, iter_pages, page_id_for
from ..preprocess.pipeline import dump_dir_for, run_pipeline
from .fixtures import (
    FixtureError,
    STATUS_EMPTY,
    load_fixtures,
    scaffold_fixture,
)
from .metrics import PENDING_METRICS, compute_proxy_metrics, score_channel_methods

DEFAULT_EVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "eval.toml"


# ---------------------------------------------------------------------------
# table rendering
# ---------------------------------------------------------------------------


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), line]
    for r in rows:
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def _fmt(v, spec: str = "", dash: str = "-") -> str:
    if v is None:
        return dash
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if spec:
        return format(v, spec)
    return str(v)


# Phases 3 and 4 implement two of spec section 8's metrics. The "NOT IMPLEMENTED"
# tables in the Phase 1 and Phase 2 reports have to stay honest in that direction
# too, so this overrides the blanket "phase 3+" wording for the metrics that now
# exist. Anything absent from this dict is still genuinely not implemented.
_PENDING_METRIC_STATUS = {
    "field_exact_match_accuracy": (
        "IMPLEMENTED",
        "real, every region: python -m src.eval extract "
        "(vitals alone: python -m src.eval vitals)",
    ),
    "field_character_error_rate": (
        "IMPLEMENTED",
        "real, free-text fields: python -m src.eval extract. Numeric fields stay "
        "exact-match by design",
    ),
}

_IMPLEMENTED_METRIC_JSON = {
    "field_exact_match_accuracy": "implemented_see_phase4_extract_metrics_json",
    "field_character_error_rate": "implemented_see_phase4_extract_metrics_json",
}


def _pending_metric_rows() -> list[list[str]]:
    rows = []
    for name in PENDING_METRICS:
        status, blocked = _PENDING_METRIC_STATUS.get(
            name, ("NOT IMPLEMENTED", "requires the flagging rules of spec section 4 "
                   "(phase 5)")
        )
        rows.append([name, status, blocked])
    return rows


def _pending_metric_json() -> dict[str, str]:
    return {
        name: _IMPLEMENTED_METRIC_JSON.get(
            name, "not_implemented_requires_flagging_phase5"
        )
        for name in PENDING_METRICS
    }


def _load_eval_config(path: str | Path):
    """The eval config reuses the preprocess Config loader but has its own
    required sections."""
    import tomllib

    from ..preprocess.config import Config

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"eval config not found: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    cfg = Config(data, p)
    for sec in ("fixtures", "vitals", "extract", "proxy", "channel_probe"):
        cfg.section(sec)
    return cfg


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_init_fixtures(args: argparse.Namespace) -> int:
    cfg = apply_cli_overrides(load_config(args.config), args.set)
    ecfg = _load_eval_config(args.eval_config)
    fixtures_dir = PROJECT_ROOT / str(ecfg.get("fixtures", "dir"))

    files: list[Path] = []
    for i in args.inputs:
        files.extend(discover_inputs(i))
    if not files:
        print("no supported input files found", file=sys.stderr)
        return 2

    created = existing = 0
    for f in files:
        for page in iter_pages(f, cfg):
            try:
                rel = f.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = f
            path, made = scaffold_fixture(
                fixtures_dir, page.page_id, rel, page.page_index, args.overwrite
            )
            created += int(made)
            existing += int(not made)
            print(f"{'created' if made else 'exists '} {path.name}  ({page.describe()})")

    print(f"\n{created} fixture(s) created, {existing} already present in {fixtures_dir}")
    print(
        "Ground truth is EMPTY in every new fixture. Fill it in by reading the "
        "original scans; leave unreadable values null."
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = apply_cli_overrides(load_config(args.config), args.set)
    ecfg = _load_eval_config(args.eval_config)
    fixtures_dir = PROJECT_ROOT / str(ecfg.get("fixtures", "dir"))

    fixtures = load_fixtures(fixtures_dir)
    if not fixtures:
        print(
            f"no fixtures in {fixtures_dir}. Run:\n"
            f"  python -m src.eval init-fixtures samples/",
            file=sys.stderr,
        )
        return 2

    print("MedScribe Local -- eval harness (PHASE 1)")
    print(f"preprocess config : {cfg.path}")
    print(f"eval config       : {ecfg.path}")
    print(f"fixtures          : {len(fixtures)} from {fixtures_dir}")
    print(f"channel sep       : {cfg.get('channel_separation', 'method')}")
    print(f"deskew            : {cfg.get('deskew', 'method')} "
          f"(estimate_on={cfg.get('deskew', 'estimate_on')})")
    print(f"binarise          : {cfg.get('binarize', 'method')}")
    print()

    metrics = []
    with_gt = 0
    for fx in fixtures:
        src = fx.resolve_source(PROJECT_ROOT)
        with_gt += int(fx.has_ground_truth)
        for page in iter_pages(src, cfg):
            if page.page_index != fx.page_index:
                continue
            dump = dump_dir_for(page.page_id) if args.dump_intermediates else None
            result = run_pipeline(page, cfg, dump_dir=dump)
            metrics.append(compute_proxy_metrics(result, cfg, ecfg))

    # ---- proxy table: deskew -------------------------------------------
    print("PREPROCESSING PROXY METRICS (computed on real scans)")
    print()
    print("[1] Deskew -- residual skew re-measured on the printed footer band")
    rows = [
        [
            m.page_id,
            _fmt(m.deskew_estimated_deg, "+.2f"),
            _fmt(m.deskew_applied),
            _fmt(m.deskew_crosscheck_deg, "+.2f"),
            _fmt(m.deskew_crosscheck_disagreement_deg, "+.2f"),
            _fmt(m.skew_before_deg, "+.2f"),
            _fmt(m.skew_after_deg, "+.2f"),
            _fmt(m.skew_reduction_deg, "+.2f"),
        ]
        for m in metrics
    ]
    print(render_table(
        ["page_id", "est_deg", "applied", "xcheck_deg", "xcheck_diff",
         "before_deg", "after_deg", "reduced_deg"], rows
    ))
    print()
    print(f"est_deg     : angle from [deskew].method = "
          f"{cfg.get('deskew', 'method')!r}, measured on the whole page.")
    print(f"xcheck_deg  : second, independent estimate from "
          f"{cfg.get('deskew', 'crosscheck_method')!r}. Recorded only -- it is")
    print("              never averaged into or substituted for est_deg.")
    print("before/after: residual skew of the PRINTED FOOTER band (Hough), which")
    print("              is horizontal by construction on every page of this")
    print("              template. Measured pre- and post-rotation.")
    closed = [m.page_id for m in metrics if m.skew_check_closed_loop]
    if closed:
        print(f"  NOTE: on {len(closed)} page(s) the pipeline took its angle from the "
              f"printed footer")
        print("  band as well, so before/after is a CLOSED-LOOP consistency check "
              "there, not an")
        print("  independent one. Set [deskew].method = \"hough\" to keep the check "
              "independent.")
    else:
        print("              The pipeline never reads this band, so the check is "
              "independent.")
    print()

    # ---- proxy table: contrast -----------------------------------------
    print("[2] Adaptive contrast -- entropy / std / ink-paper contrast, pre vs post CLAHE")
    rows = [
        [
            m.page_id,
            _fmt(m.entropy_before, ".2f"),
            _fmt(m.entropy_after, ".2f"),
            _fmt(m.std_before, ".1f"),
            _fmt(m.std_after, ".1f"),
            _fmt(m.ink_contrast_before, ".0f"),
            _fmt(m.ink_contrast_after, ".0f"),
        ]
        for m in metrics
    ]
    print(render_table(
        ["page_id", "H_pre", "H_post", "std_pre", "std_post", "inkC_pre", "inkC_post"],
        rows,
    ))
    print()

    # ---- proxy table: binarise + crop ----------------------------------
    print("[3] Binarise + crop -- foreground ratio sanity, letterhead band removal")
    rows = [
        [
            m.page_id,
            _fmt(m.fg_ratio, ".4f"),
            _fmt(m.fg_ratio_cropped, ".4f"),
            _fmt(m.crop_top_frac, ".3f"),
            _fmt(m.crop_bottom_frac, ".3f"),
            _fmt(m.crop_retained_area_frac, ".3f"),
            _fmt(m.header_band_removed),
            _fmt(m.footer_band_removed),
            _fmt(m.edge_ink_top, ".3f"),
            _fmt(m.edge_ink_bottom, ".3f"),
        ]
        for m in metrics
    ]
    print(render_table(
        ["page_id", "fg_full", "fg_crop", "crop_top", "crop_bot", "kept_area",
         "hdr_clear", "ftr_clear", "edge_top", "edge_bot"],
        rows,
    ))
    print("hdr_clear/ftr_clear: crop line provably clears the worst-case printed")
    print("band position measured across the corpus (strict; see config/eval.toml).")
    print("edge_top/edge_bot: ink density at the crop lines -- high means content")
    print("runs into the cut and may be clipped.")
    print()

    # ---- warnings -------------------------------------------------------
    warned = [m for m in metrics if m.warnings]
    print(f"[4] Warnings ({len(warned)} of {len(metrics)} page(s))")
    if not warned:
        print("  none")
    for m in warned:
        for w in m.warnings:
            print(f"  {m.page_id}: {w}")
    print()

    # ---- honest statement of what is NOT measured -----------------------
    print("[5] Real eval metrics from spec section 8 -- STATUS")
    print(render_table(["metric", "status", "blocked on"], _pending_metric_rows()))
    print()
    empty = sum(1 for f in fixtures if f.ground_truth_status == STATUS_EMPTY)
    print(
        f"Ground truth: {with_gt}/{len(fixtures)} fixture(s) transcribed, "
        f"{empty} empty. No accuracy figure is produced from empty fixtures."
    )
    print(
        "Phase 1 scope per spec section 9: preprocessing + dumps + eval skeleton. "
        "Region segmentation, VLM extraction, formulary, UI and export are NOT built."
    )

    if args.json_out:
        out = {
            "phase": 1,
            "preprocess_config": str(cfg.path),
            "eval_config": str(ecfg.path),
            "pending_metrics": _pending_metric_json(),
            "pages": [m.as_dict() for m in metrics],
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


def cmd_channels(args: argparse.Namespace) -> int:
    """Rank channel-separation methods on the real scans.

    Reports, per method, the retained handwriting signal and the retained
    letterhead signal, both as SNR over paper noise, plus their ratio
    (selectivity). Higher selectivity = better ink-vs-letterhead separation.
    """
    cfg = apply_cli_overrides(load_config(args.config), args.set)
    ecfg = _load_eval_config(args.eval_config)
    methods = list(ecfg.get("channel_probe", "methods"))

    if args.inputs:
        files: list[Path] = []
        for i in args.inputs:
            files.extend(discover_inputs(i))
    else:
        fixtures_dir = PROJECT_ROOT / str(ecfg.get("fixtures", "dir"))
        files = [fx.resolve_source(PROJECT_ROOT) for fx in load_fixtures(fixtures_dir)]
    if not files:
        print("no inputs", file=sys.stderr)
        return 2

    per_page: dict[str, list] = {}
    for f in files:
        for page in iter_pages(f, cfg):
            per_page[page.page_id] = score_channel_methods(
                page.image, cfg, ecfg, methods, downstream=not args.no_downstream
            )

    min_keep = float(ecfg.get_opt("channel_probe", "min_keep_hand_snr", 0.0))
    print("CHANNEL-SEPARATION COMPARISON on real scans "
          f"({len(per_page)} page(s), {len(methods)} method(s))")
    print()
    print("keep_hand : handwriting stroke SNR over paper noise    -- want HIGH")
    print("lose_*    : letterhead print stroke SNR                -- want LOW")
    print("select    : keep_hand / mean(lose_*)                   -- want HIGH")
    print("Stroke masks come from the source luminance and are identical for every")
    print("method, so no method is graded on a mask it produced itself.")
    print(f"Methods with keep_hand < {min_keep} destroyed the ink and are excluded")
    print("from the ranking (blank output would otherwise score infinitely well).")
    print()

    keys = ["keep_hand", "lose_logo", "lose_footer", "lose_drname", "selectivity",
            "bin_cov_hand", "bin_cov_logo", "bin_cov_footer", "bin_cov_drname",
            "bin_selectivity"]
    agg: dict[str, dict[str, list[float]]] = {m: {k: [] for k in keys} for m in methods}
    usable: dict[str, bool] = {m: True for m in methods}
    for scores in per_page.values():
        for s in scores:
            usable[s.method] = usable[s.method] and s.usable
            for k in keys:
                v = getattr(s, k)
                if v is not None:
                    agg[s.method][k].append(v)

    def mean(m: str, k: str):
        vals = agg[m][k]
        return float(np.mean(vals)) if vals else None

    def worst(m: str, k: str):
        vals = agg[m][k]
        return float(np.min(vals)) if vals else None

    print("[A] Grey-level amplitude selectivity, straight out of channel separation")
    rows = []
    for mth in methods:
        rows.append([
            mth,
            "yes" if usable[mth] else "NO-INK",
            _fmt(mean(mth, "keep_hand"), ".2f"),
            _fmt(mean(mth, "lose_logo"), ".2f"),
            _fmt(mean(mth, "lose_footer"), ".2f"),
            _fmt(mean(mth, "lose_drname"), ".2f"),
            _fmt(mean(mth, "selectivity"), ".3f"),
            _fmt(worst(mth, "selectivity"), ".3f"),
        ])
    rows.sort(key=lambda r: (r[1] == "yes", float(r[6]) if r[6] != "-" else -1),
              reverse=True)
    print(render_table(
        ["method", "usable", "keep_hand", "lose_logo", "lose_footer", "lose_drname",
         "select_mean", "select_worst"],
        rows,
    ))
    print()

    best = None
    if not args.no_downstream:
        print("[B] Post-CLAHE + post-binarisation ink coverage per region "
              "(the decision-relevant test)")
        print("    cov_* = fraction of pixels that survive as ink in that region.")
        print("    Want cov_hand high and the letterhead cov_* low.")
        rows2 = []
        for mth in methods:
            rows2.append([
                mth,
                "yes" if usable[mth] else "NO-INK",
                _fmt(mean(mth, "bin_cov_hand"), ".4f"),
                _fmt(mean(mth, "bin_cov_logo"), ".4f"),
                _fmt(mean(mth, "bin_cov_footer"), ".4f"),
                _fmt(mean(mth, "bin_cov_drname"), ".4f"),
                _fmt(mean(mth, "bin_selectivity"), ".3f"),
                _fmt(worst(mth, "bin_selectivity"), ".3f"),
            ])
        rows2.sort(key=lambda r: (r[1] == "yes", float(r[6]) if r[6] != "-" else -1),
                   reverse=True)
        print(render_table(
            ["method", "usable", "cov_hand", "cov_logo", "cov_footer", "cov_drname",
             "binsel_mean", "binsel_worst"],
            rows2,
        ))
        print()
        ranked = [r for r in rows2 if r[1] == "yes"]
        best = ranked[0][0] if ranked else None
        criterion = "MEAN post-binarisation selectivity"
        # Rank the same methods by worst-page selectivity and by handwriting
        # retention. These routinely disagree with the mean, and picking on the
        # mean alone hides the page that decides whether the tool is usable.
        alt = [
            ("worst-page selectivity", "bin_selectivity", worst),
            ("handwriting retention (cov_hand)", "bin_cov_hand", mean),
        ]
    else:
        ranked = [r for r in rows if r[1] == "yes"]
        best = ranked[0][0] if ranked else None
        criterion = "MEAN grey-level selectivity"
        alt = [
            ("worst-page selectivity", "selectivity", worst),
            ("handwriting retention (keep_hand SNR)", "keep_hand", mean),
        ]

    if best:
        usable_methods = [m for m in methods if usable[m]]
        print(f"Ranked by {criterion}, the top usable method is: {best}")
        for label, key, agg_fn in alt:
            scored = [(agg_fn(m, key), m) for m in usable_methods
                      if agg_fn(m, key) is not None]
            if not scored:
                continue
            scored.sort(reverse=True)
            top = scored[0]
            print(f"Ranked by {label}, it is: {top[1]} ({top[0]:.4f})")
        print()
        print("These criteria do not have to agree, and when they disagree the mean")
        print("is usually the wrong tiebreaker: the worst page is the one that decides")
        print("whether the tool is usable, and retention matters because the")
        print("documented failure mode on these scans is faint ink vanishing rather")
        print("than letterhead surviving. Read the whole table before changing")
        print(f"[channel_separation].method in {cfg.path}.")

    if args.per_page:
        print()
        for pid, scores in per_page.items():
            print(f"--- {pid}")
            prows = [
                [s.method, "yes" if s.usable else "NO-INK",
                 _fmt(s.keep_hand, ".2f"), _fmt(s.lose_logo, ".2f"),
                 _fmt(s.lose_footer, ".2f"), _fmt(s.lose_drname, ".2f"),
                 _fmt(s.selectivity, ".3f"), _fmt(s.bin_cov_hand, ".4f"),
                 _fmt(s.bin_selectivity, ".3f")]
                for s in sorted(scores, key=lambda s: -(s.bin_selectivity
                                                        or s.selectivity or -1))
            ]
            print(render_table(
                ["method", "usable", "keep_hand", "lose_logo", "lose_footer",
                 "lose_drname", "select", "cov_hand", "binsel"], prows))
            print()

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {pid: [s.as_dict() for s in sc] for pid, sc in per_page.items()}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    """Phase 2 region-segmentation proxy metrics on the real scans.

    Deliberately separate from ``run``: Phase 1's tables are about preprocessing
    quality and are not re-derived here, and mixing the two would blur which
    phase a number belongs to.
    """
    from ..segment.anchors import load_templates
    from ..segment.config import (
        ANCHOR_NAMES,
        DEFAULT_SEGMENT_CONFIG_PATH,
        REGION_NAMES,
        check_specs,
        load_segment_config,
    )
    from ..segment.pipeline import PRESCRIPTION_REGION, segment_page
    from .segment_metrics import PENDING_SEGMENT_METRICS, compute_segment_proxy

    cfg = apply_cli_overrides(load_config(args.config), args.set)
    scfg = load_segment_config(args.segment_config or DEFAULT_SEGMENT_CONFIG_PATH)
    templates = load_templates(scfg)

    if args.inputs:
        files: list[Path] = []
        for i in args.inputs:
            files.extend(discover_inputs(i))
    else:
        ecfg = _load_eval_config(args.eval_config)
        fixtures_dir = PROJECT_ROOT / str(ecfg.get("fixtures", "dir"))
        files = [fx.resolve_source(PROJECT_ROOT) for fx in load_fixtures(fixtures_dir)]
    if not files:
        print("no inputs", file=sys.stderr)
        return 2

    src_page = str(scfg.get("templates", "source_page_id"))
    metrics = []
    for f in files:
        for page in iter_pages(f, cfg):
            presult = run_pipeline(page, cfg, dump_dir=None)
            res = segment_page(page, presult, scfg, templates=templates)
            metrics.append(compute_segment_proxy(res, scfg))

    print("MedScribe Local -- eval harness (PHASE 2: region segmentation)")
    print(f"preprocess config : {cfg.path}")
    print(f"segment config    : {scfg.path}")
    print(f"pages             : {len(metrics)} real scan(s)")
    print(f"anchor floors     : min_score={scfg.get('anchors', 'min_score')} "
          f"min_peak_ratio={scfg.get('anchors', 'min_peak_ratio')}")
    print()
    print("SEGMENTATION PROXY METRICS (computed on real scans)")
    print("No region-localisation accuracy and no field accuracy is reported: there")
    print("is no ground truth for either. Everything below is a proxy or a geometric")
    print("residual. See table [6].")
    print()

    def _pid(m):
        return m.page_id + (" *" if m.is_template_source else "")

    # ---- [1] anchors ----------------------------------------------------
    print("[1] Template anchoring -- match confidence and the frame it produced")
    rows = [
        [
            _pid(m),
            _fmt(m.anchor_scores.get("logo"), ".3f"),
            _fmt(m.anchor_peak_ratios.get("logo"), ".2f"),
            _fmt(m.anchor_found.get("logo")),
            _fmt(m.anchor_scores.get("registration_block"), ".3f"),
            _fmt(m.anchor_peak_ratios.get("registration_block"), ".2f"),
            _fmt(m.anchor_found.get("registration_block")),
            _fmt(m.anchor_span_px, ".0f"),
            _fmt(m.frame_scale, ".3f"),
            _fmt(m.anchor_y_residual_px, "+.0f"),
            str(m.frame_source),
        ]
        for m in metrics
    ]
    print(render_table(
        ["page_id", "logo_sc", "logo_pk", "logo_ok", "reg_sc", "reg_pk", "reg_ok",
         "span_px", "scale", "y_resid", "frame"],
        rows,
    ))
    print()
    print("logo_sc/reg_sc: TM_CCOEFF_NORMED peak of the letterhead template match.")
    print("logo_pk/reg_pk: peak / best response outside a neighbourhood of the peak")
    print("               (sharpness). Below the floor the anchor is NOT reported --")
    print("               spec section 1.3, flag rather than guess a position.")
    print("scale        : measured magnification of the printed letterhead on this")
    print("               scan (anchor span / reference anchor span).")
    print(f"* marks the page the templates were cropped from ({src_page}). Its")
    print("  scores are a self-match, not evidence that anchoring generalises;")
    print("  the floors in config/segment.toml were set from the other pages only.")
    print()

    # ---- [2] frame check ------------------------------------------------
    check_names = list(check_specs(scfg))
    if check_names:
        headers = ["page_id"]
        for n in check_names:
            headers += [f"{n[:12]}_sc", "resid_px", "resid_u", "ok"]
        rows = []
        for m in metrics:
            row = [_pid(m)]
            for n in check_names:
                row += [
                    _fmt(m.check_scores.get(n), ".3f"),
                    _fmt(m.check_residual_px.get(n), ".0f"),
                    _fmt(m.check_residual_units.get(n), ".3f"),
                    _fmt(m.check_found.get(n)),
                ]
            rows.append(row)
        print("[2] Anchor-frame accuracy -- printed landmarks the frame did NOT use")
        print(render_table(headers, rows))
        print()
        print("resid_px: distance between where the landmark actually is and where the")
        print("          two-anchor frame predicts it. This is the ONLY accuracy figure")
        print("          Phase 2 produces and it is purely geometric.")
        print("resid_u : the same residual in anchor units, i.e. how far off a region")
        print("          box is expected to be near that landmark.")
        print("These landmarks are never fed back into the frame (Phase 1's crosscheck")
        print("rule), so the residual is an independent measurement.")
        for n in check_names:
            vals = [
                m.check_residual_units[n]
                for m in metrics
                if not m.is_template_source
                and m.check_residual_units.get(n) is not None
            ]
            tol = scfg.section("frame_check")[n].get(
                "max_residual_units", scfg.get("frame_check", "max_residual_units")
            )
            if vals:
                print(f"  {n:<16} residual {min(vals):.3f}-{max(vals):.3f} units "
                      f"(tolerance {tol}); template source page excluded, it is 0 "
                      f"by construction")
        print("The residual grows with distance from the two top-of-page anchors, so a")
        print("region low on the page is less precisely placed than one near them.")
        print()

    # ---- [3] region geometry sanity -------------------------------------
    short = {n: n.split("_")[0][:6] for n in REGION_NAMES}
    print("[3] Region geometry sanity -- non-degenerate, inside the page")
    rows = []
    for m in metrics:
        row = [_pid(m)]
        for n in REGION_NAMES:
            r = next((x for x in m.regions if x.name == n), None)
            if r is None or not r.enabled:
                row.append("off")
            elif not r.geometry_ok:
                row.append("DEGEN")
            elif r.clamped:
                row.append(f"clip{r.clamped_frac * 100:.0f}%")
            else:
                row.append("ok")
        row.append(f"{m.regions_geometry_ok}/{m.regions_total}")
        rows.append(row)
    print(render_table(
        ["page_id", *[short[n] for n in REGION_NAMES], "ok"], rows
    ))
    print("clipN% : the box ran off the page edge and lost N% of its requested area.")
    print("DEGEN  : the box came out empty or below [proxy].min_region_px.")
    print()

    # ---- [4] region ink density -----------------------------------------
    print("[4] Region ink density -- does each crop actually contain content?")
    rows = []
    for m in metrics:
        row = [_pid(m)]
        for n in REGION_NAMES:
            r = next((x for x in m.regions if x.name == n), None)
            if r is None or r.ink_ratio is None:
                row.append("-")
            else:
                row.append(f"{r.ink_ratio:.4f}" + ("" if r.ink_in_band else "!"))
        rows.append(row)
    print(render_table(["page_id", *[short[n] for n in REGION_NAMES]], rows))
    print("! marks a value outside the band measured across the corpus (see")
    print("  [proxy.ink_band] in config/segment.toml). Same idea as Phase 1's")
    print("  fg_ratio band: too low means the box missed its content, too high means")
    print("  it caught printed matter or a scan artefact.")
    for n in REGION_NAMES:
        vals = [
            r.ink_ratio
            for m in metrics
            for r in m.regions
            if r.name == n and r.ink_ratio is not None
        ]
        band = next(
            (r.ink_band for m in metrics for r in m.regions if r.name == n), None
        )
        if vals and band:
            print(f"  {n:<24} observed {min(vals):.4f}-{max(vals):.4f}  "
                  f"band [{band[0]}, {band[1]}]")
    print()

    # ---- [5] prescription line split ------------------------------------
    print(f"[5] Prescription line split -- ink projection over {PRESCRIPTION_REGION}")
    rows = []
    for m in metrics:
        presc = next((r for r in m.regions if r.name == PRESCRIPTION_REGION), None)
        rows.append([
            _pid(m),
            _fmt(None if presc is None else presc.height_px),
            _fmt(None if presc is None else presc.clip_source),
            _fmt(m.line_count),
            _fmt(m.line_split_ok),
            _fmt(m.line_height_min_px),
            _fmt(m.line_height_median_px),
            _fmt(m.line_height_max_px),
        ])
    print(render_table(
        ["page_id", "block_h", "bottom_from", "lines", "trusted", "h_min",
         "h_med", "h_max"],
        rows,
    ))
    print("bottom_from: whether the block's bottom edge came from the measured")
    print("             footer landmark or from the configured box (see")
    print("             [regions.prescription_block].clip_bottom_to_landmark).")
    print("trusted    : False means the band count was implausible, so no line crops")
    print("             are written -- the block is reported unsplit with a reason")
    print("             rather than as a plausible-looking line list.")
    print("There is no ground-truth line count, so 'lines' is a shape measurement,")
    print("not an accuracy. Verify against data/segments/<page_id>/*_line_N.png.")
    print()

    # ---- warnings -------------------------------------------------------
    warned = [m for m in metrics if m.warnings]
    print(f"[6] Warnings ({len(warned)} of {len(metrics)} page(s))")
    if not warned:
        print("  none")
    for m in metrics:
        for w in m.warnings:
            print(f"  {m.page_id}: {w}")
    print()

    # ---- honest statement -----------------------------------------------
    print("[7] Phase 2 metrics that are NOT implemented, and why")
    print(render_table(
        ["metric", "status", "blocked on"],
        [[k, "NOT IMPLEMENTED", v] for k, v in PENDING_SEGMENT_METRICS.items()],
    ))
    print()
    print("[8] Spec section 8 metrics -- STATUS")
    print(render_table(["metric", "status", "blocked on"], _pending_metric_rows()))
    print()
    anchored = sum(1 for m in metrics if m.anchored)
    print(f"Anchored: {anchored}/{len(metrics)} page(s).")
    print(
        "Phase 2 scope per spec section 9 item 2: template-anchored region "
        "segmentation with config-driven geometry. VLM extraction, formulary, "
        "validation, review UI and export are NOT built."
    )

    if args.json_out:
        out = {
            "phase": 2,
            "stage": "region_segmentation",
            "preprocess_config": str(cfg.path),
            "segment_config": str(scfg.path),
            "template_source_page_id": src_page,
            "note": (
                "Proxy metrics and geometric residuals only. No region-localisation "
                "accuracy and no field accuracy: there is no ground truth for "
                "either. The template source page's own anchor scores are a "
                "self-match and are marked is_template_source."
            ),
            "pending_segment_metrics": PENDING_SEGMENT_METRICS,
            "pending_extraction_metrics": _pending_metric_json(),
            "pages": [m.as_dict() for m in metrics],
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


def cmd_vitals(args: argparse.Namespace) -> int:
    """Phase 3: per-field exact-match accuracy for the vitals region, on real scans.

    The first real accuracy figure in this codebase (spec section 8). Deliberately
    a separate subcommand from ``run`` and ``segment`` for the same reason those
    are separate from each other: a number belongs to exactly one phase, and
    interleaving proxy metrics with real accuracy would blur which is which.

    Extraction is driven through ``src.extract`` rather than reimplemented, so the
    values scored here are byte-identical to the ones
    ``python -m src.extract vitals`` produced -- the harness cannot accidentally
    grade a differently-parsed answer.
    """
    from ..extract.client import OllamaClient
    from ..extract.config import (
        DEFAULT_EXTRACT_CONFIG_PATH,
        VITALS_FIELDS,
        VITALS_REGION,
        field_spec,
        fixtures_dir as extract_fixtures_dir,
        load_extract_config,
        ollama_base_url,
        prompt_spec,
        region_spec,
        reject_nested_overrides,
    )
    from ..extract.pipeline import extract_region
    from .fixtures import STATUS_COMPLETE
    from .vitals_metrics import (
        PENDING_VITALS_METRICS,
        score_page,
        summarise_fields,
        summarise_overall,
    )

    ecfg = _load_eval_config(args.eval_config)
    reject_nested_overrides(args.set_extract)
    xcfg = apply_cli_overrides(
        load_extract_config(args.extract_config or DEFAULT_EXTRACT_CONFIG_PATH),
        args.set_extract,
    )

    fixtures_dir = PROJECT_ROOT / str(ecfg.get("fixtures", "dir"))
    warnings: list[str] = []
    if extract_fixtures_dir(xcfg).resolve() != fixtures_dir.resolve():
        # Both configs name a fixtures directory. eval.toml wins here (this is the
        # eval harness), and the disagreement is reported rather than resolved
        # quietly -- a silent choice between two page sets is how an accuracy
        # figure ends up describing a different corpus than the operator thinks.
        warnings.append(
            f"[fixtures].dir in {ecfg.path} is {fixtures_dir} but "
            f"[inputs].fixtures_dir in {xcfg.path} is {extract_fixtures_dir(xcfg)}; "
            f"using the eval config's value. Make them agree."
        )

    all_fixtures = load_fixtures(fixtures_dir)
    if not all_fixtures:
        print(
            f"no fixtures in {fixtures_dir}. Run:\n"
            f"  python -m src.eval init-fixtures samples/",
            file=sys.stderr,
        )
        return 2

    score_partial = bool(ecfg.get("vitals", "score_partial_fixtures"))
    tolerance = float(ecfg.get("vitals", "exact_match_tolerance"))
    allow_inference = bool(ecfg.get("vitals", "run_extraction_if_missing"))

    # spec section 8: never score against un-transcribed ground truth. An empty
    # fixture would otherwise score six null-vs-null matches and print 100%.
    scored = [fx for fx in all_fixtures if fx.has_ground_truth]
    excluded_empty = [fx.page_id for fx in all_fixtures if not fx.has_ground_truth]
    excluded_partial: list[str] = []
    if not score_partial:
        excluded_partial = [
            fx.page_id for fx in scored if fx.ground_truth_status != STATUS_COMPLETE
        ]
        scored = [fx for fx in scored if fx.ground_truth_status == STATUS_COMPLETE]
    if not scored:
        print(
            f"no fixture in {fixtures_dir} has transcribed ground truth, so no "
            f"accuracy can be computed. Spec section 8: never score against "
            f"un-transcribed fixtures.",
            file=sys.stderr,
        )
        return 2

    rspec = region_spec(xcfg, VITALS_REGION)
    pspec = prompt_spec(xcfg, str(rspec["prompt"]))
    decimals = {n: int(field_spec(xcfg, n)["decimals"]) for n in VITALS_FIELDS}

    print("MedScribe Local -- eval harness (PHASE 3: vitals field accuracy)")
    print(f"eval config    : {ecfg.path}")
    print(f"extract config : {xcfg.path}")
    print(f"ollama         : {ollama_base_url(xcfg)}")
    print(f"model          : {xcfg.get('model', 'name')}")
    print(f"prompt         : {rspec['prompt']} (version {pspec['version']})")
    print(f"region         : {rspec['segment_region']}")
    print(f"pages scored   : {len(scored)} of {len(all_fixtures)} fixture(s)")
    print(f"metric         : per-field EXACT match, tolerance {tolerance:g}")
    print()
    if excluded_empty:
        print(f"excluded (ground truth not transcribed): {excluded_empty}")
    if excluded_partial:
        print(f"excluded ([vitals].score_partial_fixtures=false): {excluded_partial}")
    for w in warnings:
        print(f"! {w}")
    print("NOTE: the tables below contain a real patient's observations. They are")
    print("      printed for the operator and written under data/out/, which")
    print("      .gitignore excludes as patient-derived data.")
    print()

    client = OllamaClient(xcfg)
    if allow_inference or args.refresh:
        for note in client.preflight():
            print(f"preflight ok   : {note}")
        print()

    pages = []
    for fx in scored:
        result = extract_region(
            fx.page_id,
            xcfg,
            client,
            region=VITALS_REGION,
            refresh=args.refresh,
            allow_inference=allow_inference or args.refresh,
        )
        pages.append(
            score_page(fx, result, VITALS_FIELDS, decimals, tolerance)
        )

    summaries = summarise_fields(pages, VITALS_FIELDS)
    overall = summarise_overall(pages)

    def _pid(p) -> str:
        return p.page_id + (" ~" if p.partial_ground_truth else "")

    def _v(x) -> str:
        return "." if x is None else str(x)

    # ---- [1] ground truth ----------------------------------------------
    print("[1] Ground truth -- transcribed by a human from the original scan")
    print(render_table(
        ["page_id", *VITALS_FIELDS],
        [[_pid(p), *[_v(p.score(n).ground_truth) for n in VITALS_FIELDS]] for p in pages],
    ))
    print(". = not recorded on that slip (a genuine absence, not a gap in "
          "transcription)")
    if overall["pages_with_partial_ground_truth"]:
        print("~ marks a fixture whose ground_truth_status is 'partial'. On this")
        print("  corpus the only untranscribed field there is the DATE, which the")
        print("  vitals prompt does not ask for, so its six vitals are fully")
        print("  verified -- but in general a null in a partial fixture could mean")
        print("  'not transcribed yet' and the harness cannot tell.")
    print()

    # ---- [2] extracted --------------------------------------------------
    print("[2] Extracted by the model -- values as parsed from its reply")
    rows = []
    for p in pages:
        row = [_pid(p)]
        for n in VITALS_FIELDS:
            s = p.score(n)
            if s.predicted is not None:
                row.append(str(s.predicted))
            elif s.extract_status == "absent":
                row.append(".")
            else:
                row.append(s.extract_status.upper().replace("_", "-"))
        rows.append(row)
    print(render_table(["page_id", *VITALS_FIELDS], rows))
    print(".          = the model reported no value written for that label.")
    print("UPPER-CASE = extraction failed and the value is WITHHELD (spec section")
    print("             1.3, never guess). Nothing is substituted for it.")
    print()

    # ---- [3] outcome grid ------------------------------------------------
    print("[3] Outcome per field")
    print(render_table(
        ["page_id", *VITALS_FIELDS, "correct"],
        [[_pid(p), *[p.score(n).mark for n in VITALS_FIELDS],
          f"{p.n_correct}/{len(p.scores)}"] for p in pages],
    ))
    print("ok         : a recorded value, read exactly right.")
    print("ok-null    : nothing recorded on the slip and none reported -- a real")
    print("             success, scored rather than skipped.")
    print("WRONG      : a value was reported and it is not the ground truth.")
    print("SPURIOUS   : nothing is recorded, but a value was reported.")
    print("MISSED-abs : a value IS recorded; the model said there is none.")
    print("MISSED-wh  : a value IS recorded; extraction withheld what it read.")
    print("WRONG and SPURIOUS are the dangerous outcomes -- a human reviewing the")
    print("draft gets no signal. MISSED-* lose the value but assert nothing false.")
    print()

    # ---- [4] every incorrect field --------------------------------------
    bad = [s for p in pages for s in p.scores if not s.correct]
    print(f"[4] Every field that was not exactly right ({len(bad)})")
    if not bad:
        print("  none")
    else:
        print(render_table(
            ["page_id", "field", "truth", "extracted", "outcome", "why"],
            [[s.page_id, s.name, _v(s.ground_truth), _v(s.predicted), s.outcome,
              s.reason or "-"] for s in bad],
        ))
    print()

    # ---- [5] per-field accuracy ------------------------------------------
    print("[5] PER-FIELD EXACT-MATCH ACCURACY (spec section 8), on real scans")
    print(render_table(
        ["field", "acc", "n_ok", "n", "rec_acc", "rec_ok", "n_rec",
         "wrong", "spur", "miss_abs", "miss_wh"],
        [[
            f.name,
            _fmt(f.accuracy, ".3f"),
            str(f.n_correct),
            str(f.n_scored),
            _fmt(f.recorded_accuracy, ".3f"),
            str(f.n_recorded_correct),
            str(f.n_recorded),
            str(f.counts["wrong"]),
            str(f.counts["spurious"]),
            str(f.counts["missed_absent"]),
            str(f.counts["missed_withheld"]),
        ] for f in summaries],
    ))
    print("acc     : correct / scored, counting a correctly-reported ABSENCE as")
    print("          correct (the ok-null case).")
    print("rec_acc : correct / fields the slip actually records a value for. The")
    print("          harder and more clinically relevant subset -- getting 'no MAC")
    print("          recorded' right is easy, reading 22.65 kg is not.")
    print()

    # ---- [6] overall -----------------------------------------------------
    print("[6] Overall")
    print(render_table(
        ["measure", "value", "n"],
        [
            ["exact-match accuracy (all fields)",
             _fmt(overall["exact_match_accuracy"], ".3f"),
             f"{overall['n_correct']}/{overall['n_fields_scored']}"],
            ["exact-match accuracy (recorded fields only)",
             _fmt(overall["recorded_field_accuracy"], ".3f"),
             f"{overall['n_recorded_correct']}/{overall['n_recorded_on_slip']}"],
            ["emitted a wrong value (no warning to the reviewer)",
             _fmt(overall["emitted_wrong_rate"], ".3f"),
             f"{overall['n_emitted_wrong']}/{overall['n_fields_scored']}"],
            ["withheld (truth lost, nothing false asserted)",
             _fmt(overall["withheld_rate"], ".3f"),
             f"{overall['n_withheld']}/{overall['n_fields_scored']}"],
        ],
    ))
    print()
    print("The last two rows are the distinction spec section 8 cares about most:")
    print("\"a system that is 70% accurate and correctly flags its own 30% is far")
    print("more useful here than one that is 85% accurate and silently wrong\".")
    print("They are NOT flag precision/recall -- there is no flag state until")
    print("phase 5, and this is the raw material for it, not a substitute.")
    print()

    # ---- [7] provenance / cost -------------------------------------------
    print("[7] Provenance and inference cost")
    print(render_table(
        ["page_id", "gt_status", "cached", "wall_s", "json", "reply_field"],
        [[p.page_id, p.ground_truth_status, "yes" if p.from_cache else "no",
          f"{p.duration_s:.2f}", "ok" if p.json_ok else "FAILED",
          p.raw_text_field or "-"] for p in pages],
    ))
    print("cached=yes means the reply came from the spec section 3.5 cache, keyed")
    print("on (crop image hash, region, model, prompt version). Re-run with")
    print("--refresh to re-infer.")
    warned = [(p, w) for p in pages for w in p.warnings]
    print()
    print(f"[8] Warnings ({len(warned)})")
    if not warned:
        print("  none")
    for p, w in warned:
        print(f"  {p.page_id}: {w}")
    print()

    # ---- [9] honest statement of what is still missing -------------------
    print("[9] Spec section 8 metrics -- STATUS after Phase 3")
    print(render_table(
        ["metric", "status", "blocked on"],
        [["field_exact_match_accuracy (vitals)", "IMPLEMENTED",
          "this table; other regions are phase 4"]]
        + [[k, "NOT IMPLEMENTED", v] for k, v in PENDING_VITALS_METRICS.items()],
    ))
    print()
    print(
        "Phase 3 scope per spec section 9 item 3: single-region VLM extraction, "
        "one region type, measured against ground truth. The other regions, "
        "two-pass agreement, the formulary, flagging, the review UI and export "
        "are NOT built."
    )

    out_path = Path(args.json_out or (PROJECT_ROOT / "data" / "out"
                                      / "phase3_vitals_metrics.json"))
    out = {
        "phase": 3,
        "stage": "vlm_field_extraction",
        "region": str(rspec["segment_region"]),
        "eval_config": str(ecfg.path),
        "extract_config": str(xcfg.path),
        "model": str(xcfg.get("model", "name")),
        "prompt": str(rspec["prompt"]),
        "prompt_version": str(pspec["version"]),
        "model_options": dict(xcfg.section("model")["options"]),
        "metric": "per-field exact match",
        "exact_match_tolerance": tolerance,
        "field_decimals": decimals,
        "note": (
            "Real accuracy, measured on real clinic scans against human-transcribed "
            "ground truth (spec section 8 forbids synthetic evaluation). One region "
            "only (vitals column), one prompt, one pass -- spec section 9 item 3. "
            "A null prediction is never a guessed value: it means the field was "
            "reported absent by the model, or withheld because it failed JSON, key, "
            "numeric or plausibility-range validation (spec section 1.3)."
        ),
        "excluded_pages": {
            "ground_truth_empty": excluded_empty,
            "ground_truth_partial": excluded_partial,
        },
        "pending_metrics": PENDING_VITALS_METRICS,
        "per_field": [f.as_dict() for f in summaries],
        "overall": overall,
        "pages": [p.as_dict() for p in pages],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


def _truth_for(fixture, region: str, name: str):
    """The ground-truth value for one region field.

    The only place extraction field names meet fixture keys, and it is a lookup
    rather than a mapping: src/extract/config.py refuses to load a region whose
    field names are not fixture keys, so there is nothing to translate.
    """
    from ..extract.config import VITALS_REGION

    fields = fixture.fields or {}
    if region == VITALS_REGION:
        return (fields.get("vitals") or {}).get(name)
    return fields.get(name)


def _short(v, width: int = 26) -> str:
    """A value as one table cell, truncated with a visible marker."""
    if v is None:
        return "."
    if isinstance(v, list):
        if not v:
            return "[]"
        parts = [x.get("text", "?") if isinstance(x, dict) else str(x) for x in v]
        s = "[" + " | ".join(parts) + "]"
    else:
        s = str(v)
    return s if len(s) <= width else s[: width - 1] + "~"


def cmd_extract(args: argparse.Namespace) -> int:
    """Phase 4: accuracy, CER and two-pass agreement for every region.

    One command rather than one per region, for a reason the per-phase split does
    not cover: the two-pass agreement rate is a cross-region metric (spec section
    4.1 applies it to every region), and computing it in six separate commands
    would mean six partial answers to one question. Per-region output is still
    available with ``--region``.

    ``python -m src.eval vitals`` is left exactly as Phase 3 wrote it. Vitals is
    ALSO scored here, by the shared Phase 4 code, which makes the two tables a
    cross-check on each other: they must agree, and if they ever stop agreeing
    that is a real bug rather than a rounding difference.
    """
    from ..extract.agreement import run_two_pass_region
    from ..extract.client import OllamaClient
    from ..extract.config import (
        DEFAULT_EXTRACT_CONFIG_PATH,
        EXTRACT_REGIONS,
        PRESCRIPTION_LINE_REGION,
        VITALS_REGION,
        field_spec,
        field_type,
        fixtures_dir as extract_fixtures_dir,
        load_extract_config,
        ollama_base_url,
        region_fields,
        region_prompt_names,
        region_spec,
        reject_nested_overrides,
    )
    from ..extract.config import TYPE_NUMBER
    from ..extract.pipeline import line_split_trusted
    from .fixtures import STATUS_COMPLETE
    from .region_metrics import (
        PENDING_REGION_METRICS,
        TextCompare,
        align_prescriptions,
        score_prescription_pairs,
        score_prescription_positional,
        score_region_field,
        summarise_agreement,
        summarise_field,
        summarise_scores,
    )

    ecfg = _load_eval_config(args.eval_config)
    reject_nested_overrides(args.set_extract)
    xcfg = apply_cli_overrides(
        load_extract_config(args.extract_config or DEFAULT_EXTRACT_CONFIG_PATH),
        args.set_extract,
    )

    fixtures_dir = PROJECT_ROOT / str(ecfg.get("fixtures", "dir"))
    warnings: list[str] = []
    if extract_fixtures_dir(xcfg).resolve() != fixtures_dir.resolve():
        warnings.append(
            f"[fixtures].dir in {ecfg.path} is {fixtures_dir} but "
            f"[inputs].fixtures_dir in {xcfg.path} is {extract_fixtures_dir(xcfg)}; "
            f"using the eval config's value. Make them agree."
        )

    regions = args.region or [str(r) for r in ecfg.get("extract", "regions")]
    unknown = [r for r in regions if r not in EXTRACT_REGIONS]
    if unknown:
        print(f"error: unknown region(s) {unknown}. Known: {list(EXTRACT_REGIONS)}",
              file=sys.stderr)
        return 2

    all_fixtures = load_fixtures(fixtures_dir)
    if not all_fixtures:
        print(f"no fixtures in {fixtures_dir}", file=sys.stderr)
        return 2
    score_partial = bool(ecfg.get("extract", "score_partial_fixtures"))
    tolerance = float(ecfg.get("extract", "exact_match_tolerance"))
    allow_inference = bool(ecfg.get("extract", "run_extraction_if_missing"))
    cmp = TextCompare(
        casefold=bool(ecfg.get("extract", "text_compare_casefold")),
        collapse_whitespace=bool(ecfg.get("extract", "text_compare_collapse_whitespace")),
    )
    item_max_cer = float(ecfg.get("extract", "list_item_match_max_cer"))
    product_max_cer = float(
        ecfg.section("extract")["prescription_line"]["product_match_max_cer"]
    )

    # spec section 8: never score against un-transcribed ground truth.
    scored = [fx for fx in all_fixtures if fx.has_ground_truth]
    excluded_empty = [fx.page_id for fx in all_fixtures if not fx.has_ground_truth]
    excluded_partial: list[str] = []
    if not score_partial:
        excluded_partial = [fx.page_id for fx in scored
                            if fx.ground_truth_status != STATUS_COMPLETE]
        scored = [fx for fx in scored if fx.ground_truth_status == STATUS_COMPLETE]
    if not scored:
        print("no fixture has transcribed ground truth; nothing to score",
              file=sys.stderr)
        return 2

    print("MedScribe Local -- eval harness (PHASE 4: all regions, two passes)")
    print(f"eval config    : {ecfg.path}")
    print(f"extract config : {xcfg.path}")
    print(f"ollama         : {ollama_base_url(xcfg)}")
    print(f"model          : {xcfg.get('model', 'name')}")
    print(f"regions        : {', '.join(regions)}")
    print(f"pages scored   : {len(scored)} of {len(all_fixtures)} fixture(s)")
    print(f"text compare   : casefold={cmp.casefold} "
          f"collapse_whitespace={cmp.collapse_whitespace}")
    print()
    if excluded_empty:
        print(f"excluded (ground truth not transcribed): {excluded_empty}")
    if excluded_partial:
        print(f"excluded ([extract].score_partial_fixtures=false): {excluded_partial}")
    for w in warnings:
        print(f"! {w}")
    print("NOTE: the tables below contain real patient data. They are printed for")
    print("      the operator and written under data/out/, which .gitignore")
    print("      excludes as patient-derived.")
    print()

    client = OllamaClient(xcfg)
    if allow_inference or args.refresh:
        for note in client.preflight():
            print(f"preflight ok   : {note}")
        print()

    # -- run ---------------------------------------------------------------
    scores_a: dict[str, list] = {r: [] for r in regions}
    scores_b: dict[str, list] = {r: [] for r in regions}
    positional_scores: list = []
    comparisons: dict[tuple[str, str], list] = {}
    alignments: list = []
    page_rows: dict[str, list[list[str]]] = {r: [] for r in regions}
    run_warnings: list[str] = []
    # Every extraction result (both passes), for the inference-timing table.
    timed: dict[str, list] = {r: [] for r in regions}

    for region in regions:
        fields = region_fields(xcfg, region)
        per_line = bool(region_spec(xcfg, region).get("per_line"))
        for fx in scored:
            if per_line:
                trusted, why = line_split_trusted(fx.page_id, xcfg)
                two_pass = run_two_pass_region(
                    fx.page_id, xcfg, client, region, refresh=args.refresh,
                    allow_inference=allow_inference or args.refresh,
                )
                gt_entries = list((fx.fields or {}).get("prescriptions") or [])
                results_a = [t.pass_a for t in two_pass]
                results_b = [t.pass_b for t in two_pass]
                timed[region].extend(results_a + results_b)
                align = align_prescriptions(
                    fx.page_id, gt_entries, results_a, trusted, why, cmp,
                    product_max_cer,
                )
                alignments.append(align)
                if not trusted:
                    run_warnings.append(
                        f"{fx.page_id}: {why}; excluded from prescription scoring "
                        f"entirely (Phase 2 wrote no line crops)"
                    )
                scores_a[region].extend(score_prescription_pairs(
                    fx.page_id, gt_entries, results_a, align, cmp, fields, "a"))
                align_b = align_prescriptions(
                    fx.page_id, gt_entries, results_b, trusted, why, cmp,
                    product_max_cer,
                )
                scores_b[region].extend(score_prescription_pairs(
                    fx.page_id, gt_entries, results_b, align_b, cmp, fields, "b"))
                if align.positional_eligible:
                    positional_scores.extend(score_prescription_positional(
                        fx.page_id, gt_entries, results_a, cmp, fields, "a"))
                for t in two_pass:
                    for name in fields:
                        comparisons.setdefault((region, name), []).append(
                            (t.page_id, t.line_index, t.fields[name])
                        )
                for t in two_pass:
                    page_rows[region].append([
                        f"{fx.page_id} L{t.line_index}",
                        *[_short(t.pass_a.fields.get(n)) for n in fields],
                        *[_short(t.pass_b.fields.get(n)) for n in fields],
                        f"{t.n_agree}/{t.n_fields}",
                    ])
                continue

            two_pass = run_two_pass_region(
                fx.page_id, xcfg, client, region, refresh=args.refresh,
                allow_inference=allow_inference or args.refresh,
            )
            for t in two_pass:
                timed[region].extend([t.pass_a, t.pass_b])
                for name in fields:
                    ftype = field_type(xcfg, name)
                    decimals = (int(field_spec(xcfg, name)["decimals"])
                                if ftype == TYPE_NUMBER else 0)
                    truth = _truth_for(fx, region, name)
                    sa = score_region_field(fx.page_id, region, name, ftype, truth,
                                            t.pass_a, cmp, item_max_cer, decimals,
                                            tolerance, "a")
                    sb = score_region_field(fx.page_id, region, name, ftype, truth,
                                            t.pass_b, cmp, item_max_cer, decimals,
                                            tolerance, "b")
                    scores_a[region].append(sa)
                    scores_b[region].append(sb)
                    comparisons.setdefault((region, name), []).append(
                        (t.page_id, t.line_index, t.fields[name])
                    )
                page_rows[region].append([
                    fx.page_id + (" ~" if fx.ground_truth_status != STATUS_COMPLETE
                                  else ""),
                    *[_short(_truth_for(fx, region, n)) for n in fields],
                    *[_short(t.pass_a.fields.get(n)) for n in fields],
                    *[_short(t.pass_b.fields.get(n)) for n in fields],
                    f"{t.n_agree}/{t.n_fields}",
                ])
                run_warnings.extend(f"{fx.page_id} {region} pass A: {w}"
                                    for w in t.pass_a.warnings)
                run_warnings.extend(f"{fx.page_id} {region} pass B: {w}"
                                    for w in t.pass_b.warnings)

    # -- [1] what was read --------------------------------------------------
    print("[1] Ground truth vs both passes, per region")
    print()
    for region in regions:
        fields = region_fields(xcfg, region)
        pa, pb = region_prompt_names(xcfg, region)
        print(f"--- {region}  ({region_spec(xcfg, region)['segment_region']}), "
              f"prompts {pa} / {pb}")
        if region == PRESCRIPTION_LINE_REGION:
            print(render_table(
                ["page/line", *[f"A:{f}" for f in fields],
                 *[f"B:{f}" for f in fields], "agree"],
                page_rows[region],
            ))
            print("Ground truth is NOT shown per line here: a line crop has no "
                  "ground-truth")
            print("entry until it is aligned, and the alignment is table [4].")
        else:
            print(render_table(
                ["page_id", *[f"GT:{f}" for f in fields],
                 *[f"A:{f}" for f in fields], *[f"B:{f}" for f in fields], "agree"],
                page_rows[region],
            ))
        print()
    print(". = no value (ground truth: not recorded on the slip; extraction: the")
    print("    model reported nothing there). [] = an explicitly empty list.")
    print("~ after a page_id marks a fixture whose ground_truth_status is partial.")
    print()

    # -- [2] per-field accuracy and CER ------------------------------------
    print("[2] PER-FIELD ACCURACY AND CHARACTER ERROR RATE (spec section 8), pass A")
    rows = []
    summaries: dict[str, list] = {}
    for region in regions:
        summaries[region] = [
            summarise_field(region, n, [s for s in scores_a[region] if s.name == n])
            for n in region_fields(xcfg, region)
        ]
        for f in summaries[region]:
            rows.append([
                region, f.name,
                _fmt(f.accuracy, ".3f"), f"{f.n_correct}/{f.n_scored}",
                _fmt(f.recorded_accuracy, ".3f"),
                f"{f.n_recorded_correct}/{f.n_recorded}",
                _fmt(f.mean_cer, ".3f"), _fmt(f.char_weighted_cer, ".3f"),
                str(f.n_cer_scored),
                str(f.n_emitted_wrong), str(f.n_withheld),
            ])
    print(render_table(
        ["region", "field", "acc", "n_ok/n", "rec_acc", "rec_ok/n_rec",
         "cer_mean", "cer_charw", "n_cer", "wrong", "withheld"],
        rows,
    ))
    print("acc      : correct / scored, counting a correctly-reported ABSENCE as")
    print("           correct. rec_acc: only fields the slip records a value for.")
    print("cer_mean : mean character error rate over the fields where BOTH ground")
    print("           truth and prediction hold a value (n_cer of them). A withheld")
    print("           field is NOT scored as CER 1.0 -- declining to answer is a")
    print("           different behaviour from transcribing badly, and it is counted")
    print("           in the `withheld` column instead.")
    print("cer_charw: total edit distance / total reference characters, so a long")
    print("           diagnosis weighs more than a two-character form abbreviation.")
    print("wrong    : a value was emitted and it is not the truth -- the failure a")
    print("           human reviewing the draft gets no warning about.")
    if PRESCRIPTION_LINE_REGION in regions:
        print()
        print("READ THE prescription_line ROWS WITH TABLE [4]. They are scored over the")
        print("CONTENT-ANCHORED PAIRS ONLY -- the line crops whose product name could be")
        print("paired with a ground-truth entry. A prescription the model missed entirely")
        print("is not in these rows at all; it is counted as gt_unmatched in table [4].")
        print("So this accuracy answers 'given that the drug was found, were its dose and")
        print("frequency read correctly', and table [4] answers 'was it found'. Neither")
        print("number is the whole answer and neither is presented as one.")
    print()

    # -- [3] pass A vs pass B ----------------------------------------------
    print("[3] Pass A vs pass B -- the same fields scored for both prompts")
    rows = []
    for region in regions:
        a, b = summarise_scores(scores_a[region]), summarise_scores(scores_b[region])
        rows.append([
            region,
            _fmt(a["exact_match_accuracy"], ".3f"),
            f"{a['n_correct']}/{a['n_fields_scored']}",
            _fmt(a["mean_cer"], ".3f"),
            _fmt(b["exact_match_accuracy"], ".3f"),
            f"{b['n_correct']}/{b['n_fields_scored']}",
            _fmt(b["mean_cer"], ".3f"),
        ])
    print(render_table(
        ["region", "A acc", "A n_ok/n", "A cer", "B acc", "B n_ok/n", "B cer"], rows
    ))
    print("Both prompts are reported because spec section 4.1 asks for two passes,")
    print("not for a better one. Neither is promoted to 'the answer' anywhere in")
    print("this codebase -- picking between them is phase 5.")
    print()

    # -- [4] prescription alignment ----------------------------------------
    if PRESCRIPTION_LINE_REGION in regions:
        print("[4] Prescription line alignment -- reported, never assumed")
        print(render_table(
            ["page_id", "crops", "gt", "split_ok", "positional?", "matched",
             "gt_unmatched", "extra", "declined", "reason"],
            [[
                a.page_id, str(a.n_line_crops), str(a.n_ground_truth),
                _fmt(a.split_trusted), _fmt(a.positional_eligible),
                str(a.n_matched), str(a.n_missed), str(a.n_spurious),
                str(a.n_null_lines), a.exclusion_reason or "-",
            ] for a in alignments],
        ))
        print("crops/gt    : line crops Phase 2 wrote vs prescriptions in the ground")
        print("              truth. Where they differ, positional scoring is REFUSED")
        print("              for that page and the reason is printed -- a zip() to the")
        print("              shorter list would score every line against the wrong")
        print("              entry and report it as model error.")
        print("matched     : content-anchored pairs (product name, CER <= "
              f"{product_max_cer:g}).")
        print("declined    : crops the model reported as 'not a medicine line' -- the")
        print("              'for <name>:' heading band, mostly. Counted separately:")
        print("              declining a heading is correct behaviour, not a miss.")
        print()
        elig = [a for a in alignments if a.positional_eligible]
        if positional_scores:
            pos = summarise_scores(positional_scores)
            print(f"    Positional scoring, the {len(elig)} eligible page(s) "
                  f"{[a.page_id for a in elig]}:")
            print(f"      exact-match accuracy {_fmt(pos['exact_match_accuracy'], '.3f')} "
                  f"({pos['n_correct']}/{pos['n_fields_scored']} fields), "
                  f"mean CER {_fmt(pos['mean_cer'], '.3f')}")
            print("      This is the STRICT reading: it holds the model to line N of")
            print("      the crop directory being prescription N of the slip. Compare")
            print("      it with the content-anchored numbers in table [2]; a large")
            print("      gap means the line splitter, not the model, is the problem.")
        else:
            print("    Positional scoring: NO eligible pages. Nothing is reported for")
            print("    it rather than a number computed over a mis-paired list.")
        print()

    # -- [5] two-pass agreement --------------------------------------------
    print("[5] TWO-PASS AGREEMENT (spec section 4.1) -- a signal, not a flag")
    correctness: dict[tuple[str, str], dict] = {}
    for region in regions:
        for s in scores_a[region]:
            correctness.setdefault((region, s.name), {})[(s.page_id, s.line_index)] = \
                s.correct
    rows = []
    agreements = []
    for region in regions:
        for name in region_fields(xcfg, region):
            summ = summarise_agreement(
                region, name, comparisons.get((region, name), []),
                correctness.get((region, name)),
            )
            agreements.append(summ)
            rows.append([
                region, name, str(summ.n), _fmt(summ.rate, ".3f"),
                f"{summ.n_agree_both_read}/{summ.n_both_read}",
                _fmt(summ.rate_both_read, ".3f"),
                str(summ.n_both_null),
                str(summ.n_agree_and_correct), str(summ.n_agree_and_incorrect),
                str(summ.n_disagree_and_correct), str(summ.n_disagree_and_incorrect),
            ])
    print(render_table(
        ["region", "field", "n", "agree", "both_read", "agree|read", "both_null",
         "agr+ok", "agr+BAD", "dis+ok", "dis+bad"],
        rows,
    ))
    total_n = sum(a.n for a in agreements)
    total_agree = sum(a.n_agree for a in agreements)
    total_read = sum(a.n_both_read for a in agreements)
    total_agree_read = sum(a.n_agree_both_read for a in agreements)
    total_agr_bad = sum(a.n_agree_and_incorrect for a in agreements)
    total_dis_ok = sum(a.n_disagree_and_correct for a in agreements)
    print()
    print(f"OVERALL two-pass agreement: {total_agree}/{total_n} = "
          f"{(total_agree / total_n) if total_n else float('nan'):.3f}")
    print(f"  restricted to fields where BOTH passes read a value: "
          f"{total_agree_read}/{total_read} = "
          f"{(total_agree_read / total_read) if total_read else float('nan'):.3f}")
    print()
    print("READ THIS NUMBER CAREFULLY. It is not accuracy, not confidence, and not")
    print("flag precision/recall -- there is no flag state anywhere in this codebase")
    print("until phase 5. It is the raw signal spec section 4.1 asks for:")
    print("  agr+BAD : the two passes AGREED and were still wrong "
          f"({total_agr_bad} fields).")
    print("            Agreement does not certify a reading, and this column is how")
    print("            far it can be trusted at all.")
    print("  dis+ok  : the passes disagreed but pass A happened to be right "
          f"({total_dis_ok}).")
    print("            In phase 5 these become false alarms -- a human is asked to")
    print("            check something that was fine. That cost is real and is")
    print("            measured here BEFORE any flagging rule is written.")
    print("  both_null is separated out because two passes that both read nothing")
    print("            agree trivially, and a lot of these fields are legitimately")
    print("            blank. `agree|read` is the honest headline.")
    print()

    # -- [6] every incorrect field -----------------------------------------
    bad = [s for region in regions for s in scores_a[region] if not s.correct]
    print(f"[6] Every pass-A field that was not exactly right ({len(bad)})")
    if not bad:
        print("  none")
    else:
        print(render_table(
            ["page_id", "region", "field", "truth", "extracted", "outcome", "why"],
            [[s.page_id + ("" if s.line_index is None else f" L{s.line_index}"),
              s.region, s.name, _short(s.ground_truth, 30), _short(s.predicted, 30),
              s.outcome, (s.reason or "-")[:70]] for s in bad],
        ))
    print()

    # -- [7] ground-truth caveats ------------------------------------------
    print("[7] GROUND-TRUTH CAVEATS -- numbers above that measure the fixtures,")
    print("    not the model. Recorded here rather than corrected: changing ground")
    print("    truth from inside the eval harness is how a harness stops being one.")
    print(f"  patient_id : {ecfg.section('extract')['header']['note']}")
    print("               NO LONGER A CAVEAT. The fixtures recorded null for all 7")
    print("               while all 7 slips carry an ID; the cause was the")
    print("               transcription template's '(e.g. J-0000):' example slot,")
    print("               which the doctor filled in and the transcriber read as")
    print("               empty. Corrected in the fixtures, so the number above now")
    print("               measures the model. Kept here as the worked example of")
    print("               what this block is for.")
    print(f"  complaints : {ecfg.section('extract')['complaints']['note']}")
    print(f"  date       : {ecfg.section('extract')['date']['note']}")
    print("               6 of the 7 fixtures record a date; 8a83's is null because")
    print("               the transcription sheet left it blank, not because the slip")
    print("               is blank. So this region's denominator is 7 pages but only")
    print("               6 scoreable dates, and a correct read of the 7th counts")
    print("               SPURIOUS -- the patient_id situation, one field over.")
    print("               (The geometry gap that made every date null is FIXED:")
    print("                [regions.date_row] in config/segment.toml, x 1.19-1.64,")
    print("                y 0.148-0.272, read by [regions.date] in the extract")
    print("                config. Dates now come from a crop that contains them.)")
    print()

    # -- [7b] inference timing ---------------------------------------------
    from .timing_metrics import summarise_timing
    timing = {r: summarise_timing(timed[r]) for r in regions}
    timing["ALL"] = summarise_timing([x for r in regions for x in timed[r]])
    print("[7b] INFERENCE TIMING -- medians of Ollama's own counters, per call")
    print(render_table(
        ["region", "calls", "cached", "total_s", "load_s", "prompt_s", "gen_s",
         "prompt_tok", "gen_tok", "prompt_tok/s", "gen_tok/s", "max_total_s"],
        [[r, str(t["n_calls"]), str(t["n_from_cache"]),
          _fmt(t["median_total_s"], ".2f"), _fmt(t["median_load_s"], ".2f"),
          _fmt(t["median_prompt_eval_s"], ".2f"), _fmt(t["median_eval_s"], ".2f"),
          _fmt(t["median_prompt_tokens"], ".0f"), _fmt(t["median_eval_tokens"], ".0f"),
          _fmt(t["median_prompt_tok_per_s"], ".1f"),
          _fmt(t["median_eval_tok_per_s"], ".1f"),
          _fmt(t["max_total_s"], ".2f")] for r, t in timing.items()],
    ))
    print("One call = one region crop, one prompt. total_s is Ollama's")
    print("total_duration (load + prompt eval + generation). Cached results keep the")
    print("counters of the run that produced them, so `cached` calls report past")
    print("timings, not this run's; use --refresh for a fresh measurement.")
    print()

    # -- [8] warnings and pending metrics ----------------------------------
    print(f"[8] Warnings ({len(run_warnings)})")
    if not run_warnings:
        print("  none")
    for w in run_warnings:
        print(f"  {w}")
    print()
    print("[9] Spec section 8 metrics -- STATUS after Phase 4")
    print(render_table(
        ["metric", "status", "blocked on"],
        [
            ["field_exact_match_accuracy", "IMPLEMENTED",
             "every region; table [2]"],
            ["field_character_error_rate", "IMPLEMENTED",
             "free-text fields; table [2]. Numeric fields stay exact-match"],
            ["two_pass_agreement_rate", "IMPLEMENTED",
             "table [5]; spec section 4.1. A signal, not a flag"],
        ]
        + [[k, "NOT IMPLEMENTED", v] for k, v in PENDING_REGION_METRICS.items()],
    ))
    print()
    print("Phase 4 scope per spec section 9 item 4: the remaining regions plus "
          "two-pass\nagreement. The formulary, the validation/flagging state "
          "machine, the review UI\nand export are NOT built, and nothing here "
          "resolves a disagreement or flags a\nfield.")

    out_path = Path(args.json_out or (PROJECT_ROOT / "data" / "out"
                                      / "phase4_extract_metrics.json"))
    out = {
        "phase": 4,
        "stage": "vlm_field_extraction_all_regions",
        "eval_config": str(ecfg.path),
        "extract_config": str(xcfg.path),
        "model": str(xcfg.get("model", "name")),
        "regions": regions,
        "prompts": {r: list(region_prompt_names(xcfg, r)) for r in regions},
        "text_compare": {"casefold": cmp.casefold,
                         "collapse_whitespace": cmp.collapse_whitespace},
        "exact_match_tolerance": tolerance,
        "list_item_match_max_cer": item_max_cer,
        "product_match_max_cer": product_max_cer,
        "note": (
            "Real accuracy on real clinic scans against human-transcribed ground "
            "truth (spec section 8 forbids synthetic evaluation). Two passes per "
            "region with differently-worded prompts (spec section 4.1); the "
            "agreement between them is recorded as a SIGNAL and is never resolved "
            "into a single answer, flagged, or used as a confidence score -- that "
            "is phase 5. A null prediction is never a guessed value."
        ),
        "ground_truth_caveats": {
            "patient_id": str(ecfg.section("extract")["header"]["note"]),
            "complaints_duration": str(ecfg.section("extract")["complaints"]["note"]),
            "date": str(ecfg.section("extract")["date"]["note"]),
        },
        "excluded_pages": {
            "ground_truth_empty": excluded_empty,
            "ground_truth_partial": excluded_partial,
        },
        "pending_metrics": PENDING_REGION_METRICS,
        "per_field_pass_a": [f.as_dict() for r in regions for f in summaries[r]],
        "per_region_overall": {
            r: {"pass_a": summarise_scores(scores_a[r]),
                "pass_b": summarise_scores(scores_b[r])}
            for r in regions
        },
        "two_pass_agreement": {
            "per_field": [a.as_dict() for a in agreements],
            "overall": {
                "n_compared": total_n,
                "n_agree": total_agree,
                "agreement_rate": (total_agree / total_n) if total_n else None,
                "n_both_read": total_read,
                "n_agree_both_read": total_agree_read,
                "agreement_rate_both_read": (
                    (total_agree_read / total_read) if total_read else None
                ),
                "n_agreed_and_wrong": total_agr_bad,
                "n_disagreed_and_pass_a_right": total_dis_ok,
                "note": (
                    "NOT flag precision/recall and NOT a confidence score. There is "
                    "no flag state until phase 5; spec section 4 forbids "
                    "self-reported confidence and this is the independent signal it "
                    "prescribes instead."
                ),
            },
        },
        "prescription_alignment": [a.as_dict() for a in alignments],
        "prescription_positional": (
            summarise_scores(positional_scores) if positional_scores else {
                "n_fields_scored": 0,
                "note": "no page had a trustworthy line split AND a matching count",
            }
        ),
        "inference_timing": timing,
        "scores_pass_a": [s.as_dict() for r in regions for s in scores_a[r]],
        "scores_pass_b": [s.as_dict() for r in regions for s in scores_b[r]],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.eval",
        description="MedScribe Local evaluation harness (Phase 1 skeleton). "
        "Preprocessing proxy metrics only; field accuracy/CER/flag metrics "
        "require extraction and are not implemented yet.",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG_PATH))
    # --set is attached to each subcommand (not the top level) so it can be
    # written after the subcommand name, which is what everyone types.
    overrides = argparse.ArgumentParser(add_help=False)
    overrides.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one preprocess-config value for this run only "
        "(repeatable, TOML values): --set 'channel_separation.suppress_weight=2.0'",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init-fixtures", parents=[overrides], help="scaffold empty ground-truth fixtures")
    i.add_argument("inputs", nargs="+")
    i.add_argument("--overwrite", action="store_true")
    i.set_defaults(func=cmd_init_fixtures)

    r = sub.add_parser("run", parents=[overrides], help="run pipeline over fixtures and print tables")
    r.add_argument("--dump-intermediates", action="store_true")
    r.add_argument("--json-out", default=None)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser(
        "segment",
        parents=[overrides],
        help="Phase 2 region-segmentation proxy metrics (anchor confidence, frame "
        "residual, region sanity, line-split shape)",
    )
    s.add_argument("inputs", nargs="*", help="default: the fixture sources")
    s.add_argument(
        "--segment-config",
        default=None,
        help="Phase 2 segmentation config TOML (default: config/segment.toml)",
    )
    s.add_argument("--json-out", default=None)
    s.set_defaults(func=cmd_segment)

    # No parents=[overrides]: `vitals` never touches the preprocess config (it
    # scores the crops Phase 2 already wrote), so a --set for preprocessing would
    # be accepted and silently ignored. It gets --set-extract instead.
    vt = sub.add_parser(
        "vitals",
        help="Phase 3 REAL per-field exact-match accuracy for the vitals region, "
        "against human-transcribed ground truth on the real scans",
    )
    vt.add_argument(
        "--extract-config",
        default=None,
        help="Phase 3 extraction config TOML (default: config/extract.toml)",
    )
    vt.add_argument(
        "--set-extract",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one EXTRACT config value for this run only (repeatable, "
        "TOML values). This is how a second model is scored for comparison: "
        "--set-extract 'model.name=\"llava:7b\"'",
    )
    vt.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached model replies and re-infer every page",
    )
    vt.add_argument(
        "--json-out",
        default=None,
        help="metrics JSON (default: data/out/phase3_vitals_metrics.json)",
    )
    vt.set_defaults(func=cmd_vitals)

    # Same reasoning as `vitals`: no parents=[overrides]. It scores the crops
    # Phase 2 already wrote, so a preprocess --set would be silently ignored.
    ex = sub.add_parser(
        "extract",
        help="Phase 4 REAL per-field accuracy + CER + two-pass agreement for all "
        "every region, against human-transcribed ground truth on the real scans",
    )
    ex.add_argument(
        "--extract-config",
        default=None,
        help="extraction config TOML (default: config/extract.toml)",
    )
    ex.add_argument(
        "--set-extract",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one EXTRACT config value for this run only (repeatable, "
        "TOML values): --set-extract 'model.name=\"llava:7b\"'",
    )
    ex.add_argument(
        "--region",
        action="append",
        default=[],
        help="score only this region (repeatable); default: [extract].regions "
        "from config/eval.toml",
    )
    ex.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached model replies and re-infer both passes",
    )
    ex.add_argument(
        "--json-out",
        default=None,
        help="metrics JSON (default: data/out/phase4_extract_metrics.json)",
    )
    ex.set_defaults(func=cmd_extract)

    c = sub.add_parser("channels", parents=[overrides], help="rank channel-separation methods empirically")
    c.add_argument("inputs", nargs="*", help="default: the fixture sources")
    c.add_argument("--per-page", action="store_true")
    c.add_argument(
        "--no-downstream",
        action="store_true",
        help="skip the post-CLAHE/post-binarisation coverage test (faster)",
    )
    c.add_argument("--json-out", default=None)
    c.set_defaults(func=cmd_channels)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FixtureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # narrowed below; Phase 3 extraction faults
        # Imported lazily so the Phase 1/2 subcommands keep working even if the
        # extraction package is unavailable.
        from ..extract.client import OllamaError
        from ..extract.pipeline import ExtractError

        if isinstance(exc, ExtractError):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if isinstance(exc, OllamaError):
            print(f"ollama error: {exc}", file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
