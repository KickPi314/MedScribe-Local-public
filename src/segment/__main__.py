"""CLI for region segmentation (Phase 2).

    python -m src.segment run samples/ --dump-crops
    python -m src.segment anchors samples/          # anchor confidence only
    python -m src.segment make-templates samples/   # regenerate template assets

Offline only: nothing here opens a socket.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..preprocess.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    apply_cli_overrides,
    load_config,
)
from ..preprocess.ingest import discover_inputs, iter_pages
from ..preprocess.pipeline import _write_png, run_pipeline
from .anchors import (
    build_frame,
    check_landmarks,
    deskewed_source_gray,
    load_templates,
    match_anchors,
)
from .config import (
    ANCHOR_NAMES,
    DEFAULT_SEGMENT_CONFIG_PATH,
    check_specs,
    load_segment_config,
    template_path,
    template_specs,
    templates_dir,
)
from .pipeline import (
    DEFAULT_SEGMENT_ROOT,
    PRESCRIPTION_REGION,
    segment_dir_for,
    segment_page,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="PDF/image file(s) or directory/directories")
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Phase 1 preprocess config TOML (default: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument(
        "--segment-config",
        default=str(DEFAULT_SEGMENT_CONFIG_PATH),
        help=f"Phase 2 segmentation config TOML (default: {DEFAULT_SEGMENT_CONFIG_PATH})",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one SEGMENT config value for this run only (repeatable, "
        "TOML values): --set 'anchors.min_score=0.5'",
    )
    p.add_argument(
        "--set-preprocess",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one PREPROCESS config value for this run only",
    )


def _load_both(args):
    cfg = apply_cli_overrides(load_config(args.config), args.set_preprocess)
    scfg = apply_cli_overrides(load_segment_config(args.segment_config), args.set)
    return cfg, scfg


def _files(args) -> list[Path]:
    files: list[Path] = []
    for i in args.inputs:
        files.extend(discover_inputs(i))
    return files


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg, scfg = _load_both(args)
    files = _files(args)
    if not files:
        print("no supported input files found", file=sys.stderr)
        return 2

    templates = load_templates(scfg)
    dump_root = Path(args.dump_root) if args.dump_root else DEFAULT_SEGMENT_ROOT

    print(f"preprocess config : {cfg.path}")
    print(f"segment config    : {scfg.path}")
    print(f"templates         : {templates_dir(scfg)}")
    for n in ANCHOR_NAMES:
        t = templates[n]
        print(f"  {n:<20} {t.shape[1]}x{t.shape[0]}px at reference dpi "
              f"{scfg.get('templates', 'reference_dpi')}")
    print(f"anchor floors     : min_score={scfg.get('anchors', 'min_score')} "
          f"min_peak_ratio={scfg.get('anchors', 'min_peak_ratio')}")
    print(f"crop source stage : {scfg.get('source', 'stage')}")
    print(f"inputs            : {len(files)} file(s)")
    if args.dump_crops:
        print(f"dumping crops to  : {dump_root}")
    print()

    n_pages = anchored = 0
    for f in files:
        for page in iter_pages(f, cfg):
            presult = run_pipeline(page, cfg, dump_dir=None)
            dump_dir = (
                segment_dir_for(page.page_id, dump_root) if args.dump_crops else None
            )
            res = segment_page(page, presult, scfg, dump_dir=dump_dir, templates=templates)
            n_pages += 1
            anchored += int(res.anchored)

            lg = res.anchors.get("logo")
            rg = res.anchors.get("registration_block")

            def _a(m):
                if m is None or m.score is None:
                    return "n/a"
                return f"{m.score:.3f}/{m.peak_ratio:.1f}{'' if m.found else '!'}"

            if res.anchored:
                frame_txt = (
                    f"frame={res.frame.source} span={res.frame.span_px:.0f}px "
                    f"scale={res.frame.scale:.3f}"
                )
                ok = sum(1 for r in res.regions if r.geometry_ok)
                nlines = len(res.lines) if res.line_info.get("split_ok") else 0
                region_txt = (
                    f"regions={ok}/{len(res.regions)} "
                    f"lines={nlines if nlines else 'UNSPLIT'}"
                )
            else:
                frame_txt = "frame=UNANCHORED"
                region_txt = "regions=0 lines=0"

            print(
                f"{page.page_id} logo={_a(lg)} reg={_a(rg)} {frame_txt} {region_txt} "
                f"[{res.duration_s:.2f}s]"
            )
            for w in res.warnings:
                print(f"    ! {w}")

    print(f"\n{n_pages} page(s) processed, {anchored} anchored, "
          f"{n_pages - anchored} unanchored.")
    if args.dump_crops:
        print(f"Crops       : {dump_root}/<page_id>/<region_name>.png")
        print(f"Lines       : {dump_root}/<page_id>/{PRESCRIPTION_REGION}_line_N.png")
        print(f"Overlay     : {dump_root}/<page_id>/overlay.png  (anchor + region boxes)")
        print(f"Manifest    : {dump_root}/<page_id>/manifest.json")
    print(
        "No field values are extracted here and no accuracy figure is produced: "
        "Phase 2 is geometry only. Proxy signals: python -m src.eval segment"
    )
    return 0


# ---------------------------------------------------------------------------
# anchors (probe)
# ---------------------------------------------------------------------------


def cmd_anchors(args: argparse.Namespace) -> int:
    """Anchor-confidence probe -- the table the confidence floors were set from."""
    from ..eval.__main__ import render_table

    cfg, scfg = _load_both(args)
    files = _files(args)
    if not files:
        print("no supported input files found", file=sys.stderr)
        return 2
    templates = load_templates(scfg)
    src_page = str(scfg.get("templates", "source_page_id"))

    rows = []
    check_rows = []
    check_names = list(check_specs(scfg))
    for f in files:
        for page in iter_pages(f, cfg):
            presult = run_pipeline(page, cfg, dump_dir=None)
            gray = deskewed_source_gray(page, presult)
            matches = match_anchors(
                gray, scfg, page.effective_dpi, templates=templates, page_id=page.page_id
            )
            frame = build_frame(matches, scfg)
            checks = {
                c.name: c
                for c in check_landmarks(
                    gray, frame, scfg, page.effective_dpi, templates=templates
                )
            }
            crow = [page.page_id + (" *" if page.page_id == src_page else "")]
            for n in check_names:
                c = checks.get(n)
                crow += [
                    f"{c.score:.3f}" if c and c.score is not None else "-",
                    f"{c.residual_x_px:+.0f}" if c and c.residual_x_px is not None else "-",
                    f"{c.residual_y_px:+.0f}" if c and c.residual_y_px is not None else "-",
                    f"{c.residual_units:.3f}" if c and c.residual_units is not None else "-",
                ]
            check_rows.append(crow)
            lg, rg = matches["logo"], matches["registration_block"]
            rows.append([
                page.page_id + (" *" if page.page_id == src_page else ""),
                f"{lg.score:.3f}" if lg.score is not None else "-",
                f"{lg.peak_ratio:.2f}" if lg.peak_ratio is not None else "-",
                f"{lg.sweep_factor:+.2f}" if lg.sweep_factor is not None else "-",
                "yes" if lg.found else "NO",
                f"{rg.score:.3f}" if rg.score is not None else "-",
                f"{rg.peak_ratio:.2f}" if rg.peak_ratio is not None else "-",
                f"{rg.sweep_factor:+.2f}" if rg.sweep_factor is not None else "-",
                "yes" if rg.found else "NO",
                f"{frame.span_px:.0f}" if frame.span_px else "-",
                f"{frame.scale:.3f}" if frame.scale else "-",
                f"{frame.anchor_y_residual_px:+.0f}"
                if frame.anchor_y_residual_px is not None
                else "-",
                frame.source,
            ])

    print("TEMPLATE-ANCHOR CONFIDENCE on real scans")
    print()
    print("[1] The two frame anchors")
    print(render_table(
        ["page_id", "logo_sc", "logo_pk", "logo_f", "logo_ok",
         "reg_sc", "reg_pk", "reg_f", "reg_ok",
         "span_px", "scale", "y_resid", "frame"],
        rows,
    ))
    print()
    print("*_sc  : TM_CCOEFF_NORMED peak. *_pk : peak / best response outside a")
    print("        neighbourhood of the peak (match sharpness, 1.0 = no real peak).")
    print("*_f   : sweep factor that won, on top of the dpi-derived scale. A value")
    print("        at the sweep edge means [anchors].scale_sweep_halfwidth is too tight.")
    print("scale : span_px / the reference page's anchor span. This is the measured")
    print("        magnification of the printed letterhead on this scan.")
    print(f"* marks the page the templates were cropped from ({src_page}); its scores")
    print("  are a self-match and are NOT evidence that anchoring generalises.")
    print()

    if check_names:
        headers = ["page_id"]
        for n in check_names:
            headers += [f"{n[:10]}_sc", "dx_px", "dy_px", "resid_u"]
        print("[2] Independent frame check -- printed landmarks the frame did NOT use")
        print(render_table(headers, check_rows))
        print()
        print("dx/dy : measured landmark position minus the position the anchor frame")
        print("        predicts for it. This is the only accuracy figure Phase 2 can")
        print("        produce, and it is a GEOMETRIC residual in pixels -- not a field")
        print("        accuracy, of which there is none until extraction exists.")
        print("resid_u: the same residual in anchor units. Region boxes are accurate to")
        print("        roughly this, so it is the number that decides how much padding")
        print(f"        the boxes in [regions] need. Tolerance: "
              f"{scfg.get('frame_check', 'max_residual_units')}")
        print("These landmarks are matched but never fed back into the frame, so the")
        print("residual is an independent measurement (Phase 1's crosscheck rule).")
    return 0


# ---------------------------------------------------------------------------
# make-templates
# ---------------------------------------------------------------------------


def cmd_make_templates(args: argparse.Namespace) -> int:
    """Regenerate the template PNGs from the configured source page and boxes.

    The assets in config/templates/ are checked in, but they are *derived*: this
    command is how they were produced and how they are reproduced, so the boxes
    in [templates.<anchor>].reference_box_px are the real definition rather than
    an opaque binary.
    """
    cfg, scfg = _load_both(args)
    files = _files(args)
    want = str(scfg.get("templates", "source_page_id"))
    ref_hw = list(scfg.get("templates", "reference_shape_hw"))
    ref_dpi = list(scfg.get("templates", "reference_dpi"))
    outdir = templates_dir(scfg)

    for f in files:
        for page in iter_pages(f, cfg):
            if page.page_id != want:
                continue
            presult = run_pipeline(page, cfg, dump_dir=None)
            gray = deskewed_source_gray(page, presult)
            h, w = gray.shape[:2]
            if [h, w] != ref_hw:
                print(
                    f"error: source page {want} is {w}x{h}px but "
                    f"[templates].reference_shape_hw says {ref_hw[1]}x{ref_hw[0]}px; "
                    f"the reference boxes would not mean what they say",
                    file=sys.stderr,
                )
                return 2
            if page.effective_dpi is not None:
                got = [round(page.effective_dpi[0], 1), round(page.effective_dpi[1], 1)]
                exp = [round(ref_dpi[0], 1), round(ref_dpi[1], 1)]
                if got != exp:
                    print(
                        f"error: source page dpi {got} != [templates].reference_dpi "
                        f"{exp}; dpi normalisation would be wrong",
                        file=sys.stderr,
                    )
                    return 2
            outdir.mkdir(parents=True, exist_ok=True)
            for name, spec in template_specs(scfg).items():
                x0, x1, y0, y1 = spec["reference_box_px"]
                crop = gray[int(y0):int(y1), int(x0):int(x1)]
                p = template_path(scfg, spec["file"])
                _write_png(p, crop)
                kind = "anchor" if name in ANCHOR_NAMES else "frame-check"
                print(f"wrote {p}  ({crop.shape[1]}x{crop.shape[0]}px, {kind}, from "
                      f"{want} box [{x0},{x1},{y0},{y1}])")
            print(
                "\nThese crops are pre-printed clinic letterhead only (wordmark, "
                "doctor name/qualifications). They contain no patient handwriting, "
                "so they are not PHI -- verify by eye before committing."
            )
            return 0

    print(
        f"error: template source page {want} not found in the given inputs. "
        f"[templates].source_page_id is a content hash, so point this command at "
        f"the directory holding that scan.",
        file=sys.stderr,
    )
    return 2


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.segment",
        description="MedScribe Local region segmentation (Phase 2, spec section "
        "3.4). Template-anchored region geometry. Offline, CPU-only. No field "
        "extraction and no accuracy figures -- geometry only.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="anchor + crop regions + split prescription lines")
    _add_common(r)
    r.add_argument(
        "--dump-crops",
        action="store_true",
        help="write region/line crops, an overlay and a manifest to "
        "data/segments/<page_id>/",
    )
    r.add_argument("--dump-root", default=None, help="override the crop dump root")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("anchors", help="report template-match confidence per page")
    _add_common(a)
    a.set_defaults(func=cmd_anchors)

    t = sub.add_parser(
        "make-templates", help="regenerate config/templates/*.png from the source page"
    )
    _add_common(t)
    t.set_defaults(func=cmd_make_templates)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
