"""CLI for the preprocessing pipeline.

    python -m src.preprocess run samples/ --dump-intermediates
    python -m src.preprocess run samples/one.pdf --dump-intermediates --out data/out
    python -m src.preprocess probe samples/

Offline only: nothing here opens a socket.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .config import DEFAULT_CONFIG_PATH, ConfigError, apply_cli_overrides, load_config
from .ingest import discover_inputs, iter_pages
from .pipeline import DEFAULT_DUMP_ROOT, dump_dir_for, run_pipeline, _write_png


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "inputs",
        nargs="+",
        help="PDF/image file(s) or directory/directories to process",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"pipeline config TOML (default: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value for this run only (repeatable). "
        "Values are parsed as TOML, so strings need quotes: "
        "--set 'channel_separation.method=\"red_channel\"'",
    )


def cmd_run(args: argparse.Namespace) -> int:
    cfg = apply_cli_overrides(load_config(args.config), args.set)

    files: list[Path] = []
    for i in args.inputs:
        files.extend(discover_inputs(i))
    if not files:
        print("no supported input files found", file=sys.stderr)
        return 2

    dump_root = Path(args.dump_root) if args.dump_root else DEFAULT_DUMP_ROOT
    out_root = Path(args.out) if args.out else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    order = cfg.step_order()
    active = [s for s in order if cfg.step_enabled(s)]
    print(f"config      : {cfg.path}")
    print(f"step order  : {' -> '.join(order)}")
    print(f"active steps: {' -> '.join(active) if active else '(none)'}")
    print(f"channel sep : {cfg.get('channel_separation', 'method')}")
    print(f"inputs      : {len(files)} file(s)")
    if args.dump_intermediates:
        print(f"dumping to  : {dump_root}")
    print()

    n_pages = 0
    for f in files:
        for page in iter_pages(f, cfg):
            dump_dir = (
                dump_dir_for(page.page_id, dump_root) if args.dump_intermediates else None
            )
            result = run_pipeline(page, cfg, dump_dir=dump_dir)
            n_pages += 1
            skew = result.stage("deskew")
            skew_txt = ""
            if skew is not None and skew.enabled:
                a = skew.info.get("estimated_angle_deg")
                skew_txt = (
                    f" skew={a:+.2f}deg{'(applied)' if skew.info.get('applied') else '(not applied)'}"
                    if a is not None
                    else " skew=unmeasured"
                )
            binf = result.stage("binarize")
            fg_txt = ""
            if binf is not None and binf.enabled:
                fg_txt = f" fg={binf.info.get('foreground_ratio'):.4f}"
            fh, fw = result.final.shape[:2]
            print(
                f"{page.describe()} -> {fw}x{fh}px"
                f"{skew_txt}{fg_txt} [{result.total_duration_s:.2f}s]"
            )
            if out_root:
                _write_png(out_root / f"{page.page_id}.png", result.final)

    print(f"\n{n_pages} page(s) processed.")
    if args.dump_intermediates:
        print(f"Intermediates: {dump_root}/<page_id>/<index>_<step>.png")
        print("Each dump dir also holds contact_sheet.png and manifest.json.")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Report what ingestion actually produces -- resolution, source path kind.

    Use this before trusting any DPI assumption.
    """
    cfg = apply_cli_overrides(load_config(args.config), args.set)
    files: list[Path] = []
    for i in args.inputs:
        files.extend(discover_inputs(i))
    print(f"{'page_id':>18}  {'pixels':>12}  {'eff. dpi':>11}  source")
    print("-" * 62)
    for f in files:
        for page in iter_pages(f, cfg):
            h, w = page.shape
            dpi = (
                f"{page.effective_dpi[0]:.0f}x{page.effective_dpi[1]:.0f}"
                if page.effective_dpi
                else "unknown"
            )
            print(f"{page.page_id:>18}  {w}x{h:<8}  {dpi:>11}  {page.source_kind}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.preprocess",
        description="MedScribe Local preprocessing pipeline (Phase 1). Offline, CPU-only.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline over inputs")
    _add_common(r)
    r.add_argument(
        "--dump-intermediates",
        action="store_true",
        help="write every stage's output image to data/dumps/<page_id>/",
    )
    r.add_argument("--dump-root", default=None, help="override the dump root directory")
    r.add_argument("--out", default=None, help="also write the final image per page here")
    r.set_defaults(func=cmd_run)

    pr = sub.add_parser("probe", help="report ingested resolution per page")
    _add_common(pr)
    pr.set_defaults(func=cmd_probe)

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
