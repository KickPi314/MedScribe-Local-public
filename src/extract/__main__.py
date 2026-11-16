"""CLI for per-region VLM extraction (Phases 3-4).

    python -m src.extract probe              # is Ollama up, is the model pulled
    python -m src.extract vitals             # extract the vitals region, all pages
    python -m src.extract vitals --refresh   # ignore the cache, re-infer
    python -m src.extract vitals --page-id a19dd7cb2c29_p0 --show-raw
    python -m src.extract regions            # ALL regions, both passes
    python -m src.extract regions --region diagnosis --page-id a19dd7cb2c29_p0

This prints WHAT WAS EXTRACTED and WHY EACH EMPTY FIELD IS EMPTY. It prints no
accuracy figure: scoring against ground truth is the eval harness's job, and
mixing the two would let extraction grade itself.

    python -m src.eval vitals                # vitals accuracy vs ground truth
    python -m src.eval extract               # all regions + agreement

``regions`` runs each region through BOTH of its prompts and prints the two
passes side by side with an agreement mark. It never resolves a disagreement --
that is phase 5 (see src/extract/agreement.py).

The only socket this codebase opens is the loopback connection to Ollama; see
src/extract/client.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..preprocess.config import ConfigError, apply_cli_overrides
from ..eval.fixtures import FixtureError
from .client import OllamaClient, OllamaError
from .config import (
    DEFAULT_EXTRACT_CONFIG_PATH,
    EXTRACT_REGIONS,
    VITALS_FIELDS,
    VITALS_REGION,
    load_extract_config,
    ollama_base_url,
    output_dir,
    prompt_spec,
    region_fields,
    region_prompt_names,
    region_spec,
    reject_nested_overrides,
)
from .pipeline import (
    ExtractError,
    ExtractionResult,
    crop_path_for,
    discover_page_ids,
    extract_region,
    line_indices_for,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--extract-config",
        default=str(DEFAULT_EXTRACT_CONFIG_PATH),
        help=f"Phase 3 extraction config TOML (default: {DEFAULT_EXTRACT_CONFIG_PATH})",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one EXTRACT config value for this run only (repeatable, "
        "TOML values): --set 'model.name=\"llava:7b\"'",
    )


def _load(args):
    reject_nested_overrides(args.set)
    return apply_cli_overrides(load_extract_config(args.extract_config), args.set)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """Setup check. Run this first -- it separates "Ollama is not running" from
    "the model cannot read the handwriting", which look identical in a batch log."""
    cfg = _load(args)
    client = OllamaClient(cfg)
    print(f"extract config : {cfg.path}")
    print(f"ollama         : {ollama_base_url(cfg)}  (loopback only, spec section 1.1)")
    print(f"endpoint       : /api/{cfg.get('ollama', 'endpoint')}")
    print(f"model          : {client.model}")
    for note in client.preflight():
        print(f"  ok: {note}")
    print(f"models pulled  : {', '.join(client.available_models())}")
    print()

    from ..eval.__main__ import render_table

    rows = []
    for name in EXTRACT_REGIONS:
        rspec = region_spec(cfg, name)
        pa, pb = region_prompt_names(cfg, name)
        rows.append([
            name,
            str(rspec["segment_region"]),
            str(rspec["crop_file"]),
            f"{prompt_spec(cfg, pa)['version']} / {prompt_spec(cfg, pb)['version']}",
            ", ".join(region_fields(cfg, name)),
        ])
    print(render_table(
        ["region", "segment_region", "crop_file", "prompt versions (A / B)", "fields"],
        rows,
    ))
    print("Both prompt versions are part of the cache key (spec section 3.5), so the")
    print("two passes cache independently and neither can serve the other's reply.")
    print()

    pages = discover_page_ids(cfg)
    print(f"pages from fixtures : {len(pages)}")
    missing: list[str] = []
    for pid, status in pages:
        marks = []
        for name in EXTRACT_REGIONS:
            if bool(region_spec(cfg, name).get("per_line")):
                idx = line_indices_for(pid, cfg, name)
                marks.append(f"{name}={len(idx)} lines" if idx
                             else f"{name}=NO-SPLIT")
                if not idx:
                    missing.append(f"{pid}/{name}")
                continue
            crop = crop_path_for(pid, cfg, name)
            marks.append(f"{name}={'ok' if crop.is_file() else 'MISSING'}")
            if not crop.is_file():
                missing.append(f"{pid}/{name}")
        print(f"  {pid}  gt={status:<8} " + "  ".join(marks))
    if missing:
        print(
            f"\n{len(missing)} region crop(s) absent: {missing}. Extraction reads "
            f"what Phase 2 wrote:\n  python -m src.segment run samples --dump-crops",
            file=sys.stderr,
        )
        return 2
    print("\nAll inputs present. Extract with: python -m src.extract regions")
    return 0


# ---------------------------------------------------------------------------
# vitals
# ---------------------------------------------------------------------------


def _fmt_value(res: ExtractionResult, name: str) -> str:
    """A field's cell. An empty field shows WHY it is empty, never a blank.

    Spec section 10: "do not silently drop a field that failed extraction". The
    distinction on show is between ``.`` (the model says nothing is written there,
    a real answer) and an uppercase status (extraction failed, value withheld --
    which is what phase 5 will render as `flagged`).
    """
    read = res.reads.get(name)
    if read is None:
        return "?"
    if read.ok:
        return f"{read.value}"
    if read.status == "absent":
        return "."
    return read.status.upper().replace("_", "-")


def cmd_vitals(args: argparse.Namespace) -> int:
    from ..eval.__main__ import render_table

    cfg = _load(args)
    client = OllamaClient(cfg)
    rspec = region_spec(cfg, VITALS_REGION)
    pspec = prompt_spec(cfg, str(rspec["prompt"]))

    pages = discover_page_ids(cfg)
    if args.page_id:
        wanted = set(args.page_id)
        unknown = wanted - {p for p, _ in pages}
        if unknown:
            print(
                f"error: no fixture for page_id(s) {sorted(unknown)}. Known: "
                f"{sorted(p for p, _ in pages)}",
                file=sys.stderr,
            )
            return 2
        pages = [(p, s) for p, s in pages if p in wanted]
    if not pages:
        print("error: no fixtures found, so there are no pages to extract",
              file=sys.stderr)
        return 2

    print("MedScribe Local -- VLM field extraction (PHASE 3)")
    print(f"extract config : {cfg.path}")
    print(f"ollama         : {ollama_base_url(cfg)}")
    print(f"model          : {client.model}")
    print(f"prompt         : {rspec['prompt']} (version {pspec['version']})")
    print(f"sampling       : {dict(cfg.section('model')['options'])}")
    print(f"region         : {rspec['segment_region']}")
    print(f"pages          : {len(pages)}")
    print(f"cache          : {'on' if cfg.get('cache', 'enabled') else 'OFF'}"
          f"{' (--refresh: ignoring existing entries)' if args.refresh else ''}")
    print()

    # Preflight before the loop: an unreachable server should fail in one second
    # with a clear message, not after the first page's timeout.
    for note in client.preflight():
        print(f"preflight ok   : {note}")
    print()

    results: list[ExtractionResult] = []
    for pid, status in pages:
        res = extract_region(pid, cfg, client, refresh=args.refresh)
        results.append(res)
        vals = " ".join(f"{n}={_fmt_value(res, n)}" for n in VITALS_FIELDS)
        print(
            f"{pid} gt={status:<8} {'CACHED' if res.from_cache else 'inferred'} "
            f"[{res.duration_s:6.2f}s] json={'ok' if res.json_ok else 'FAILED'} "
            f"read={res.n_ok}/{len(VITALS_FIELDS)} {vals}"
        )
        for w in res.warnings:
            print(f"    ! {w}")

    print()
    print("[1] Extracted vitals -- values as parsed, per page")
    rows = [
        [r.page_id, *[_fmt_value(r, n) for n in VITALS_FIELDS],
         "ok" if r.json_ok else "NO",
         f"{r.n_ok}/{r.n_absent}/{r.n_failed}"]
        for r in results
    ]
    print(render_table(["page_id", *VITALS_FIELDS, "json", "ok/abs/fail"], rows))
    print()
    print(".            : the model reported no value written for that label. This")
    print("               is a real answer -- 3 of the 7 real slips record no MAC or")
    print("               Abd -- and it is scored separately from a failure.")
    print("UPPER-CASE   : extraction failed and the value is WITHHELD (spec section")
    print("               1.3). MISSING-KEY, UNPARSEABLE, OUT-OF-RANGE, NO-JSON.")
    print("               Nothing is substituted for it. Phase 5 turns these into")
    print("               the `flagged` field state with the reason in the UI.")
    print("ok/abs/fail  : counts of read / genuinely-absent / withheld fields.")
    print()

    withheld = [(r, n, rd) for r in results for n, rd in r.reads.items() if rd.failed]
    print(f"[2] Withheld fields and the reason ({len(withheld)})")
    if not withheld:
        print("  none")
    for r, n, rd in withheld:
        print(f"  {r.page_id} {n}: [{rd.status}] {rd.reason}")
    print()

    print("[3] Inference cost (CPU-only, spec section 1.2)")
    rows = [
        [
            r.page_id,
            "yes" if r.from_cache else "no",
            f"{r.duration_s:.2f}",
            str(r.model_meta.get("prompt_eval_count", "-")),
            str(r.model_meta.get("eval_count", "-")),
            str(r.model_meta.get("done_reason", "-")),
            r.raw_text_field or "-",
        ]
        for r in results
    ]
    print(render_table(
        ["page_id", "cached", "wall_s", "prompt_tok", "gen_tok", "done", "reply_field"],
        rows,
    ))
    print("reply_field : which JSON field of Ollama's reply the text came from. On")
    print("              the verified local stack this is 'thinking', not 'response'")
    print("              -- see [model].response_fields in the extract config.")
    print()

    if args.show_raw:
        print("[4] Raw model replies (evidence for debugging, never a fallback value)")
        for r in results:
            print(f"--- {r.page_id} ({r.raw_text_field})")
            print(r.raw_text.strip() or "(empty)")
        print()

    print(f"Per-page results : {output_dir(cfg)}/<page_id>.{rspec['segment_region']}.json")
    print(
        "No accuracy figure is printed here: extraction does not grade itself. "
        "Score against ground truth with:\n  python -m src.eval vitals"
    )
    return 0


# ---------------------------------------------------------------------------
# regions (Phase 4: all regions, both passes)
# ---------------------------------------------------------------------------


def _cell(value, status: str, width: int = 28) -> str:
    """One value's cell. An empty cell always says WHY it is empty."""
    if value is None:
        return "." if status == "absent" else status.upper().replace("_", "-")
    if isinstance(value, list):
        if not value:
            return "[]"
        parts = [
            v.get("text", "?") if isinstance(v, dict) else str(v) for v in value
        ]
        s = "[" + " | ".join(parts) + "]"
    else:
        s = str(value)
    return s if len(s) <= width else s[: width - 1] + "…"


def cmd_regions(args: argparse.Namespace) -> int:
    """Run every selected region through BOTH prompts and show them side by side.

    Prints no accuracy: `python -m src.eval extract` scores this against ground
    truth. Prints no resolved value either -- where the passes disagree, both
    readings are shown, because choosing between them is phase 5's job (spec
    section 4 makes disagreement a signal, not a decision).
    """
    from ..eval.__main__ import render_table
    from .agreement import run_two_pass_region
    from .pipeline import line_split_trusted

    cfg = _load(args)
    client = OllamaClient(cfg)

    regions = args.region or list(EXTRACT_REGIONS)
    unknown = [r for r in regions if r not in EXTRACT_REGIONS]
    if unknown:
        print(f"error: unknown region(s) {unknown}. Known: {list(EXTRACT_REGIONS)}",
              file=sys.stderr)
        return 2

    pages = discover_page_ids(cfg)
    if args.page_id:
        wanted = set(args.page_id)
        unknown_p = wanted - {p for p, _ in pages}
        if unknown_p:
            print(f"error: no fixture for page_id(s) {sorted(unknown_p)}",
                  file=sys.stderr)
            return 2
        pages = [(p, s) for p, s in pages if p in wanted]
    if not pages:
        print("error: no fixtures found, so there are no pages to extract",
              file=sys.stderr)
        return 2

    print("MedScribe Local -- per-region VLM extraction, two passes (PHASE 4)")
    print(f"extract config : {cfg.path}")
    print(f"ollama         : {ollama_base_url(cfg)}")
    print(f"model          : {client.model}")
    print(f"regions        : {', '.join(regions)}")
    print(f"pages          : {len(pages)}")
    print(f"cache          : {'on' if cfg.get('cache', 'enabled') else 'OFF'}"
          f"{' (--refresh: ignoring existing entries)' if args.refresh else ''}")
    print()
    for note in client.preflight():
        print(f"preflight ok   : {note}")
    print()

    excluded: list[str] = []
    all_results = []
    for region in regions:
        fields = region_fields(cfg, region)
        pa, pb = region_prompt_names(cfg, region)
        print(f"=== {region}  ({region_spec(cfg, region)['segment_region']}), "
              f"prompts {pa} / {pb}")
        rows = []
        for pid, gt in pages:
            if bool(region_spec(cfg, region).get("per_line")):
                ok, why = line_split_trusted(pid, cfg)
                if not ok:
                    excluded.append(f"{pid} {region}: {why}")
                    rows.append([pid, "-", "EXCLUDED: Phase 2 did not trust the "
                                 "line split", *["" for _ in fields[1:]], ""])
                    continue
            results = run_two_pass_region(pid, cfg, client, region,
                                          refresh=args.refresh)
            all_results.extend(results)
            for r in results:
                label = pid if r.line_index is None else f"{pid} L{r.line_index}"
                for pas, tag in ((r.pass_a, "A"), (r.pass_b, "B")):
                    rows.append([
                        label if tag == "A" else "",
                        tag,
                        *[_cell(pas.reads[f].value, pas.reads[f].status)
                          for f in fields],
                        "" if tag == "A" else
                        f"{r.n_agree}/{r.n_fields} agree",
                    ])
        print(render_table(["page_id", "pass", *fields, "two-pass"], rows))
        print()

    print("A / B       : the two differently-worded prompts (spec section 4.1).")
    print("              Both readings are shown. Nothing here picks a winner --")
    print("              resolving a disagreement is phase 5.")
    print(".           : the model reported nothing written there (a real answer).")
    print("UPPER-CASE  : extraction failed and the value is WITHHELD (spec 1.3).")
    print("[]          : an explicitly empty list -- 'no complaints recorded'.")
    print()
    if excluded:
        print(f"Excluded crops ({len(excluded)}):")
        for e in excluded:
            print(f"  {e}")
        print()

    if args.show_raw:
        print("Raw model replies (evidence for debugging, never a fallback value)")
        for r in all_results:
            for pas, tag in ((r.pass_a, "A"), (r.pass_b, "B")):
                print(f"--- {r.page_id} {r.region_key} "
                      f"{'' if r.line_index is None else f'L{r.line_index} '}"
                      f"pass {tag} ({pas.raw_text_field})")
                print(pas.raw_text.strip() or "(empty)")
        print()

    print(f"Per-pass results : {output_dir(cfg)}/<page_id>.<region>[.lineN].<prompt>.json")
    print(f"Agreement files  : {cfg.get('output', 'agreement_dir')}/")
    print("No accuracy figure is printed here: extraction does not grade itself. "
          "Score with:\n  python -m src.eval extract")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.extract",
        description="MedScribe Local single-region VLM extraction (Phase 3, spec "
        "section 9 item 3). Reads the Phase 2 vitals-column crops through a local "
        "Ollama VLM. Offline (loopback only), CPU-only. Prints what was extracted "
        "and why each empty field is empty; accuracy is python -m src.eval vitals.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="check Ollama, the model and the input crops")
    _add_common(pr)
    pr.set_defaults(func=cmd_probe)

    v = sub.add_parser("vitals", help="extract the vitals column for every fixture page")
    _add_common(v)
    v.add_argument(
        "--page-id",
        action="append",
        default=[],
        help="extract only this page_id (repeatable); default: every fixture page",
    )
    v.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached replies and re-infer (spec section 3.5 cache key: "
        "image hash, region, model, prompt version)",
    )
    v.add_argument(
        "--show-raw",
        action="store_true",
        help="print each raw model reply",
    )
    v.set_defaults(func=cmd_vitals)

    rg = sub.add_parser(
        "regions",
        help="Phase 4: extract every region (or --region NAME) through BOTH of "
        "its prompts and show the two passes side by side",
    )
    _add_common(rg)
    rg.add_argument(
        "--region",
        action="append",
        default=[],
        choices=list(EXTRACT_REGIONS),
        help="extract only this region (repeatable); default: all of them",
    )
    rg.add_argument(
        "--page-id",
        action="append",
        default=[],
        help="extract only this page_id (repeatable); default: every fixture page",
    )
    rg.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached replies and re-infer both passes",
    )
    rg.add_argument("--show-raw", action="store_true", help="print each raw reply")
    rg.set_defaults(func=cmd_regions)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OllamaError as exc:
        print(f"ollama error: {exc}", file=sys.stderr)
        return 3
    except FixtureError as exc:
        print(f"fixture error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
