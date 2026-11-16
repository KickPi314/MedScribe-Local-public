"""The non-UI half of the TUI: paths, discovery, status, replay, stage runs, eval.

Everything the screens show is produced here by calling the existing pipeline
modules, so it can be tested without a terminal. The rules this module keeps:

* **Replay by default.** :func:`replay_page` reads the spec section 3.5 cache
  through ``src.extract.agreement.run_two_pass`` -- the same call
  ``python -m src.extract regions`` makes -- with ``allow_inference=False``
  and a :class:`ReplayOnlyClient`, which has no transport and raises if it is
  ever asked to send something. ``refresh`` is passed as ``False`` at every
  call site and is not a parameter of anything in this package.
* **Inference only when asked, and only for what is missing.**
  ``replay_page(..., infer=True, client=...)`` is the one path that may send
  requests. The UI reaches it only after :func:`ollama_version` has been
  called and its answer shown, and after the operator confirms. Cached passes
  are served from the cache as usual, so only the passes
  :func:`plan_inference` lists as "not cached" are sent.
* **The eval is the eval CLI.** :func:`run_eval` calls
  ``src.eval.__main__.main`` in-process and reads the metrics JSON it writes.
  The tables are not recomputed here. It is pointed at a generated copy of the
  eval config (:func:`write_cache_only_eval_config`) whose only change is
  ``run_extraction_if_missing = false`` in ``[vitals]`` and ``[extract]``:
  that makes the CLI skip its Ollama preflight and turns a cache miss into a
  loud ``ExtractError`` instead of a re-inference.
* **Stage runs are the CLIs' calls, for one page.** :func:`run_preprocess_stage`
  is ``python -m src.preprocess run <file> --dump-intermediates`` and
  :func:`run_segment_stage` is ``python -m src.segment run <file>
  --dump-crops``, both restricted to one page_id and writing to the same
  places.

PHI: :attr:`PageRef.source` is the scan's path, and scan filenames contain
patient names. It is held for display and for re-opening the scan; this module
never writes it anywhere. Every file this module writes is keyed on page_id
and lives under ``data/out/`` (or wherever a test points the workspace).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import re
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from ..eval.fixtures import FixtureError, load_fixtures
from ..extract.agreement import TwoPassField, TwoPassResult, compare, run_two_pass
from ..extract.client import OllamaClient, OllamaError
from ..extract.config import (
    DEFAULT_EXTRACT_CONFIG_PATH,
    EXTRACT_REGIONS,
    cache_dir,
    fixtures_dir,
    load_extract_config,
    ollama_base_url,
    prompt_spec,
    region_fields,
    region_prompt_names,
    region_spec,
    reject_nested_overrides,
    segments_dir,
)
from ..extract.parse import FieldRead
from ..extract.pipeline import (
    ExtractError,
    ExtractionResult,
    crop_path_for,
    extract_region,
    line_indices_for,
    line_split_trusted,
)
from ..preprocess import ingest as _ingest
from ..preprocess.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    Config,
    ConfigError,
    apply_cli_overrides,
    load_config,
)
from ..preprocess.ingest import PDF_SUFFIXES, discover_inputs, iter_pages, page_id_for
from ..preprocess.pipeline import DEFAULT_DUMP_ROOT, dump_dir_for, run_pipeline
from ..segment.anchors import load_templates
from ..segment.config import DEFAULT_SEGMENT_CONFIG_PATH, load_segment_config
from ..segment.pipeline import segment_dir_for, segment_page

DEFAULT_EVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "eval.toml"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "samples"
#: Where the TUI writes its own files: the derived eval config and the eval
#: metrics JSON. Under data/out/ because the metrics embed ground truth (PHI),
#: and kept apart from the CLI's own data/out/phase*_metrics.json so viewing
#: the eval in the TUI never overwrites the operator's last CLI run.
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "out" / "tui"

EVAL_KINDS = ("vitals", "extract")


def rel(path: Path | str) -> str:
    """A path as the operator reads it: project-relative where possible."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except (ValueError, OSError):
        return str(p).replace("\\", "/")


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


@dataclass
class Workspace:
    """Every path and config the TUI reads, resolved once at start-up.

    The defaults are the CLIs' defaults. Tests build one pointed entirely at a
    temporary directory; ``python -m src.tui`` builds one from its arguments.
    ``extract_overrides`` uses the CLIs' ``SECTION.KEY=VALUE`` syntax and is
    applied exactly as ``--set-extract`` is, and it is forwarded to the eval CLI
    as ``--set-extract``, so the TUI's views and the eval read one config.
    """

    inputs: list[Path] = field(default_factory=lambda: [DEFAULT_INPUT_DIR])
    preprocess_config_path: Path = DEFAULT_CONFIG_PATH
    segment_config_path: Path = DEFAULT_SEGMENT_CONFIG_PATH
    extract_config_path: Path = DEFAULT_EXTRACT_CONFIG_PATH
    extract_overrides: list[str] = field(default_factory=list)
    eval_config_path: Path = DEFAULT_EVAL_CONFIG_PATH
    #: ``{section: {key: value}}`` merged over the eval config before the
    #: cache-only copy is written. Tests use it to point ``[fixtures].dir`` at a
    #: temporary directory.
    eval_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    dumps_root: Path = DEFAULT_DUMP_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    #: Builds the client for the explicit inference action. Never called in
    #: replay. Tests substitute a fake that records calls and opens nothing.
    client_factory: Callable[[Config], Any] = OllamaClient

    preprocess_cfg: Config = field(init=False)
    segment_cfg: Config = field(init=False)
    extract_cfg: Config = field(init=False)
    eval_data: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.inputs = [Path(p) for p in self.inputs]
        self.preprocess_config_path = Path(self.preprocess_config_path)
        self.segment_config_path = Path(self.segment_config_path)
        self.extract_config_path = Path(self.extract_config_path)
        self.eval_config_path = Path(self.eval_config_path)
        self.dumps_root = Path(self.dumps_root)
        self.out_dir = Path(self.out_dir)
        self.extract_overrides = [str(o) for o in self.extract_overrides]

        reject_nested_overrides(self.extract_overrides)
        self.preprocess_cfg = load_config(self.preprocess_config_path)
        self.segment_cfg = load_segment_config(self.segment_config_path)
        self.extract_cfg = apply_cli_overrides(
            load_extract_config(self.extract_config_path), self.extract_overrides
        )
        self.eval_data = self._load_eval_data()

    def _load_eval_data(self) -> dict[str, Any]:
        p = self.eval_config_path
        if not p.is_file():
            raise ConfigError(f"eval config not found: {p}")
        with p.open("rb") as fh:
            data = tomllib.load(fh)
        for section, kv in self.eval_overrides.items():
            data.setdefault(section, {})
            data[section].update(dict(kv))
        if not isinstance(data.get("fixtures"), dict) or "dir" not in data["fixtures"]:
            raise ConfigError(f"[fixtures].dir missing from {p}")
        return data

    # -- where things are ------------------------------------------------
    @property
    def segments_root(self) -> Path:
        """Where Phase 2 crops are read from AND written to.

        One source of truth: ``[inputs].segments_dir`` of the extract config, the
        directory extraction reads. The segment stage writes there too, so a
        re-segmented page is what the next replay reads.
        """
        return segments_dir(self.extract_cfg)

    @property
    def fixtures_root(self) -> Path:
        return fixtures_dir(self.extract_cfg)

    @property
    def eval_fixtures_root(self) -> Path:
        return PROJECT_ROOT / str(self.eval_data["fixtures"]["dir"])

    @property
    def cache_root(self) -> Path:
        return cache_dir(self.extract_cfg)

    @property
    def model(self) -> str:
        return str(self.extract_cfg.get("model", "name"))

    @property
    def ollama_url(self) -> str:
        return ollama_base_url(self.extract_cfg)

    def fixtures_disagree(self) -> bool:
        """The eval CLI warns about this too; the TUI shows it up front."""
        try:
            return self.fixtures_root.resolve() != self.eval_fixtures_root.resolve()
        except OSError:
            return True


# ---------------------------------------------------------------------------
# page discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageRef:
    """One page of one input file, identified the way every stage identifies it."""

    page_id: str
    #: The scan's path. Its filename contains a patient name: shown on screen,
    #: used to re-open the scan, never written anywhere.
    source: Path
    page_index: int
    n_pages: int

    @property
    def source_label(self) -> str:
        return self.source.name + (f" [p{self.page_index}]" if self.n_pages > 1 else "")


def _page_count(path: Path) -> int:
    if path.suffix.lower() in PDF_SUFFIXES:
        if _ingest.fitz is None:  # pragma: no cover - pymupdf is a requirement
            raise RuntimeError("PyMuPDF (pymupdf) is required to read PDF input")
        doc = _ingest.fitz.open(path)
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    return 1


def discover_pages(inputs: list[Path]) -> tuple[list[PageRef], list[str]]:
    """``(pages, problems)`` for the given files/directories.

    page_ids come from :func:`src.preprocess.ingest.page_id_for`, the function
    every stage keys its output on, so a page here is the same page in
    data/dumps/, data/segments/, the cache and the fixtures. Pages are counted
    with PyMuPDF without decoding any pixels.
    """
    pages: list[PageRef] = []
    problems: list[str] = []
    seen: dict[str, Path] = {}
    for root in inputs:
        try:
            files = discover_inputs(root)
        except FileNotFoundError:
            problems.append(f"input not found: {rel(root)}")
            continue
        for f in files:
            try:
                n = _page_count(f)
            except Exception as exc:  # a broken PDF must not hide the others
                problems.append(f"cannot open {f.name}: {exc}")
                continue
            for i in range(n):
                pid = page_id_for(f, i)
                if pid in seen:
                    problems.append(
                        f"{f.name} has the same content as {seen[pid].name} "
                        f"(page_id {pid}); listed once"
                    )
                    continue
                seen[pid] = f
                pages.append(PageRef(page_id=pid, source=f, page_index=i, n_pages=n))
    return pages, problems


# ---------------------------------------------------------------------------
# per-stage status, read from what each stage wrote
# ---------------------------------------------------------------------------


@dataclass
class DumpStatus:
    """Phase 1 output for one page (data/dumps/<page_id>/manifest.json)."""

    present: bool
    dump_dir: Path
    n_steps: int = 0
    skew: str = ""
    fg_ratio: float | None = None
    total_s: float | None = None
    error: str = ""


def read_dump_status(ws: Workspace, page_id: str) -> DumpStatus:
    d = dump_dir_for(page_id, ws.dumps_root)
    m = d / "manifest.json"
    if not m.is_file():
        return DumpStatus(False, d)
    try:
        data = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return DumpStatus(False, d, error=f"unreadable manifest: {exc}")
    out = DumpStatus(True, d, total_s=data.get("total_duration_s"))
    for s in data.get("stages") or []:
        if s.get("name") == "source" or not s.get("enabled"):
            continue
        out.n_steps += 1
        info = s.get("info") or {}
        if s.get("name") == "deskew":
            a = info.get("estimated_angle_deg")
            if isinstance(a, (int, float)):
                out.skew = f"{a:+.2f}deg {'applied' if info.get('applied') else 'not applied'}"
        if s.get("name") == "binarize" and isinstance(info.get("foreground_ratio"), (int, float)):
            out.fg_ratio = float(info["foreground_ratio"])
    return out


@dataclass
class SegmentStatus:
    """Phase 2 output for one page (data/segments/<page_id>/manifest.json)."""

    present: bool
    seg_dir: Path
    anchored: bool = False
    frame_source: str = ""
    frame_reason: str = ""
    regions_ok: int = 0
    regions_total: int = 0
    split_ok: bool = False
    n_lines: int = 0
    split_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def read_segment_status(ws: Workspace, page_id: str) -> SegmentStatus:
    d = segment_dir_for(page_id, ws.segments_root)
    m = d / "manifest.json"
    if not m.is_file():
        return SegmentStatus(False, d)
    try:
        data = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SegmentStatus(False, d, error=f"unreadable manifest: {exc}")
    frame = data.get("frame") or {}
    regions = [r for r in data.get("regions") or [] if r.get("enabled", True)]
    lines = data.get("prescription_lines") or {}
    info = lines.get("info") or {}
    return SegmentStatus(
        present=True,
        seg_dir=d,
        anchored=bool(frame.get("ok")),
        frame_source=str(frame.get("source") or ""),
        frame_reason=str(frame.get("reason") or ""),
        regions_ok=sum(1 for r in regions if r.get("geometry_ok")),
        regions_total=len(regions),
        split_ok=bool(info.get("split_ok")),
        n_lines=len(lines.get("dumps") or []),
        split_reason=str(info.get("reason") or ""),
        warnings=[str(w) for w in data.get("warnings") or []],
    )


def read_fixture_statuses(ws: Workspace) -> tuple[dict[str, str], str]:
    """``({page_id: ground_truth_status}, problem)``. A missing fixture dir is
    a problem to show, not a crash: the pipeline runs without ground truth."""
    try:
        fixtures = load_fixtures(ws.fixtures_root)
    except FixtureError as exc:
        return {}, str(exc)
    return {fx.page_id: fx.ground_truth_status for fx in fixtures}, ""


# ---------------------------------------------------------------------------
# cache replay
# ---------------------------------------------------------------------------


class ReplayOnlyClient:
    """The client handed to ``extract_region`` whenever inference is not allowed.

    ``extract_region`` only touches its client on a cache miss with
    ``allow_inference=True``, so in replay this object is never used at all. It
    exists so that, if that ever changed, the result would be a loud error
    rather than a request: it has no transport to send one with.
    """

    def __init__(self, cfg: Config) -> None:
        self.model = str(cfg.get("model", "name"))
        self.base_url = ollama_base_url(cfg)

    def _refuse(self, *_args: Any, **_kwargs: Any) -> Any:
        raise OllamaError(
            "replay mode: the TUI only reads the extraction cache here, so no "
            "request was sent to Ollama"
        )

    generate = _refuse
    version = _refuse
    available_models = _refuse
    preflight = _refuse


#: Short labels for the setup errors extract_region raises. Matched on its own
#: message text; anything unrecognised is shown as "error" with the full text.
_ERROR_LABELS = (
    ("no cached extraction", "not cached"),
    ("replay mode", "not cached"),
    ("no phase 2 manifest", "no manifest"),
    ("crop for page", "no crop"),
    ("was not anchored", "unanchored"),
    ("geometry as unsound", "bad geometry"),
    ("line split as", "untrusted split"),
    ("is disabled in config", "region disabled"),
)


def classify_error(message: str) -> str:
    low = (message or "").lower()
    for needle, label in _ERROR_LABELS:
        if needle in low:
            return label
    return "error" if message else ""


@dataclass
class PassView:
    """One pass over one crop: the result, or why there is none."""

    prompt_name: str
    prompt_version: str
    result: ExtractionResult | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None

    def read(self, name: str) -> FieldRead | None:
        return None if self.result is None else self.result.reads.get(name)


@dataclass
class CropView:
    """Both passes over one crop (a region, or one prescription line)."""

    region: str
    segment_region: str
    line_index: int | None
    fields: tuple[str, ...]
    crop_path: Path | None
    a: PassView
    b: PassView
    #: ``agreement.compare`` output -- present only when BOTH passes have a
    #: result. Never resolved into a single value (src/extract/agreement.py).
    comparison: dict[str, TwoPassField] | None = None
    two_pass: TwoPassResult | None = None
    ollama_error: str = ""

    def passes(self) -> tuple[tuple[str, PassView], tuple[str, PassView]]:
        return (("A", self.a), ("B", self.b))

    @property
    def state(self) -> str:
        """"" when both passes are present, else what is missing and why."""
        if self.a.ok and self.b.ok:
            return ""
        if self.a.ok:
            return f"B {classify_error(self.b.error)}"
        if self.b.ok:
            return f"A {classify_error(self.a.error)}"
        return classify_error(self.a.error or self.b.error)

    @property
    def n_agree_read(self) -> int:
        """Both passes read a value and it is the same value."""
        if not self.comparison:
            return 0
        return sum(1 for f in self.comparison.values() if f.agree and f.both_read)

    @property
    def n_diff(self) -> int:
        if not self.comparison:
            return 0
        return sum(1 for f in self.comparison.values() if not f.agree)

    @property
    def n_both_null(self) -> int:
        if not self.comparison:
            return 0
        return sum(1 for f in self.comparison.values() if f.both_null)

    def n_withheld(self, tag: str) -> int:
        pv = self.a if tag == "A" else self.b
        if pv.result is None:
            return 0
        return sum(1 for r in pv.result.reads.values() if r.failed)


@dataclass
class RegionView:
    region: str
    segment_region: str
    fields: tuple[str, ...]
    prompts: tuple[str, str]
    versions: tuple[str, str]
    per_line: bool
    crops: list[CropView] = field(default_factory=list)
    #: Phase 2 refused to split the prescription block; its reason, verbatim.
    excluded: str = ""
    #: A setup problem for the whole region (e.g. no Phase 2 manifest).
    error: str = ""


@dataclass
class PageReplay:
    page_id: str
    regions: list[RegionView] = field(default_factory=list)
    inferred: bool = False
    duration_s: float = 0.0
    #: Why an inference run stopped sending requests early, if it did.
    aborted: str = ""

    @property
    def crops(self) -> list[CropView]:
        return [c for r in self.regions for c in r.crops]

    @property
    def n_passes(self) -> int:
        return 2 * len(self.crops)

    @property
    def n_passes_present(self) -> int:
        return sum(int(c.a.ok) + int(c.b.ok) for c in self.crops)

    @property
    def n_agree_read(self) -> int:
        return sum(c.n_agree_read for c in self.crops)

    @property
    def n_diff(self) -> int:
        return sum(c.n_diff for c in self.crops)

    @property
    def n_both_null(self) -> int:
        return sum(c.n_both_null for c in self.crops)

    @property
    def n_withheld(self) -> int:
        return sum(c.n_withheld("A") + c.n_withheld("B") for c in self.crops)

    @property
    def n_problem_regions(self) -> int:
        return sum(1 for r in self.regions if r.error)


def _region_view(cfg: Config, region: str) -> RegionView:
    rspec = region_spec(cfg, region)
    pa, pb = region_prompt_names(cfg, region)
    return RegionView(
        region=region,
        segment_region=str(rspec["segment_region"]),
        fields=region_fields(cfg, region),
        prompts=(pa, pb),
        versions=(str(prompt_spec(cfg, pa)["version"]), str(prompt_spec(cfg, pb)["version"])),
        per_line=bool(rspec.get("per_line")),
    )


def _replay_crop(page_id: str, cfg: Config, rv: RegionView, line_index: int | None, *,
                 client: Any, allow_inference: bool, write_output: bool) -> CropView:
    try:
        crop_path: Path | None = crop_path_for(page_id, cfg, rv.region, line_index)
    except ExtractError:
        crop_path = None
    crop = CropView(
        region=rv.region,
        segment_region=rv.segment_region,
        line_index=line_index,
        fields=rv.fields,
        crop_path=crop_path,
        a=PassView(rv.prompts[0], rv.versions[0]),
        b=PassView(rv.prompts[1], rv.versions[1]),
    )
    try:
        tp = run_two_pass(page_id, cfg, client, rv.region, line_index=line_index,
                          refresh=False, write_output=write_output,
                          allow_inference=allow_inference)
    except OllamaError as exc:
        crop.ollama_error = str(exc)
    except ExtractError:
        pass  # diagnosed per pass below
    except Exception as exc:  # one bad crop must not take the page down
        crop.a.error = crop.b.error = f"{type(exc).__name__}: {exc}"
        return crop
    else:
        crop.a.result, crop.b.result = tp.pass_a, tp.pass_b
        crop.comparison = tp.fields
        crop.two_pass = tp
        return crop

    # run_two_pass stops at the first pass that fails, which hides whether the
    # other one is cached. Ask for each pass on its own -- cache only, writing
    # nothing -- so the screen can say WHICH pass is missing and why.
    replay_client = ReplayOnlyClient(cfg)
    for pv in (crop.a, crop.b):
        try:
            pv.result = extract_region(page_id, cfg, replay_client, region=rv.region,
                                       refresh=False, write_output=False,
                                       allow_inference=False, prompt_name=pv.prompt_name,
                                       line_index=line_index)
        except ExtractError as exc:
            pv.error = str(exc)
        except Exception as exc:
            pv.error = f"{type(exc).__name__}: {exc}"
    if crop.a.result is not None and crop.b.result is not None:
        crop.comparison = compare(crop.a.result, crop.b.result, rv.fields)
    return crop


def replay_page(
    ws: Workspace,
    page_id: str,
    *,
    infer: bool = False,
    client: Any = None,
    write_output: bool = False,
    progress: Callable[[int, int, CropView], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PageReplay:
    """Both passes of every region of one page.

    ``infer=False`` (the default, and every path the UI takes without an
    explicit confirmation) is a pure cache read: ``allow_inference=False`` and a
    :class:`ReplayOnlyClient`. ``write_output=False`` additionally makes it
    read-only on disk; the run screen passes ``True`` so a replay refreshes
    data/out/extract/ and data/out/extract_agreement/ exactly as
    ``python -m src.extract regions --page-id <id>`` does.

    ``infer=True`` requires the caller's ``client`` and is only reached after
    the Ollama version has been shown and the operator has confirmed. It still
    never refreshes: a cached pass is served from the cache and only a miss is
    sent. The first :class:`OllamaError` stops all further requests (the rest of
    the page is still replayed from the cache), as does ``cancel()``.
    """
    cfg = ws.extract_cfg
    if infer and client is None:
        raise ValueError("inference needs the caller's explicit client")
    t0 = time.perf_counter()
    replay = PageReplay(page_id=page_id, inferred=infer)

    plan: list[tuple[RegionView, int | None]] = []
    for region in EXTRACT_REGIONS:
        rv = _region_view(cfg, region)
        replay.regions.append(rv)
        if not rv.per_line:
            plan.append((rv, None))
            continue
        try:
            trusted, why = line_split_trusted(page_id, cfg)
        except ExtractError as exc:
            rv.error = str(exc)
            continue
        if not trusted:
            rv.excluded = why or "Phase 2 reported split_ok=false"
            continue
        plan.extend((rv, i) for i in line_indices_for(page_id, cfg, region))

    replay_client = ReplayOnlyClient(cfg)
    allow = infer
    for n, (rv, i) in enumerate(plan, start=1):
        if allow and cancel is not None and cancel():
            replay.aborted = "cancelled by the operator; nothing further was sent"
            allow = False
        crop = _replay_crop(page_id, cfg, rv, i,
                            client=client if allow else replay_client,
                            allow_inference=allow, write_output=write_output)
        if crop.ollama_error and allow:
            replay.aborted = f"stopped sending after an Ollama error: {crop.ollama_error}"
            allow = False
        rv.crops.append(crop)
        if progress is not None:
            progress(n, len(plan), crop)
    replay.duration_s = time.perf_counter() - t0
    return replay


@dataclass
class InferencePlan:
    """What the explicit inference action would send, worked out from a replay."""

    page_id: str
    #: (crop label, pass tag, prompt version) for every pass with no cache entry.
    missing: list[tuple[str, str, str]] = field(default_factory=list)
    #: (crop label, reason) for crops that cannot be extracted at all -- no crop,
    #: unanchored page, untrusted split. Inference cannot fix these.
    blocked: list[tuple[str, str]] = field(default_factory=list)


def crop_label(crop: CropView) -> str:
    return crop.region if crop.line_index is None else f"{crop.region} L{crop.line_index}"


def plan_inference(replay: PageReplay) -> InferencePlan:
    plan = InferencePlan(page_id=replay.page_id)
    for rv in replay.regions:
        if rv.error:
            plan.blocked.append((rv.region, classify_error(rv.error)))
        elif rv.excluded:
            plan.blocked.append((rv.region, "untrusted split"))
    for crop in replay.crops:
        for tag, pv in crop.passes():
            if pv.ok:
                continue
            label = classify_error(pv.error)
            if label == "not cached":
                plan.missing.append((crop_label(crop), tag, pv.prompt_version))
            elif (crop_label(crop), label) not in plan.blocked:
                plan.blocked.append((crop_label(crop), label))
    return plan


def ollama_version(ws: Workspace) -> tuple[str, str]:
    """``(version, base_url)`` from ``GET /api/version`` via the existing client.

    The client refuses any host outside ``[ollama].allowed_hosts`` (loopback
    only). This is the first request of the explicit inference action, made
    before anything else is sent, so the operator sees the version before
    confirming -- an Ollama auto-update has changed readings before.
    """
    client = ws.client_factory(ws.extract_cfg)
    return str(client.version()), str(client.base_url)


# ---------------------------------------------------------------------------
# stage runs (one page)
# ---------------------------------------------------------------------------


@dataclass
class StageReport:
    stage: str
    #: "ok", "warn" (ran, but the result is not usable downstream, e.g. an
    #: unanchored page) or "failed" (raised).
    status: str = "failed"
    duration_s: float = 0.0
    summary: str = ""
    lines: list[str] = field(default_factory=list)
    error: str = ""


class StageError(RuntimeError):
    pass


def load_page_image(ws: Workspace, page: PageRef):
    """Decode one page exactly as every CLI does (``iter_pages``)."""
    for img in iter_pages(page.source, ws.preprocess_cfg):
        if img.page_index != page.page_index:
            continue
        if img.page_id != page.page_id:
            raise StageError(
                f"{page.source.name} changed since the page list was built "
                f"(page_id is now {img.page_id}); rescan the page list"
            )
        return img
    raise StageError(f"page {page.page_index} not found in {page.source.name}")


def run_preprocess_stage(ws: Workspace, page: PageRef) -> StageReport:
    """``python -m src.preprocess run <file> --dump-intermediates`` for one page."""
    rep = StageReport("preprocess")
    t0 = time.perf_counter()
    try:
        img = load_page_image(ws, page)
        dump_dir = dump_dir_for(page.page_id, ws.dumps_root)
        result = run_pipeline(img, ws.preprocess_cfg, dump_dir=dump_dir)
    except Exception as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
        rep.duration_s = time.perf_counter() - t0
        return rep
    for s in result.stages:
        rep.lines.append(
            f"{s.index:02d} {s.name:<19} {'on ' if s.enabled else 'off'} "
            f"{s.duration_s:6.2f}s" + (f"  {s.dump_path.name}" if s.dump_path else "")
        )
    # The same one-line summary the preprocess CLI prints.
    skew = result.stage("deskew")
    skew_txt = ""
    if skew is not None and skew.enabled:
        a = skew.info.get("estimated_angle_deg")
        skew_txt = (
            f" skew={a:+.2f}deg{'(applied)' if skew.info.get('applied') else '(not applied)'}"
            if isinstance(a, (int, float)) else " skew=unmeasured"
        )
    binf = result.stage("binarize")
    fg_txt = ""
    if binf is not None and binf.enabled and isinstance(binf.info.get("foreground_ratio"), (int, float)):
        fg_txt = f" fg={binf.info['foreground_ratio']:.4f}"
    fh, fw = result.final.shape[:2]
    rep.summary = f"{fw}x{fh}px{skew_txt}{fg_txt} -> {rel(dump_dir)}/"
    rep.status = "ok"
    rep.duration_s = time.perf_counter() - t0
    return rep


def crop_hashes(seg_dir: Path) -> dict[str, str]:
    """sha256 of every crop the Phase 2 manifest in ``seg_dir`` lists.

    The extraction cache key is the crop's exact bytes, so comparing these
    before and after a segment run says whether cached readings still apply.
    Only crops the manifest lists count -- a stale file it no longer names is
    ignored by extraction too.
    """
    m = seg_dir / "manifest.json"
    if not m.is_file():
        return {}
    try:
        data = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    names = [str(r["dump"]) for r in data.get("regions") or [] if r.get("dump")]
    names += [str(n) for n in (data.get("prescription_lines") or {}).get("dumps") or []]
    out: dict[str, str] = {}
    for n in names:
        p = seg_dir / n
        if p.is_file():
            out[n] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def diff_crops(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "unchanged": [n for n in after if before.get(n) == after[n]],
        "changed": [n for n in after if n in before and before[n] != after[n]],
        "new": [n for n in after if n not in before],
        "dropped": [n for n in before if n not in after],
    }


def run_segment_stage(ws: Workspace, page: PageRef) -> StageReport:
    """``python -m src.segment run <file> --dump-crops`` for one page.

    Like the CLI it re-runs Phase 1 in memory (no dumps) and segments that,
    then writes crops, overlay and manifest under the segments root. It also
    reports whether each crop's bytes changed, because a changed crop no
    longer matches its cached readings.
    """
    rep = StageReport("segment")
    t0 = time.perf_counter()
    seg_dir = segment_dir_for(page.page_id, ws.segments_root)
    try:
        img = load_page_image(ws, page)
        presult = run_pipeline(img, ws.preprocess_cfg, dump_dir=None)
        templates = load_templates(ws.segment_cfg)
        before = crop_hashes(seg_dir)
        res = segment_page(img, presult, ws.segment_cfg, dump_dir=seg_dir,
                           templates=templates)
        after = crop_hashes(seg_dir)
    except Exception as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
        rep.duration_s = time.perf_counter() - t0
        return rep

    def _a(m) -> str:
        if m is None or m.score is None:
            return "n/a"
        return f"{m.score:.3f}/{m.peak_ratio:.1f}{'' if m.found else '!'}"

    lg, rg = res.anchors.get("logo"), res.anchors.get("registration_block")
    if res.anchored:
        ok = sum(1 for r in res.regions if r.geometry_ok)
        nlines = len(res.lines) if res.line_info.get("split_ok") else 0
        rep.summary = (
            f"logo={_a(lg)} reg={_a(rg)} frame={res.frame.source} "
            f"scale={res.frame.scale:.3f} regions={ok}/{len(res.regions)} "
            f"lines={nlines if nlines else 'UNSPLIT'} -> {rel(seg_dir)}/"
        )
        rep.status = "ok"
    else:
        rep.summary = (f"logo={_a(lg)} reg={_a(rg)} frame=UNANCHORED: no region "
                       f"crops written -> {rel(seg_dir)}/")
        rep.status = "warn"
    rep.lines.extend(f"! {w}" for w in res.warnings)
    d = diff_crops(before, after)
    if before:
        rep.lines.append(
            f"crops vs previous run: {len(d['unchanged'])} unchanged, "
            f"{len(d['changed'])} changed, {len(d['new'])} new, "
            f"{len(d['dropped'])} no longer listed"
        )
        if d["changed"] or d["new"]:
            rep.lines.append(
                "  changed/new crops no longer match their cached readings "
                f"(cache key = crop bytes): {', '.join(d['changed'] + d['new'])}"
            )
    else:
        rep.lines.append(f"crops written: {len(after)}")
    rep.duration_s = time.perf_counter() - t0
    return rep


def summarise_replay(replay: PageReplay) -> StageReport:
    rep = StageReport("extract")
    rep.duration_s = replay.duration_s
    rep.summary = (
        f"{replay.n_passes_present}/{replay.n_passes} passes "
        f"{'present' if replay.inferred else 'from cache'} over {len(replay.crops)} "
        f"crop(s): {replay.n_agree_read} agree, {replay.n_diff} differ, "
        f"{replay.n_withheld} withheld"
    )
    for rv in replay.regions:
        if rv.error:
            rep.lines.append(f"! {rv.region}: {rv.error}")
        if rv.excluded:
            rep.lines.append(f"! {rv.region}: excluded, Phase 2 did not trust the "
                             f"line split ({rv.excluded})")
        for crop in rv.crops:
            for tag, pv in crop.passes():
                if not pv.ok:
                    rep.lines.append(f"! {crop_label(crop)} pass {tag} "
                                     f"({pv.prompt_version}): {pv.error}")
    if replay.aborted:
        rep.lines.append(f"! {replay.aborted}")
    complete = replay.n_passes > 0 and replay.n_passes_present == replay.n_passes
    rep.status = "ok" if complete and not replay.n_problem_regions else "warn"
    return rep


# ---------------------------------------------------------------------------
# eval (the eval CLI, cache-only)
# ---------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k, ensure_ascii=False)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        # A JSON string literal is a valid TOML basic string for everything
        # json.dumps escapes; the round-trip check in write_cache_only_eval_config
        # catches the rest.
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"cannot write a {type(v).__name__} to TOML")


def dump_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for config-shaped data (tables, scalars, arrays).

    Stdlib ``tomllib`` reads TOML but cannot write it. Comments are not kept;
    the source file stays the documented one.
    """
    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: list[str]) -> None:
        scalars = [(k, v) for k, v in table.items() if not isinstance(v, dict)]
        tables = [(k, v) for k, v in table.items() if isinstance(v, dict)]
        if prefix:
            lines.append("")
            lines.append("[" + ".".join(_toml_key(p) for p in prefix) + "]")
        for k, v in scalars:
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
        for k, v in tables:
            emit(v, prefix + [k])

    emit(data, [])
    return "\n".join(lines).lstrip("\n") + "\n"


def cache_only_eval_data(ws: Workspace) -> dict[str, Any]:
    data = copy.deepcopy(ws.eval_data)
    for section in ("vitals", "extract"):
        if not isinstance(data.get(section), dict):
            raise ConfigError(f"[{section}] missing from {ws.eval_config_path}")
        data[section]["run_extraction_if_missing"] = False
    return data


def write_cache_only_eval_config(ws: Workspace) -> Path:
    """Write the eval config the TUI's eval runs against, and return its path.

    Identical to ``ws.eval_config_path`` (plus ``ws.eval_overrides``) except
    that ``run_extraction_if_missing`` is false in ``[vitals]`` and
    ``[extract]``. With that, ``python -m src.eval vitals|extract`` is a pure
    re-scoring of cached replies: it does not preflight Ollama, and a cache miss
    raises instead of being re-inferred. Written fresh before every run and
    verified by parsing it back.
    """
    data = cache_only_eval_data(ws)
    header = (
        f"# GENERATED by python -m src.tui from {rel(ws.eval_config_path)} -- do not\n"
        "# edit; it is rewritten before every eval the TUI runs. Identical to that\n"
        "# file except run_extraction_if_missing = false in [vitals] and [extract],\n"
        "# which makes the eval a pure cache replay: no Ollama preflight, and a\n"
        "# cache miss fails loudly instead of being re-inferred. Comments are in\n"
        "# the source file.\n\n"
    )
    text = header + dump_toml(data)
    try:
        roundtrip = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"could not write a cache-only copy of the eval config ({exc})") from exc
    if roundtrip != data:
        raise ConfigError("the cache-only copy of the eval config did not round-trip; "
                          "refusing to run the eval against it")
    path = ws.out_dir / "eval_cache_only.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@dataclass
class EvalRun:
    kind: str
    argv: list[str] = field(default_factory=list)
    exit_code: int | None = None
    output: str = ""
    metrics: dict[str, Any] | None = None
    json_path: Path | None = None
    duration_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.metrics is not None

    @property
    def command(self) -> str:
        return "python -m src.eval " + " ".join(
            a if " " not in a else f'"{a}"' for a in self.argv)


# One eval at a time: the run swaps sys.stdout/sys.stderr process-wide.
_EVAL_LOCK = threading.Lock()


def eval_argv(ws: Workspace, kind: str, eval_config: Path, json_out: Path) -> list[str]:
    argv = ["--eval-config", str(eval_config), kind,
            "--extract-config", str(ws.extract_config_path)]
    for o in ws.extract_overrides:
        argv += ["--set-extract", o]
    return argv + ["--json-out", str(json_out)]


def run_eval(ws: Workspace, kind: str) -> EvalRun:
    """Run ``python -m src.eval <kind>`` in-process, cache-only, and load its JSON.

    The printed report is captured verbatim for the report tab; the tables are
    built from the metrics JSON the command writes (to ``ws.out_dir``, not the
    CLI's default path). Both come from the eval code itself.
    """
    if kind not in EVAL_KINDS:
        raise ValueError(f"unknown eval {kind!r}; expected one of {EVAL_KINDS}")
    from ..eval import __main__ as eval_cli

    run = EvalRun(kind)
    t0 = time.perf_counter()
    try:
        eval_config = write_cache_only_eval_config(ws)
    except (ConfigError, OSError) as exc:
        run.error = str(exc)
        return run
    json_out = ws.out_dir / f"eval_{kind}.json"
    run.json_path = json_out
    run.argv = eval_argv(ws, kind, eval_config, json_out)
    with contextlib.suppress(FileNotFoundError):
        json_out.unlink()  # never show a previous run's numbers for this one

    buf = io.StringIO()
    with _EVAL_LOCK:
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                run.exit_code = eval_cli.main(run.argv)
        except SystemExit as exc:  # argparse
            run.exit_code = exc.code if isinstance(exc.code, int) else 2
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
    run.output = buf.getvalue()
    run.duration_s = time.perf_counter() - t0
    if run.exit_code == 0 and json_out.is_file():
        try:
            run.metrics = json.loads(json_out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            run.error = f"could not read {rel(json_out)}: {exc}"
    elif not run.error:
        tail = [ln for ln in run.output.strip().splitlines() if ln.strip()][-3:]
        run.error = (f"python -m src.eval {kind} exited with {run.exit_code}"
                     + (": " + " / ".join(tail) if tail else ""))
    return run


def iter_fields(crop: CropView) -> Iterator[tuple[str, FieldRead | None, FieldRead | None,
                                                  TwoPassField | None]]:
    """``(name, read_a, read_b, comparison)`` for each field of a crop."""
    for name in crop.fields:
        yield (name, crop.a.read(name), crop.b.read(name),
               None if crop.comparison is None else crop.comparison.get(name))


__all__ = [
    "CropView",
    "DEFAULT_OUT_DIR",
    "DumpStatus",
    "EVAL_KINDS",
    "EvalRun",
    "InferencePlan",
    "PageRef",
    "PageReplay",
    "PassView",
    "RegionView",
    "ReplayOnlyClient",
    "SegmentStatus",
    "StageReport",
    "Workspace",
    "cache_only_eval_data",
    "classify_error",
    "crop_hashes",
    "crop_label",
    "discover_pages",
    "dump_toml",
    "iter_fields",
    "ollama_version",
    "plan_inference",
    "read_dump_status",
    "read_fixture_statuses",
    "read_segment_status",
    "rel",
    "replay_page",
    "run_eval",
    "run_preprocess_stage",
    "run_segment_stage",
    "summarise_replay",
    "write_cache_only_eval_config",
]
