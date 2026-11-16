"""Phase 3/4 orchestration: crop -> cache lookup -> Ollama -> parse -> result.

    data/segments/<page_id>/<region crop>.png      (input, written by Phase 2)
      -> data/out/extract_cache/<region>/<model>/<prompt_version>/<hash>.json
      -> data/out/extract/<page_id>.<region>[.lineN].<prompt>.json  (eval input)

Deliberately downstream of Phase 2 and blind to Phase 1: it reads the crops
``python -m src.segment run --dump-crops`` already wrote rather than re-deriving
them. Those crops are the reviewed artefacts -- the operator checked them against
``overlay.png`` before the region boxes were fixed -- and re-cropping here would
create a second code path that could disagree with them about what the model was
shown.

Pages are discovered from ``data/fixtures/*.json``, the only place page_ids are
enumerated without re-running ingestion, and the same discovery the eval harness
uses.

FAILING SAFELY (spec section 1.3). Two different kinds of thing can go wrong and
they are handled differently on purpose:

* a *setup* problem -- Ollama not running, the model not pulled, no crop for a
  page because segmentation was never run, a crop from a page Phase 2 could not
  anchor -- raises. It never degrades into a skipped page, and the message names
  the command that fixes it. Reporting accuracy over a silently-shortened page
  set would be the worst possible failure of an eval harness.
* a *model-output* problem -- an unparseable reply, an implausible number, a
  missing key -- is a normal outcome. It produces nulls with recorded reasons and
  the run continues, because "the model could not read this" is a result the
  harness exists to measure.

PHI: everything written here is keyed on ``page_id`` (a content hash). No source
filename, no patient name, ever -- the same rule as Phase 1 and Phase 2. Both
output trees live under ``data/out/``, which .gitignore excludes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..preprocess.config import PROJECT_ROOT, Config
from ..eval.fixtures import load_fixtures
from . import cache as cache_mod
from .cache import CacheKey, make_key
from .client import ModelReply, OllamaClient, OllamaError
from .config import (
    EXTRACT_REGIONS,
    PRESCRIPTION_LINE_REGION,
    VITALS_FIELDS,
    VITALS_REGION,
    fixtures_dir,
    output_dir,
    prompt_spec,
    region_fields,
    region_model_options,
    region_spec,
    segments_dir,
)
from .parse import FieldRead, ParsedReply, parse_reply

RESULT_SCHEMA_VERSION = 1


class ExtractError(RuntimeError):
    """A setup problem that must not be degraded into a skipped page."""


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    page_id: str
    region: str                     # the Phase 2 region name, e.g. "vitals_column"
    crop_path: str                  # relative to PROJECT_ROOT; no patient filename
    crop_shape_hw: list[int] | None
    image_hash: str
    model: str
    prompt_version: str
    #: The ``[regions.<name>]`` table this came from, e.g. "vitals". Distinct
    #: from ``region``, which is the Phase 2 crop name used in the cache key.
    region_key: str = ""
    #: The ``[prompts.<name>]`` table used. Two passes over the same region
    #: differ only in this and in ``prompt_version``.
    prompt_name: str = ""
    #: Which prescription line crop this is, or None for a whole-region crop.
    line_index: int | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    reads: dict[str, FieldRead] = field(default_factory=dict)
    # The model's reply, kept as EVIDENCE for debugging. Never read as a fallback
    # value: if parsing failed, the field is null, not "whatever the text said".
    raw_text: str = ""
    raw_text_field: str = ""
    json_ok: bool = False
    json_salvaged: bool = False
    from_cache: bool = False
    duration_s: float = 0.0
    model_meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.reads.values() if r.ok)

    @property
    def n_absent(self) -> int:
        """Fields the model asserted are not written on this slip."""
        return sum(1 for r in self.reads.values() if r.status == "absent")

    @property
    def n_failed(self) -> int:
        """Fields withheld because something failed (Phase 5 will flag these)."""
        return sum(1 for r in self.reads.values() if r.failed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "phase": 4,
            "stage": "vlm_field_extraction",
            # Deliberately no source filename: filenames contain patient names.
            "page_id": self.page_id,
            "region": self.region,
            "region_key": self.region_key,
            "prompt_name": self.prompt_name,
            "line_index": self.line_index,
            "crop_path": self.crop_path,
            "crop_shape_hw": self.crop_shape_hw,
            "image_hash": self.image_hash,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "fields": self.fields,
            "field_reads": {k: v.as_dict() for k, v in self.reads.items()},
            "raw_response": self.raw_text,
            "raw_response_field": self.raw_text_field,
            "json_ok": self.json_ok,
            "json_salvaged": self.json_salvaged,
            "from_cache": self.from_cache,
            "duration_s": round(self.duration_s, 3),
            "model_meta": self.model_meta,
            "warnings": self.warnings,
            "not_implemented": {
                "two_pass_agreement": (
                    "this file is ONE pass. Spec section 4.1's agreement signal is "
                    "derived by src/extract/agreement.py from two of these results "
                    "and written to data/out/extract_agreement/; it is a signal, "
                    "not a resolution -- neither pass is picked as the answer."
                ),
                "field_state_machine": (
                    "spec section 4 gives every field a state of confirmed / "
                    "extracted / flagged. Phase 3 records a per-field status and "
                    "reason instead; the state machine, and the flagging that "
                    "drives it, are phase 5."
                ),
                "self_reported_confidence": (
                    "not collected, and must not be. Spec section 4 opens by "
                    "forbidding model self-reported confidence as a signal."
                ),
            },
        }


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discover_page_ids(cfg: Config, fixtures_root: Path | None = None,
                      require_ground_truth: bool = False) -> list[tuple[str, str]]:
    """``[(page_id, ground_truth_status), ...]`` from the fixture directory.

    ``fixtures_root`` lets the eval harness pass the directory from its own
    config (``[fixtures].dir`` in config/eval.toml) so that path has one source of
    truth; a disagreement with ``[inputs].fixtures_dir`` is warned about by the
    caller rather than silently resolved.
    """
    root = Path(fixtures_root) if fixtures_root is not None else fixtures_dir(cfg)
    fixtures = load_fixtures(root)
    out = [(fx.page_id, fx.ground_truth_status) for fx in fixtures]
    if require_ground_truth:
        out = [(pid, st) for pid, st in out if st != "empty"]
    return out


def crop_path_for(page_id: str, cfg: Config, region: str = VITALS_REGION,
                  line_index: int | None = None) -> Path:
    spec = region_spec(cfg, region)
    name = str(spec["crop_file"])
    if bool(spec.get("per_line")):
        if line_index is None:
            raise ExtractError(
                f"region {region!r} is per-line; a line_index is required to name a "
                f"crop file"
            )
        name = name.format(index=line_index)
    return segments_dir(cfg) / page_id / name


def line_indices_for(page_id: str, cfg: Config,
                     region: str = PRESCRIPTION_LINE_REGION) -> list[int]:
    """The line crops Phase 2 actually wrote for this page, in top-to-bottom order.

    Read from the Phase 2 manifest's ``prescription_lines.dumps`` list rather
    than by globbing the directory, so a stale crop left behind by an earlier
    segmentation run cannot silently join the batch. Returns ``[]`` when Phase 2
    reported the split untrustworthy -- in which case there are no line crops at
    all and the page is excluded, visibly, by the caller.
    """
    data = _load_manifest(page_id, cfg)
    presc = (data.get("prescription_lines") or {})
    info = presc.get("info") or {}
    if not info.get("split_ok", False):
        return []
    return list(range(len(presc.get("dumps") or [])))


def line_split_trusted(page_id: str, cfg: Config) -> tuple[bool, str]:
    """``(trusted, reason)`` straight from the Phase 2 manifest.

    Phase 2's own words: when the band count is implausible it writes no line
    crops and records why, "rather than a plausible-looking line list". Phase 4
    propagates that judgement instead of re-deriving it.
    """
    info = ((_load_manifest(page_id, cfg).get("prescription_lines") or {})
            .get("info") or {})
    ok = bool(info.get("split_ok", False))
    return ok, "" if ok else str(info.get("reason", "Phase 2 reported split_ok=false"))


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _load_manifest(page_id: str, cfg: Config) -> dict[str, Any]:
    manifest = segments_dir(cfg) / page_id / "manifest.json"
    if not manifest.is_file():
        raise ExtractError(
            f"no Phase 2 manifest at {_rel(manifest)}, so the crop's provenance "
            f"cannot be verified. Re-run:\n"
            f"  python -m src.segment run samples --dump-crops"
        )
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractError(f"{_rel(manifest)}: invalid JSON ({exc})") from exc
    return data


def _check_manifest_geometry(page_id: str, cfg: Config, segment_region: str,
                             line_index: int | None = None) -> list[str]:
    """Refuse a crop whose Phase 2 manifest says the geometry was unsound.

    A crop cut from an unanchored page is a crop of an unknown part of the
    document. Reading numbers off it would produce plausible values with no
    provenance, which is precisely what spec section 1.3 forbids -- so this
    raises rather than extracting and hoping. Returns informational notes when
    the geometry is sound but imperfect (a clamped box).
    """
    if not bool(cfg.get("inputs", "require_manifest_geometry_ok")):
        return [
            "[inputs].require_manifest_geometry_ok is false; Phase 2 geometry was "
            "not verified before extraction"
        ]
    data = _load_manifest(page_id, cfg)
    manifest = segments_dir(cfg) / page_id / "manifest.json"

    frame = data.get("frame") or {}
    if not frame.get("ok"):
        raise ExtractError(
            f"page {page_id} was not anchored by Phase 2 "
            f"(frame reason: {frame.get('reason', 'unknown')}), so "
            f"{segment_region!r} is a crop of an unknown part of the page. Refusing "
            f"to extract field values from it -- spec section 1.3."
        )
    entry = next(
        (r for r in data.get("regions", []) if r.get("name") == segment_region), None
    )
    if entry is None:
        raise ExtractError(
            f"{_rel(manifest)} has no region named {segment_region!r}; check "
            f"[regions.{VITALS_REGION}].segment_region in the extract config against "
            f"src.segment.config.REGION_NAMES"
        )
    if not entry.get("enabled", True):
        raise ExtractError(
            f"region {segment_region!r} is disabled in config/segment.toml for page "
            f"{page_id}, so there is no crop to extract from"
        )
    if not entry.get("geometry_ok"):
        raise ExtractError(
            f"page {page_id}: Phase 2 reported region {segment_region!r} geometry as "
            f"unsound (notes: {entry.get('notes')}). Refusing to extract from it."
        )

    notes: list[str] = []
    if entry.get("clamped"):
        frac = entry.get("clamped_frac")
        notes.append(
            f"Phase 2 clamped the {segment_region!r} box to the page edge, losing "
            f"{(frac or 0) * 100:.2f}% of the requested area; part of the region may "
            f"be missing from the crop"
        )
    if frame.get("source", "").startswith("single_anchor"):
        notes.append(
            f"Phase 2 built a degraded single-anchor frame for this page "
            f"({frame.get('source')}), so the crop is less precisely placed"
        )
    if line_index is not None:
        # A line crop inherits its block's geometry AND the line splitter's own
        # verdict. Phase 2 refuses to write line crops at all when the band count
        # is implausible, so reaching here with split_ok false means the caller
        # built a filename Phase 2 never wrote.
        info = (data.get("prescription_lines") or {}).get("info") or {}
        if not info.get("split_ok", False):
            raise ExtractError(
                f"page {page_id}: Phase 2 reported the prescription line split as "
                f"untrustworthy, so line crops were never written. Refusing to "
                f"extract per-line fields from an unsplit block."
            )
        dumps = (data.get("prescription_lines") or {}).get("dumps") or []
        if line_index >= len(dumps):
            raise ExtractError(
                f"page {page_id}: Phase 2 wrote {len(dumps)} line crop(s); there is "
                f"no line {line_index}"
            )
    return notes


def _crop_shape(image_bytes: bytes) -> list[int] | None:
    """``[h, w]`` from the PNG header, without decoding the image.

    Recorded so a metrics file says what the model was actually shown. Read from
    the IHDR chunk rather than through OpenCV: this module has no other reason to
    decode pixels, and a header read cannot fail in a way that stops extraction.
    """
    if len(image_bytes) < 24 or image_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if image_bytes[12:16] != b"IHDR":
        return None
    w = int.from_bytes(image_bytes[16:20], "big")
    h = int.from_bytes(image_bytes[20:24], "big")
    return [h, w]


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


def _result_from_cache(entry: dict[str, Any], key: CacheKey, cfg: Config,
                       crop: Path, shape: list[int] | None,
                       segment_region: str, notes: list[str],
                       field_names: tuple[str, ...]) -> ExtractionResult:
    """Re-parse a cached reply rather than trusting its stored field values.

    This is the point of caching the *reply* and not just the numbers: the
    expensive part is inference, the cheap part is parsing, and the parser and its
    plausibility ranges in config/extract.toml are exactly what the operator will
    be tuning. Re-parsing means a range edit takes effect on the next run without
    a 25-minute re-inference, and it means a cached entry can never disagree with
    the config the operator is reading.
    """
    parsed = parse_reply(str(entry.get("raw_response", "")), cfg, field_names)
    res = ExtractionResult(
        page_id=str(entry.get("page_id", "")),
        region=segment_region,
        crop_path=_rel(crop),
        crop_shape_hw=shape,
        image_hash=key.image_hash,
        model=key.model,
        prompt_version=key.prompt_version,
        fields=parsed.values(),
        reads=parsed.fields,
        raw_text=str(entry.get("raw_response", "")),
        raw_text_field=str(entry.get("raw_response_field", "")),
        json_ok=parsed.json_ok,
        json_salvaged=parsed.json_salvaged,
        from_cache=True,
        duration_s=0.0,
        model_meta=dict(entry.get("model_meta") or {}),
        warnings=list(notes) + list(entry.get("warnings") or []),
    )
    return res


def extract_region(
    page_id: str,
    cfg: Config,
    client: OllamaClient,
    region: str = VITALS_REGION,
    refresh: bool = False,
    write_output: bool = True,
    allow_inference: bool = True,
    prompt_name: str | None = None,
    line_index: int | None = None,
) -> ExtractionResult:
    """Extract one region of one page. Raises :class:`ExtractError` on setup faults.

    ``allow_inference=False`` makes this a cache-only lookup that raises on a
    miss. That is what ``run_extraction_if_missing = false`` in config/eval.toml
    uses to keep scoring a pure re-read of already-produced results -- and it
    raises rather than returning nulls, because a cache miss scored as "the model
    read nothing" would silently understate accuracy.

    ``prompt_name`` selects which ``[prompts.*]`` table to send; the default is
    the region's pass-A prompt. This is the whole of what the two-pass agreement
    check needs from this function -- the prompt version is already part of the
    spec section 3.5 cache key, so the two passes cache independently and neither
    can serve the other's reply.
    """
    t0 = time.perf_counter()
    rspec = region_spec(cfg, region)
    segment_region = str(rspec["segment_region"])
    prompt_name = prompt_name or str(rspec["prompt"])
    pspec = prompt_spec(cfg, prompt_name)
    prompt_version = str(pspec["version"])
    prompt_text = str(pspec["text"]).strip()
    field_names = region_fields(cfg, region)

    crop = crop_path_for(page_id, cfg, region, line_index)
    if not crop.is_file():
        raise ExtractError(
            f"no {segment_region!r} crop for page {page_id} at {_rel(crop)}. "
            f"Extraction reads the crops Phase 2 wrote; produce them with:\n"
            f"  python -m src.segment run samples --dump-crops\n"
            f"Not skipping this page: an accuracy figure over a silently-shortened "
            f"page set would be misleading."
        )
    notes = _check_manifest_geometry(page_id, cfg, segment_region, line_index)
    image_bytes = crop.read_bytes()
    shape = _crop_shape(image_bytes)
    key = make_key(image_bytes, segment_region, cfg, prompt_version)

    if not refresh:
        entry = cache_mod.load(key, cfg)
        if entry is not None:
            res = _result_from_cache(entry, key, cfg, crop, shape, segment_region,
                                     notes, field_names)
            res.page_id = page_id
            res.region_key = region
            res.prompt_name = prompt_name
            res.line_index = line_index
            res.duration_s = time.perf_counter() - t0
            if write_output:
                _write_result(res, cfg)
            return res

    if not allow_inference:
        raise ExtractError(
            f"no cached extraction for page {page_id} region {segment_region!r} "
            f"(model {key.model!r}, prompt {prompt_version!r}), and inference is "
            f"disabled. Either run:\n"
            f"  python -m src.extract regions\n"
            f"or set run_extraction_if_missing = true in config/eval.toml. "
            f"Not scoring this page as unread -- that would understate accuracy."
        )

    reply: ModelReply = client.generate(
        prompt_text, image_bytes, options=region_model_options(cfg, region)
    )
    parsed: ParsedReply = parse_reply(reply.text, cfg, field_names)

    res = ExtractionResult(
        page_id=page_id,
        region=segment_region,
        region_key=region,
        prompt_name=prompt_name,
        line_index=line_index,
        crop_path=_rel(crop),
        crop_shape_hw=shape,
        image_hash=key.image_hash,
        model=key.model,
        prompt_version=prompt_version,
        fields=parsed.values(),
        reads=parsed.fields,
        raw_text=reply.text,
        raw_text_field=reply.text_field,
        json_ok=parsed.json_ok,
        json_salvaged=parsed.json_salvaged,
        from_cache=False,
        model_meta=reply.meta(),
        warnings=list(notes) + list(reply.warnings),
    )
    if not parsed.json_ok:
        res.warnings.append(
            f"reply was not usable JSON ({parsed.reason}); all {len(field_names)} "
            f"field(s) are null"
        )
    if parsed.json_salvaged:
        res.warnings.append(
            "the JSON object was recovered from a brace substring of a longer "
            "reply (envelope repair only -- no value was altered)"
        )

    # Successes are cached; failures are not (see [cache].store_failures). A
    # failure is usually a property of the environment rather than of the
    # (image, region, model, prompt) tuple, and caching it would make a transient
    # outage look like a permanent inability to read a page.
    cacheable = parsed.json_ok or bool(cfg.get("cache", "store_failures"))
    if bool(cfg.get("cache", "enabled")) and cacheable:
        cache_mod.store(
            key,
            cfg,
            {
                "page_id": page_id,
                "region_key": region,
                "prompt_name": prompt_name,
                "line_index": line_index,
                "crop_path": res.crop_path,
                "crop_shape_hw": shape,
                # The reply text, not the parsed numbers: see _result_from_cache.
                "raw_response": res.raw_text,
                "raw_response_field": res.raw_text_field,
                "model_meta": res.model_meta,
                "warnings": list(reply.warnings),
                "prompt_text_sha256_prefix": cache_mod.hash_image(
                    prompt_text.encode("utf-8"), cfg
                )[0][:16],
            },
        )

    res.duration_s = time.perf_counter() - t0
    if write_output:
        _write_result(res, cfg)
    return res


def result_filename(res: ExtractionResult, cfg: Config) -> str:
    """Name of the per-page artefact for one extraction result.

    ``<page_id>.<region>[.lineN][.<prompt_version>].json``. The prompt version is
    in the name because Phase 4 runs every region twice: with Phase 3's
    ``<page_id>.<region>.json`` the second pass would overwrite the first and the
    file on disk would silently be "whichever pass ran last". The line number is
    in the name for the same reason -- one prescription block is 4-7 crops.
    """
    parts = [res.page_id, res.region]
    if res.line_index is not None:
        parts.append(f"line{res.line_index}")
    if bool(cfg.get("output", "filename_includes_prompt_version")):
        parts.append(res.prompt_version)
    return ".".join(parts) + ".json"


def _write_result(res: ExtractionResult, cfg: Config) -> Path | None:
    """The human-facing per-page artefact.

    Separate from the cache on purpose: the cache is keyed for machine lookup
    (by crop hash), this is keyed for a person asking "what did the model
    actually say about page X, and why is this field empty".
    """
    if not bool(cfg.get("output", "write_per_page_json")):
        return None
    d = output_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    p = d / result_filename(res, cfg)
    p.write_text(json.dumps(res.as_dict(), indent=2), encoding="utf-8")
    return p


def extract_vitals_batch(
    cfg: Config,
    client: OllamaClient,
    page_ids: list[str],
    refresh: bool = False,
    write_output: bool = True,
) -> list[ExtractionResult]:
    """Extract the vitals region for a list of pages, in order.

    No parallelism: inference is CPU-only (spec section 1.2) and Ollama serialises
    requests to one model anyway, so concurrency here would add scheduling noise
    to the per-page timings without shortening the batch.
    """
    out: list[ExtractionResult] = []
    for pid in page_ids:
        out.append(
            extract_region(
                pid, cfg, client, region=VITALS_REGION,
                refresh=refresh, write_output=write_output,
            )
        )
    return out


def extract_page_region(
    page_id: str,
    cfg: Config,
    client: OllamaClient,
    region: str,
    prompt_name: str | None = None,
    refresh: bool = False,
    write_output: bool = True,
    allow_inference: bool = True,
) -> list[ExtractionResult]:
    """One region of one page, as a LIST of results -- one per crop.

    Every region but the prescription block has exactly one crop per page, so the
    list has one element. The prescription block has one crop per detected line,
    so the list has 4-7 -- and is EMPTY when Phase 2 declared the split
    untrustworthy. An empty list is not an error and not a zero score: the caller
    reports the page as excluded, with Phase 2's reason.
    """
    if not bool(region_spec(cfg, region).get("per_line")):
        return [extract_region(page_id, cfg, client, region=region,
                               prompt_name=prompt_name, refresh=refresh,
                               write_output=write_output,
                               allow_inference=allow_inference)]
    return [
        extract_region(page_id, cfg, client, region=region, prompt_name=prompt_name,
                       refresh=refresh, write_output=write_output,
                       allow_inference=allow_inference, line_index=i)
        for i in line_indices_for(page_id, cfg, region)
    ]


__all__ = [
    "EXTRACT_REGIONS",
    "ExtractError",
    "ExtractionResult",
    "OllamaError",
    "RESULT_SCHEMA_VERSION",
    "VITALS_FIELDS",
    "crop_path_for",
    "discover_page_ids",
    "extract_page_region",
    "extract_region",
    "extract_vitals_batch",
    "line_indices_for",
    "line_split_trusted",
    "result_filename",
]
