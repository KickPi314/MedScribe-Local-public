"""Config loading for VLM field extraction (Phases 3-4).

Same contract as ``src.preprocess.config`` and ``src.segment.config``: every
tunable lives in ``config/extract.toml``, access is explicit, and a missing key
raises :class:`~src.preprocess.config.ConfigError` rather than falling back to a
built-in default that diverges from the file the operator is reading.

The :class:`~src.preprocess.config.Config` wrapper is reused rather than
reimplemented, so ``--set SECTION.KEY=VALUE`` overrides behave identically to
Phase 1's and Phase 2's.

Three validations here are not merely "is the key present":

* the Ollama host is checked against ``[ollama].allowed_hosts``, so a typo (or
  an edit) cannot turn the one socket this codebase opens into an off-machine
  call. Spec section 1.1 forbids network access at runtime and spec section 10
  forbids a cloud fallback "even behind a flag, even for comparison"; making
  that mechanical is cheaper than trusting a comment.
* the vitals region's declared ``fields`` must be exactly the keys of
  ``Fixture.fields["vitals"]``, so extraction output and ground truth cannot
  drift apart into a shape mismatch the eval harness would have to paper over.
* Phase 4 extends that rule to every other region: each region's field names
  must be keys of ``src.eval.fixtures.empty_fields()`` (or, for the per-line
  prescription region, keys of one ``prescriptions[]`` entry). There is still no
  translation layer between extraction output and ground truth, and adding a
  field to a prompt without a place to score it now fails at config load.

PHASE 4 ADDITIONS, all config-level rather than code-level:

* every region names TWO prompts (``prompt`` and ``prompt_b``) for the spec
  section 4.1 two-pass agreement signal. Neither is "the" answer -- see
  src/extract/agreement.py.
* fields carry a ``type`` (number / text / date / string_list /
  complaint_list). Phase 3 only had numbers, so ``type`` defaults to "number"
  and the six vitals field tables are unchanged.
* a region may carry a ``[regions.X.model_options]`` table, merged over
  ``[model.options]``. Needed because the crops differ in area by 6x and image
  tokens scale with area: one global ``num_ctx`` would silently truncate the
  diagnosis crop's image tokens.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..preprocess.config import PROJECT_ROOT, Config, ConfigError

DEFAULT_EXTRACT_CONFIG_PATH = PROJECT_ROOT / "config" / "extract.toml"

# The one region Phase 3 extracted (spec section 9 item 3: "one region type").
# The name of the ``[regions.<name>]`` table, not the Phase 2 region name --
# that is ``[regions.vitals].segment_region``.
VITALS_REGION = "vitals"

# The per-line prescription region. Singled out because it is the only region
# whose crop is one of N per page, and whose fields are keys of a
# ``prescriptions[]`` entry rather than top-level fixture keys.
PRESCRIPTION_LINE_REGION = "prescription_line"

# Every extractable region, in report order. Phase 4 completes spec section
# 3.4's list; the names are the ``[regions.<name>]`` table names.
#
# ``date`` is the one region with no counterpart in spec section 3.4. It was
# added after Phase 4 measured that the handwritten visit date sat between
# ``header_strip`` and ``vitals_column`` and was in no Phase 2 crop at all, so
# every date read as null for a geometric reason. It reads the
# ``date_row`` box added to config/segment.toml for it.
EXTRACT_REGIONS = (
    "header",
    "date",
    VITALS_REGION,
    "complaints",
    "diagnosis",
    "investigations",
    PRESCRIPTION_LINE_REGION,
)

# Field order used in every table, JSON dump and cache entry. Kept as a tuple
# here so output ordering is stable regardless of dict iteration order, and
# cross-checked against the config's own list in :func:`region_spec`.
VITALS_FIELDS = ("wt_kg", "temp_f", "spo2_pct", "hr", "mac_cm", "abd_cm")

# Field value types. A field's type decides which validation path in
# src/extract/parse.py it goes through; every path ends in "a value that failed
# a check is null with a recorded reason".
TYPE_NUMBER = "number"
TYPE_TEXT = "text"
TYPE_DATE = "date"
TYPE_STRING_LIST = "string_list"
TYPE_COMPLAINT_LIST = "complaint_list"
FIELD_TYPES = (TYPE_NUMBER, TYPE_TEXT, TYPE_DATE, TYPE_STRING_LIST, TYPE_COMPLAINT_LIST)

_REQUIRED_SECTIONS = (
    "ollama",
    "model",
    "inputs",
    "regions",
    "prompts",
    "parse",
    "fields",
    "cache",
    "output",
)

_REQUIRED_REGION_KEYS = ("crop_file", "segment_region", "prompt", "prompt_b", "fields")

# Required keys per field type. "label" and "aliases" are universal: the label is
# what the human sees in a table, the aliases are alternative JSON keys accepted
# from the model (a key's spelling is not a transcribed value).
_REQUIRED_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    TYPE_NUMBER: ("label", "aliases", "min", "max", "decimals"),
    TYPE_TEXT: ("label", "aliases", "max_chars"),
    TYPE_DATE: ("label", "aliases", "min_year", "max_year"),
    TYPE_STRING_LIST: ("label", "aliases", "max_items", "max_item_chars"),
    TYPE_COMPLAINT_LIST: (
        "label",
        "aliases",
        "max_items",
        "max_item_chars",
        "min_duration",
        "max_duration",
        "duration_units",
    ),
}


def reject_nested_overrides(assignments: list[str]) -> None:
    """Refuse a ``--set`` that tries to reach into a nested TOML table.

    ``src.preprocess.config.apply_cli_overrides`` understands
    ``SECTION.KEY=VALUE`` and nothing deeper: it splits on the *first* dot, so
    ``--set 'prompts.vitals_v1.version="v2"'`` quietly creates a flat key literally
    named ``"vitals_v1.version"`` in ``[prompts]`` and the intended value is never
    read. That is a silent no-op, and a silent no-op in an eval harness is how a
    run gets reported under the wrong model or prompt version.

    This config has more nested tables than Phase 1's did ([prompts.*],
    [regions.*], [fields.*], [model.options]), so the failure is easy to hit.
    Rather than extend the Phase 1 override syntax -- which every other stage
    already depends on -- the deep case is rejected with the instruction to edit
    the file. Editing the file is the right move anyway: a prompt or a
    plausibility range is a versioned decision, not a per-run flag.

    ``--set 'model.name="llava:7b"'`` -- the swap spec section 3.2 asks for -- is
    a top-level section and key, so it is unaffected.
    """
    for a in assignments:
        lhs = a.split("=", 1)[0].strip()
        if lhs.count(".") > 1:
            section = lhs.split(".", 1)[0]
            raise ConfigError(
                f"--set {lhs!r} targets a nested table, which the "
                f"SECTION.KEY=VALUE override syntax cannot reach -- it would "
                f"silently create a flat key {lhs.split('.', 1)[1]!r} in "
                f"[{section}] and change nothing. Edit config/extract.toml "
                f"directly for values inside [prompts.*], [regions.*], [fields.*] "
                f"or [model.options]."
            )


def load_extract_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_EXTRACT_CONFIG_PATH
    if not p.is_file():
        raise ConfigError(f"extract config file not found: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    cfg = Config(data, p)
    for sec in _REQUIRED_SECTIONS:
        cfg.section(sec)

    # -- the loopback guard (see the module docstring) --------------------
    host = str(cfg.get("ollama", "host"))
    allowed = [str(h) for h in cfg.get("ollama", "allowed_hosts")]
    if host not in allowed:
        raise ConfigError(
            f"[ollama].host is {host!r}, which is not in [ollama].allowed_hosts "
            f"{allowed} in {p}. Spec section 1.1 requires zero network at "
            f"runtime: this client may only talk to a loopback address, so that "
            f"the app still works with the network adapter disabled. If you were "
            f"pointing this at a remote inference server, stop -- spec section 10 "
            f"forbids a cloud API fallback even behind a flag."
        )
    endpoint = str(cfg.get("ollama", "endpoint"))
    if endpoint not in ("generate", "chat"):
        raise ConfigError(
            f"[ollama].endpoint is {endpoint!r} in {p}; expected \"generate\" or "
            f"\"chat\""
        )
    prefill = str(cfg.get_opt("model", "assistant_prefill", ""))
    if prefill and endpoint != "chat":
        raise ConfigError(
            f"[model].assistant_prefill is set in {p} but [ollama].endpoint is "
            f"{endpoint!r}; a prefill can only be sent as a final assistant "
            f"message on /api/chat. Set endpoint = \"chat\" or clear the prefill."
        )
    fields_sec = [str(f) for f in cfg.get("model", "response_fields")]
    if not fields_sec:
        raise ConfigError(
            f"[model].response_fields is empty in {p}; there would be nowhere to "
            f"read the model's reply from"
        )

    # -- regions / prompts / fields must be complete, not partial ---------
    # Which field names are scoreable, derived from the fixture template rather
    # than restated here, so a rename in src/eval/fixtures.py breaks config load
    # instead of quietly producing an unscoreable field.
    from ..eval.fixtures import empty_fields

    fixture_keys = set(empty_fields())
    prescription_keys = {"form", "product", "dose", "frequency"}

    all_fields = cfg.section("fields")
    for region_name in EXTRACT_REGIONS:
        region = region_spec(cfg, region_name)
        for key in _REQUIRED_REGION_KEYS:
            if key not in region:
                raise ConfigError(f"[regions.{region_name}] is missing key {key!r} in {p}")

        declared = tuple(str(f) for f in region["fields"])
        if not declared:
            raise ConfigError(f"[regions.{region_name}].fields is empty in {p}")
        if region_name == VITALS_REGION and declared != VITALS_FIELDS:
            raise ConfigError(
                f"[regions.{VITALS_REGION}].fields is {list(declared)} in {p} but must "
                f"be exactly {list(VITALS_FIELDS)}, in that order -- these are the keys "
                f"of Fixture.fields['vitals'] in src/eval/fixtures.py, and extraction "
                f"output is compared against ground truth key-by-key with no "
                f"translation layer"
            )
        if region_name not in (VITALS_REGION, PRESCRIPTION_LINE_REGION):
            unknown = [f for f in declared if f not in fixture_keys]
            if unknown:
                raise ConfigError(
                    f"[regions.{region_name}].fields names {unknown} in {p}, which are "
                    f"not keys of Fixture.fields (src/eval/fixtures.py: "
                    f"{sorted(fixture_keys)}). A field with nowhere to be scored would "
                    f"cost inference time and produce a number no one can check."
                )
        if region_name == PRESCRIPTION_LINE_REGION:
            unknown = [f for f in declared if f not in prescription_keys]
            if unknown:
                raise ConfigError(
                    f"[regions.{PRESCRIPTION_LINE_REGION}].fields names {unknown} in "
                    f"{p}; a prescription line's fields must be keys of one "
                    f"Fixture.fields['prescriptions'] entry {sorted(prescription_keys)}. "
                    f"gujarati_marginalia_crop is deliberately not extractable: spec "
                    f"section 4 keeps the marginalia as an image crop and forbids "
                    f"transcribing it."
                )
            if not bool(region.get("per_line")):
                raise ConfigError(
                    f"[regions.{PRESCRIPTION_LINE_REGION}].per_line must be true in {p}"
                )
        if bool(region.get("per_line")) and "{index}" not in str(region["crop_file"]):
            raise ConfigError(
                f"[regions.{region_name}].crop_file {region['crop_file']!r} in {p} is "
                f"per-line but has no '{{index}}' placeholder for the line number"
            )

        # Two prompts per region, spec section 4.1. Both must exist and they must
        # not be the same table: two passes with one prompt cannot disagree, and
        # an agreement rate computed from that would be a fabricated signal.
        pa, pb = str(region["prompt"]), str(region["prompt_b"])
        prompt_spec(cfg, pa)
        prompt_spec(cfg, pb)
        if pa == pb:
            raise ConfigError(
                f"[regions.{region_name}] names the same prompt {pa!r} for both "
                f"passes in {p}. Spec section 4.1 requires two DIFFERENTLY WORDED "
                f"prompts; identical passes would agree by construction and the "
                f"agreement rate would be a fabricated confidence signal."
            )

        for name in declared:
            if name not in all_fields:
                raise ConfigError(
                    f"[fields.{name}] table missing from {p}; every field named in "
                    f"[regions.{region_name}].fields needs its own validation rules"
                )
            _check_field_spec(name, all_fields[name], p)
    return cfg


def _check_field_spec(name: str, spec: Any, p: Path) -> None:
    if not isinstance(spec, dict):
        raise ConfigError(f"[fields.{name}] is not a table in {p}")
    ftype = str(spec.get("type", TYPE_NUMBER))
    if ftype not in FIELD_TYPES:
        raise ConfigError(
            f"[fields.{name}].type is {ftype!r} in {p}; expected one of "
            f"{list(FIELD_TYPES)}"
        )
    for key in _REQUIRED_FIELD_KEYS[ftype]:
        if key not in spec:
            raise ConfigError(
                f"[fields.{name}] (type {ftype!r}) is missing key {key!r} in {p}"
            )
    if ftype == TYPE_NUMBER and float(spec["min"]) >= float(spec["max"]):
        raise ConfigError(
            f"[fields.{name}] has min {spec['min']} >= max {spec['max']} in {p}; "
            f"no value could ever pass, so every read would be discarded"
        )
    if ftype == TYPE_DATE and int(spec["min_year"]) >= int(spec["max_year"]):
        raise ConfigError(
            f"[fields.{name}] has min_year >= max_year in {p}; no date could pass"
        )
    if ftype == TYPE_COMPLAINT_LIST:
        units = spec["duration_units"]
        if not isinstance(units, dict) or not units:
            raise ConfigError(
                f"[fields.{name}.duration_units] must be a non-empty table of "
                f"canonical_unit = [accepted spellings] in {p}. It is the only "
                f"place a duration unit is recognised, so an empty one would "
                f"withhold every complaint list that carries a unit."
            )
    for key in ("max_chars", "max_items", "max_item_chars"):
        if key in spec and int(spec[key]) <= 0:
            raise ConfigError(f"[fields.{name}].{key} must be positive in {p}")


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------


def region_spec(cfg: Config, name: str) -> dict[str, Any]:
    spec = cfg.section("regions").get(name)
    if not isinstance(spec, dict):
        raise ConfigError(
            f"[regions.{name}] table missing from {cfg.path}. The extractable "
            f"regions are {list(EXTRACT_REGIONS)} (spec section 3.4)."
        )
    return spec


def region_fields(cfg: Config, name: str) -> tuple[str, ...]:
    """The field names a region is expected to return, in report order."""
    return tuple(str(f) for f in region_spec(cfg, name)["fields"])


def region_prompt_names(cfg: Config, name: str) -> tuple[str, str]:
    """``(pass_a_prompt, pass_b_prompt)`` for a region.

    Two prompts, not one with a variant: spec section 4.1 wants the SAME ask
    worded differently, so that a systematic misreading induced by one phrasing
    shows up as a disagreement rather than as a confident wrong answer twice.
    Neither is the primary; pass A is simply the one Phase 3 measured.
    """
    spec = region_spec(cfg, name)
    return str(spec["prompt"]), str(spec["prompt_b"])


def field_type(cfg: Config, name: str) -> str:
    return str(field_spec(cfg, name).get("type", TYPE_NUMBER))


def prompt_spec(cfg: Config, name: str) -> dict[str, Any]:
    spec = cfg.section("prompts").get(name)
    if not isinstance(spec, dict):
        raise ConfigError(
            f"[prompts.{name}] table missing from {cfg.path}, but a region asks for it"
        )
    for key in ("version", "text"):
        if key not in spec:
            raise ConfigError(f"[prompts.{name}] is missing key {key!r} in {cfg.path}")
    if not str(spec["text"]).strip():
        raise ConfigError(f"[prompts.{name}].text is empty in {cfg.path}")
    return spec


def field_spec(cfg: Config, name: str) -> dict[str, Any]:
    spec = cfg.section("fields").get(name)
    if not isinstance(spec, dict):
        raise ConfigError(f"[fields.{name}] table missing from {cfg.path}")
    return spec


def model_options(cfg: Config) -> dict[str, Any]:
    """The ``[model.options]`` table, passed through to Ollama verbatim.

    Not validated key-by-key on purpose: Ollama's option set differs between
    versions and models, and this stage has no business deciding which sampling
    knobs a future VLM understands. What it does not do is *supply* one -- an
    option absent from the config is absent from the request.
    """
    opts = cfg.section("model").get("options")
    if not isinstance(opts, dict):
        raise ConfigError(f"[model.options] table missing from {cfg.path}")
    return dict(opts)


def region_model_options(cfg: Config, region: str) -> dict[str, Any]:
    """``[model.options]`` with ``[regions.<region>.model_options]`` merged over it.

    Phase 4 needs this because image tokens scale with crop AREA and the six
    region crops differ by 6x: measured 0.15-0.21 MPx for header_strip against
    0.93-1.33 MPx for diagnosis_line. One global ``num_ctx`` sized for the vitals
    column would silently truncate the diagnosis crop's image tokens, and a
    truncated image is an unreadable one that the model would answer anyway.

    Only the keys a region names are overridden; everything else (temperature 0,
    top_k 1, seed) still comes from the one place it is documented.
    """
    opts = model_options(cfg)
    override = region_spec(cfg, region).get("model_options")
    if override is None:
        return opts
    if not isinstance(override, dict):
        raise ConfigError(
            f"[regions.{region}.model_options] is not a table in {cfg.path}"
        )
    opts.update(dict(override))
    return opts


def ollama_base_url(cfg: Config) -> str:
    host = str(cfg.get("ollama", "host"))
    port = int(cfg.get("ollama", "port"))
    # IPv6 literals need brackets in a URL authority.
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}"


def segments_dir(cfg: Config) -> Path:
    return PROJECT_ROOT / str(cfg.get("inputs", "segments_dir"))


def fixtures_dir(cfg: Config) -> Path:
    return PROJECT_ROOT / str(cfg.get("inputs", "fixtures_dir"))


def cache_dir(cfg: Config) -> Path:
    return PROJECT_ROOT / str(cfg.get("cache", "dir"))


def output_dir(cfg: Config) -> Path:
    return PROJECT_ROOT / str(cfg.get("output", "dir"))
