"""Config loading for region segmentation (Phase 2).

Same contract as ``src.preprocess.config``: every tunable lives in
``config/segment.toml``, access is explicit, and a missing key raises
:class:`~src.preprocess.config.ConfigError` rather than falling back to a
built-in default that diverges from the file the operator is reading.

The :class:`~src.preprocess.config.Config` wrapper is reused rather than
reimplemented, so ``--set SECTION.KEY=VALUE`` overrides behave identically to
Phase 1's.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from ..preprocess.config import PROJECT_ROOT, Config, ConfigError

DEFAULT_SEGMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "segment.toml"

# Anchors this stage knows how to match. Both are pre-printed letterhead, so
# neither can be corrupted by handwriting content (spec section 2).
ANCHOR_NAMES = ("logo", "registration_block")

# Region names, in the order spec section 3.4 lists them, plus ``date_row``.
# The prescription block is additionally split into per-line crops by
# src.segment.lines.
#
# ``date_row`` is NOT one of spec section 3.4's boxes. It was added after Phase 4
# measured that the handwritten visit date fell between ``header_strip`` (which
# ends at x 1.20) and ``vitals_column`` (which starts at y 0.267) and was
# therefore in no crop at all. It sits second because that is where it is on the
# page, and because ``crop_regions`` iterates this tuple directly -- a region
# table in config/segment.toml that is not named here is never cropped.
REGION_NAMES = (
    "header_strip",
    "date_row",
    "vitals_column",
    "complaints_block",
    "diagnosis_line",
    "investigations_margin",
    "prescription_block",
)

_REQUIRED_SECTIONS = (
    "templates",
    "anchors",
    "frame",
    "frame_check",
    "source",
    "regions",
    "lines",
    "dump",
    "proxy",
)

_TEMPLATE_SPEC_KEYS = ("file", "reference_box_px", "search_box_frac")


def load_segment_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_SEGMENT_CONFIG_PATH
    if not p.is_file():
        raise ConfigError(f"segment config file not found: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    cfg = Config(data, p)
    for sec in _REQUIRED_SECTIONS:
        cfg.section(sec)

    # Fail loudly on a partially-specified template/anchor/region set rather
    # than silently segmenting fewer regions than the operator expects.
    tpl = cfg.section("templates")
    for name in ANCHOR_NAMES:
        if name not in tpl:
            raise ConfigError(
                f"[templates] is missing the {name!r} table in {p}; every anchor in "
                f"{list(ANCHOR_NAMES)} needs a template definition"
            )
        for key in _TEMPLATE_SPEC_KEYS:
            if key not in tpl[name]:
                raise ConfigError(
                    f"[templates.{name}] is missing key {key!r} in {p}"
                )
    for name, spec in check_specs(cfg).items():
        for key in _TEMPLATE_SPEC_KEYS:
            if key not in spec:
                raise ConfigError(
                    f"[frame_check.{name}] is missing key {key!r} in {p}"
                )
    regions = cfg.section("regions")
    missing = [r for r in REGION_NAMES if r not in regions]
    if missing:
        raise ConfigError(
            f"[regions] in {p} omits region(s) {missing}; every region in "
            f"{list(REGION_NAMES)} must have a box (use enabled=false in "
            f"[regions.<name>] to skip one)"
        )
    return cfg


def templates_dir(cfg: Config) -> Path:
    return PROJECT_ROOT / str(cfg.get("templates", "dir"))


def template_path(cfg: Config, filename: str) -> Path:
    return templates_dir(cfg) / str(filename)


def check_specs(cfg: Config) -> dict[str, dict]:
    """The ``[frame_check.<name>]`` landmark tables, in config order.

    Landmarks are *verification* templates: matched, compared against the
    position the anchor frame predicts, and never used to build the frame.
    """
    return {
        k: v for k, v in cfg.section("frame_check").items() if isinstance(v, dict)
    }


def template_specs(cfg: Config) -> dict[str, dict]:
    """Every template that needs an asset on disk: anchors first, then checks."""
    tpl = cfg.section("templates")
    out: dict[str, dict] = {name: tpl[name] for name in ANCHOR_NAMES}
    out.update(check_specs(cfg))
    return out
