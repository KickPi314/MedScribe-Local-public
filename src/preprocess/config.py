"""Config loading for the preprocessing pipeline.

All tunable parameters live in config/preprocess.toml. This module only loads
and validates; it never supplies a tuned value that is absent from the file
without saying so, so a missing key is a loud error rather than a silent
built-in default that diverges from the config the operator is reading.

TOML is parsed with the stdlib tomllib (Python 3.11+), so no extra dependency.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "preprocess.toml"

STEP_NAMES = ("channel_separation", "deskew", "clahe", "binarize", "crop")

_REQUIRED_SECTIONS = ("ingest", "steps", "dump", *STEP_NAMES)


class ConfigError(RuntimeError):
    """Raised when the config file is missing, malformed or incomplete."""


class Config:
    """Thin read-only wrapper over the parsed TOML tree.

    Access is explicit (``cfg.get("clahe", "clip_limit")``) so that a typo in a
    parameter name fails immediately instead of silently falling back.
    """

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        self._data = data
        self.path = path

    # -- access ----------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        try:
            sec = self._data[name]
        except KeyError as exc:
            raise ConfigError(
                f"config section [{name}] missing from {self.path}"
            ) from exc
        if not isinstance(sec, dict):
            raise ConfigError(f"config section [{name}] is not a table")
        return sec

    def get(self, section: str, key: str) -> Any:
        sec = self.section(section)
        try:
            return sec[key]
        except KeyError as exc:
            raise ConfigError(
                f"config key {key!r} missing from section [{section}] in {self.path}"
            ) from exc

    def get_opt(self, section: str, key: str, default: Any) -> Any:
        return self.section(section).get(key, default)

    def step_enabled(self, step: str) -> bool:
        return bool(self.get("steps", step))

    def step_order(self) -> list[str]:
        order = list(self.get("steps", "order"))
        unknown = [s for s in order if s not in STEP_NAMES]
        if unknown:
            raise ConfigError(f"[steps].order contains unknown step(s): {unknown}")
        missing = [s for s in STEP_NAMES if s not in order]
        if missing:
            raise ConfigError(
                f"[steps].order omits step(s) {missing}; list every step and use "
                f"the boolean flags to disable them instead"
            )
        return order

    def as_dict(self) -> dict[str, Any]:
        """Deep-ish copy for embedding in dump manifests."""
        import copy

        return copy.deepcopy(self._data)

    # -- mutation (experiments only) -------------------------------------
    def with_overrides(self, overrides: dict[str, dict[str, Any]]) -> "Config":
        """Return a copy with ``{section: {key: value}}`` applied.

        Used by the eval harness to sweep one parameter (e.g. the
        channel-separation method) without editing the operator's config file.
        """
        data = self.as_dict()
        for section, kv in overrides.items():
            data.setdefault(section, {})
            data[section].update(kv)
        return Config(data, self.path)


def apply_cli_overrides(cfg: Config, assignments: list[str]) -> Config:
    """Apply ``SECTION.KEY=TOML_VALUE`` strings from a ``--set`` CLI flag.

    For one-off experiments, so a sweep does not require editing (and then
    forgetting to restore) the operator's config file.
    """
    overrides: dict[str, dict[str, Any]] = {}
    for a in assignments:
        if "=" not in a or "." not in a.split("=", 1)[0]:
            raise ConfigError(f"--set expects SECTION.KEY=VALUE, got {a!r}")
        lhs, rhs = a.split("=", 1)
        section, key = lhs.split(".", 1)
        try:
            value = tomllib.loads(f"v = {rhs}")["v"]
        except Exception as exc:
            raise ConfigError(
                f"--set value for {lhs} is not valid TOML: {rhs!r} ({exc}). "
                f"Strings need quotes, e.g. --set 'deskew.method=\"projection\"'"
            ) from exc
        overrides.setdefault(section.strip(), {})[key.strip()] = value
    return cfg.with_overrides(overrides) if overrides else cfg


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    cfg = Config(data, p)
    for sec in _REQUIRED_SECTIONS:
        cfg.section(sec)
    cfg.step_order()
    return cfg
