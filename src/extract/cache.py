"""Result cache keyed on (image hash, region, model, prompt version).

Spec section 3.5, verbatim: "Cache results per (image hash, region, model,
prompt version) so re-runs and re-tuning don't recompute everything." All four
parts are in the key and all four are stored in plain text inside each entry, so
an entry can be audited without recomputing its key.

The image hash is of the *region crop's* exact PNG bytes, not of the source page.
That is the stricter and more useful choice: re-tuning a region box in
config/segment.toml changes the crop bytes and therefore misses the cache, which
is correct -- the model would have been shown a different picture. Hashing the
page instead would silently serve a stale reading of a box that no longer exists.

Deliberately NOT a SQLite job queue. Spec section 3.5 also calls for a
persistent queue that survives restart, but that is processing-model
infrastructure for a batch of hundreds of pages (build order phase 3 is
"single-region VLM extraction, one region type, measured against ground truth").
Seven cached JSON files over one region do not need a queue, and one file per
result is inspectable with a text editor, which a database row is not. When the
queue arrives it can read these same keys.

PHI: entries hold a real patient's observations and the model's raw reply about
them, so ``[cache].dir`` lives under ``data/out/``, which .gitignore excludes for
exactly this reason. Filenames carry the crop hash and the model/prompt version
only -- never a page's source filename, which would contain a patient name.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..preprocess.config import Config, ConfigError
from .config import cache_dir

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheKey:
    """The spec section 3.5 tuple, plus the algorithm that produced the hash."""

    image_hash: str
    region: str
    model: str
    prompt_version: str
    hash_algorithm: str

    def as_dict(self) -> dict[str, str]:
        return {
            "image_hash": self.image_hash,
            "hash_algorithm": self.hash_algorithm,
            "region": self.region,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


def _slug(s: str) -> str:
    """Filesystem-safe token. ``qwen3-vl:4b`` -> ``qwen3-vl_4b``.

    Needed because model names contain ``:``, which is not a legal Windows
    filename character. The unslugged original is stored inside the entry, so the
    slug is presentation only and is never what the key is compared on.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unnamed"


def hash_image(image_bytes: bytes, cfg: Config) -> tuple[str, str]:
    """``(digest_hex, algorithm)`` for the crop bytes."""
    algo = str(cfg.get("cache", "hash_algorithm"))
    try:
        h = hashlib.new(algo)
    except ValueError as exc:
        raise ConfigError(
            f"[cache].hash_algorithm {algo!r} is not available in hashlib ({exc})"
        ) from exc
    h.update(image_bytes)
    return h.hexdigest(), algo


def make_key(image_bytes: bytes, region: str, cfg: Config, prompt_version: str) -> CacheKey:
    digest, algo = hash_image(image_bytes, cfg)
    return CacheKey(
        image_hash=digest,
        region=region,
        model=str(cfg.get("model", "name")),
        prompt_version=prompt_version,
        hash_algorithm=algo,
    )


def path_for(key: CacheKey, cfg: Config) -> Path:
    """Where the entry for ``key`` lives.

    Laid out as ``<dir>/<region>/<model>/<prompt_version>/<hash-prefix>.json`` so
    that switching model (spec section 3.2 requires the model to be swappable)
    puts the new results in a sibling directory instead of overwriting the old
    ones -- which is what makes two models comparable by running the eval harness
    twice.
    """
    n = int(cfg.get("cache", "filename_hash_chars"))
    return (
        cache_dir(cfg)
        / _slug(key.region)
        / _slug(key.model)
        / _slug(key.prompt_version)
        / f"{key.image_hash[:n]}.json"
    )


def load(key: CacheKey, cfg: Config) -> dict[str, Any] | None:
    """Return a cached entry, or ``None`` on any miss.

    Every part of the key is re-verified against the entry's stored key before it
    is trusted. The filename holds only a truncated digest, so this is what makes
    a prefix collision impossible to observe as a wrong answer: a mismatch is
    reported as a miss and the page is re-inferred, never served from the wrong
    patient's entry.
    """
    if not bool(cfg.get("cache", "enabled")):
        return None
    p = path_for(key, cfg)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get("cache_schema_version", -1)) != CACHE_SCHEMA_VERSION:
        return None
    if data.get("key") != key.as_dict():
        return None
    return data


def store(key: CacheKey, cfg: Config, payload: dict[str, Any]) -> Path:
    p = path_for(key, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "key": key.as_dict(),
        **payload,
    }
    p.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return p
