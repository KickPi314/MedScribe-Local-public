"""Single-region VLM field extraction (Phase 3, spec section 9 item 3).

Downstream of ``src.segment``: it consumes the region crops Phase 2 wrote to
``data/segments/<page_id>/`` and does not change them. The job is to turn ONE
region crop into structured field values by sending it to a local VLM with one
narrow prompt, and to say honestly which of those values can be trusted.

The region is the vitals column. That is spec section 3.4's own worked example
("Read the vitals column, output JSON with keys wt/temp/spo2/hr" beats
"transcribe this page" on both accuracy and speed) and it is the only region for
which every ground-truth fixture already carries human-verified values, which is
what makes spec section 8's requirement -- measure against real scans, never
synthetic ones -- satisfiable today.

Scope is narrow on purpose (spec section 9 says do not build this end-to-end in
one pass). NOT here, and not stubbed either: the other regions and two-pass
agreement (phase 4), the formulary and dose-sanity flagging (phase 5), the review
UI (phase 6), export and the audit log (phase 7), and the SQLite job queue from
spec section 3.5.

Two constraints shape every module in this package.

**Zero network at runtime (spec section 1.1).** ``client.py`` is the only place in
the codebase that opens a socket, and it may only talk to a loopback address from
``[ollama].allowed_hosts``. A loopback request cannot leave the machine, so the
app still behaves correctly with the network adapter disabled -- which is the test
section 1.1 sets. Spec section 10's ban on a cloud fallback "even behind a flag,
even for comparison" is enforced by that allowlist, not by a comment.

**Never guess (spec section 1.3).** The prompt asks the model to withhold values
it cannot read; ``parse.py`` does not believe that it did. Every value is
re-derived from the raw reply and must be valid JSON, under a known key, reducible
to a number, and inside a configured plausibility range. A value that fails any
check is stored as ``None`` with the reason recorded -- never repaired, never
rounded into range, never replaced by the nearest plausible number. The raw reply
is kept beside the parsed result as evidence for debugging, never as a fallback
value.

Every tunable -- host, port, model name, prompt text, prompt version, sampling
parameters, unit tokens, plausibility ranges, cache layout -- lives in
``config/extract.toml``.
"""

from .config import (
    DEFAULT_EXTRACT_CONFIG_PATH,
    VITALS_FIELDS,
    VITALS_REGION,
    load_extract_config,
)
from .client import ModelReply, OllamaClient, OllamaError
from .parse import FieldRead, ParsedReply, parse_vitals
from .cache import CacheKey, make_key
from .pipeline import (
    ExtractError,
    ExtractionResult,
    crop_path_for,
    discover_page_ids,
    extract_region,
    extract_vitals_batch,
)

__all__ = [
    "DEFAULT_EXTRACT_CONFIG_PATH",
    "VITALS_FIELDS",
    "VITALS_REGION",
    "load_extract_config",
    "ModelReply",
    "OllamaClient",
    "OllamaError",
    "FieldRead",
    "ParsedReply",
    "parse_vitals",
    "CacheKey",
    "make_key",
    "ExtractError",
    "ExtractionResult",
    "crop_path_for",
    "discover_page_ids",
    "extract_region",
    "extract_vitals_batch",
]
