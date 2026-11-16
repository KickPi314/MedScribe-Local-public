"""Two-pass agreement: a signal, not a decision (Phase 4, spec section 4.1).

Spec section 4 opens by forbidding self-reported model confidence and then gives
the first independent substitute for it:

    "Two-pass agreement. Run each region twice with differently-worded prompts.
     Disagreement between passes is your strongest error signal. Cost is 2x,
     worth it."

This module runs a region through both of its configured prompts and records,
per field, whether the two passes produced the same value. That is all it does.

WHAT IT DELIBERATELY DOES NOT DO, because these are phase 5:

* it does not pick a winner. There is no ``value`` on :class:`TwoPassField`, no
  "consensus", no "prefer pass A". Both passes' values are kept side by side and
  the caller has to look at both. Collapsing them here would destroy the signal
  the spec is asking for and would smuggle in a resolution rule that nobody has
  measured.
* it does not flag anything. There is no field state, no threshold at which a
  disagreement becomes an action. Spec section 4's ``confirmed`` / ``extracted``
  / ``flagged`` state machine is phase 5, and this is its raw material.
* it does not treat agreement as correctness. Two passes can agree on a wrong
  reading -- the eval harness measures exactly how often that happens on the real
  corpus, and reports it under its own name rather than as accuracy.

AGREEMENT IS ON THE PARSED VALUE, not on the raw reply text. Two replies that
say ``"5.990"`` and ``5.99`` have read the same weight and differ only in JSON
formatting; two that say ``"PHARYNGITIS"`` and ``"PHARYNGITIS "`` differ only in
whitespace the parser already collapses. Comparing raw text would report those as
disagreements and drown the real ones. Case and spelling are NOT normalised here
-- "Pantop" vs "Pentop" is exactly the disagreement spec section 6 wants shown to
a human.

BOTH-NULL IS RECORDED SEPARATELY. Two passes that both read nothing agree that
there is nothing, which is genuine agreement (and the right answer on the five
pages with no investigations) -- but it is a much weaker signal than two passes
agreeing on the same transcribed string, and an agreement rate that quietly
counted mostly-empty regions as agreement would flatter the model. So
``both_null`` travels with every comparison and the eval harness reports the
rates with and without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..preprocess.config import PROJECT_ROOT, Config
from .client import OllamaClient
from .config import region_fields, region_prompt_names, region_spec
from .pipeline import ExtractionResult, extract_region, line_indices_for

AGREEMENT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def canonical(value: Any) -> Any:
    """A value in the form two passes are compared in.

    JSON-normalising only: an int and a float that are numerically equal compare
    equal (5 vs 5.0 is a formatting difference, not two different heart rates),
    and lists/dicts are compared structurally. No case folding, no fuzzy
    matching, no rounding -- those would decide that two different readings are
    "close enough", which is the judgement this module exists to hand to a human.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    return value


def values_agree(a: Any, b: Any) -> bool:
    return canonical(a) == canonical(b)


@dataclass
class TwoPassField:
    """One field, seen twice. Deliberately has no single ``value``."""

    name: str
    value_a: Any
    value_b: Any
    status_a: str
    status_b: str

    @property
    def agree(self) -> bool:
        return values_agree(self.value_a, self.value_b)

    @staticmethod
    def _empty(value: Any) -> bool:
        """No value here.

        An EMPTY LIST counts as no value, alongside ``None``. The parser stores
        ``[]`` when the model explicitly reported "no complaints" / "no
        investigations", which is a real answer and is scored as one -- but for
        the agreement rate it belongs with the nulls, because two passes that both
        found nothing agree trivially and 5 of the 7 real slips request no
        investigations at all. Counting those as "both read a value and agreed"
        would put the strict agreement rate up by a third for free.
        """
        return value is None or (isinstance(value, list) and not value)

    @property
    def both_null(self) -> bool:
        """Both passes produced no value (absent, empty, or withheld)."""
        return self._empty(self.value_a) and self._empty(self.value_b)

    @property
    def both_read(self) -> bool:
        return not self._empty(self.value_a) and not self._empty(self.value_b)

    @property
    def status_agree(self) -> bool:
        """Same *reason*, not just the same value.

        Two passes that both produced null -- one because the model reported the
        field absent, one because the reply failed to parse -- agree on the value
        and disagree on why. Phase 5 will care about that difference; recording
        it now costs one boolean.
        """
        return self.status_a == self.status_b

    @property
    def detail(self) -> str:
        if self.agree:
            return ""
        return f"pass A {self.value_a!r} [{self.status_a}] vs pass B {self.value_b!r} [{self.status_b}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.name,
            "pass_a_value": self.value_a,
            "pass_b_value": self.value_b,
            "pass_a_status": self.status_a,
            "pass_b_status": self.status_b,
            "agree": self.agree,
            "status_agree": self.status_agree,
            "both_null": self.both_null,
            "both_read": self.both_read,
            "detail": self.detail,
        }


@dataclass
class TwoPassResult:
    """Both passes over one crop, plus the per-field comparison."""

    page_id: str
    region_key: str
    segment_region: str
    line_index: int | None
    pass_a: ExtractionResult
    pass_b: ExtractionResult
    fields: dict[str, TwoPassField] = field(default_factory=dict)

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def n_agree(self) -> int:
        return sum(1 for f in self.fields.values() if f.agree)

    @property
    def n_agree_both_read(self) -> int:
        return sum(1 for f in self.fields.values() if f.agree and f.both_read)

    @property
    def n_both_read(self) -> int:
        return sum(1 for f in self.fields.values() if f.both_read)

    @property
    def disagreements(self) -> list[TwoPassField]:
        return [f for f in self.fields.values() if not f.agree]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGREEMENT_SCHEMA_VERSION,
            "phase": 4,
            "stage": "two_pass_agreement",
            "page_id": self.page_id,
            "region_key": self.region_key,
            "region": self.segment_region,
            "line_index": self.line_index,
            "prompt_a": self.pass_a.prompt_name,
            "prompt_a_version": self.pass_a.prompt_version,
            "prompt_b": self.pass_b.prompt_name,
            "prompt_b_version": self.pass_b.prompt_version,
            "model": self.pass_a.model,
            "n_fields": self.n_fields,
            "n_agree": self.n_agree,
            "fields": [f.as_dict() for f in self.fields.values()],
            "pass_a": self.pass_a.as_dict(),
            "pass_b": self.pass_b.as_dict(),
            "not_implemented": {
                "resolution": (
                    "no winner is picked and no merged value exists. Deciding which "
                    "pass to believe is a validation rule, and spec section 4's "
                    "validation and flagging arrive in phase 5. Both readings are "
                    "kept so a human can see them side by side (spec section 6: "
                    "'passes disagreed: Pantop / Pentop')."
                ),
                "flagging": (
                    "a disagreement is not a flag. There is no field state machine "
                    "until phase 5; this file records the signal it will consume."
                ),
                "self_reported_confidence": (
                    "not collected, and must not be -- spec section 4 opens by "
                    "forbidding it. Agreement between two differently-worded prompts "
                    "is the independent substitute."
                ),
            },
        }


# ---------------------------------------------------------------------------
# running both passes
# ---------------------------------------------------------------------------


def compare(pass_a: ExtractionResult, pass_b: ExtractionResult,
            field_names: tuple[str, ...]) -> dict[str, TwoPassField]:
    out: dict[str, TwoPassField] = {}
    for name in field_names:
        ra, rb = pass_a.reads.get(name), pass_b.reads.get(name)
        out[name] = TwoPassField(
            name=name,
            value_a=None if ra is None else ra.value,
            value_b=None if rb is None else rb.value,
            status_a="missing_read" if ra is None else ra.status,
            status_b="missing_read" if rb is None else rb.status,
        )
    return out


def run_two_pass(
    page_id: str,
    cfg: Config,
    client: OllamaClient,
    region: str,
    line_index: int | None = None,
    refresh: bool = False,
    write_output: bool = True,
    allow_inference: bool = True,
) -> TwoPassResult:
    """Run one crop through both of a region's prompts and compare.

    Inference is not reimplemented here: both passes go through the same
    :func:`src.extract.pipeline.extract_region` a single pass uses, so a value
    scored as "pass A" is byte-identical to what ``python -m src.extract vitals``
    produces, and each pass hits its own cache entry (the spec section 3.5 key
    includes the prompt version).
    """
    prompt_a, prompt_b = region_prompt_names(cfg, region)
    a = extract_region(page_id, cfg, client, region=region, prompt_name=prompt_a,
                       line_index=line_index, refresh=refresh,
                       write_output=write_output, allow_inference=allow_inference)
    b = extract_region(page_id, cfg, client, region=region, prompt_name=prompt_b,
                       line_index=line_index, refresh=refresh,
                       write_output=write_output, allow_inference=allow_inference)
    res = TwoPassResult(
        page_id=page_id,
        region_key=region,
        segment_region=str(region_spec(cfg, region)["segment_region"]),
        line_index=line_index,
        pass_a=a,
        pass_b=b,
        fields=compare(a, b, region_fields(cfg, region)),
    )
    if write_output:
        write_agreement(res, cfg)
    return res


def run_two_pass_region(
    page_id: str,
    cfg: Config,
    client: OllamaClient,
    region: str,
    refresh: bool = False,
    write_output: bool = True,
    allow_inference: bool = True,
) -> list[TwoPassResult]:
    """Two passes over every crop of one region of one page.

    One element for a whole-region crop; one per line for the prescription
    block; empty when Phase 2 did not trust the line split (the caller reports
    that page as excluded rather than as zero agreement).
    """
    per_line = bool(region_spec(cfg, region).get("per_line"))
    indices: list[int | None] = (
        list(line_indices_for(page_id, cfg, region)) if per_line else [None]
    )
    return [
        run_two_pass(page_id, cfg, client, region, line_index=i, refresh=refresh,
                     write_output=write_output, allow_inference=allow_inference)
        for i in indices
    ]


def agreement_dir(cfg: Config) -> Path:
    return PROJECT_ROOT / str(cfg.get("output", "agreement_dir"))


def write_agreement(res: TwoPassResult, cfg: Config) -> Path | None:
    """One JSON per compared crop, under data/out/ (gitignored, PHI)."""
    if not bool(cfg.get("output", "write_agreement_json")):
        return None
    d = agreement_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    parts = [res.page_id, res.segment_region]
    if res.line_index is not None:
        parts.append(f"line{res.line_index}")
    p = d / (".".join(parts) + ".agreement.json")
    p.write_text(json.dumps(res.as_dict(), indent=2), encoding="utf-8")
    return p


__all__ = [
    "AGREEMENT_SCHEMA_VERSION",
    "TwoPassField",
    "TwoPassResult",
    "agreement_dir",
    "canonical",
    "compare",
    "run_two_pass",
    "run_two_pass_region",
    "values_agree",
    "write_agreement",
]
