"""Per-field exact-match accuracy for the vitals region (Phase 3, spec section 8).

This is the first *real* accuracy metric in the codebase. Phase 1 and Phase 2
could only produce proxies, because there was nothing to compare against ground
truth; Phase 3 extracts six numeric fields per page and every fixture carries
human-transcribed values for them, so exact-match accuracy is finally
computable -- on real scans, which is the one thing spec section 8 is emphatic
about ("Evaluate on real scans only. Synthetic evaluation will lie to you").

WHY EXACT MATCH AND NOTHING ELSE HERE. Vitals are structured numeric fields.
22.65 kg and 22.6 kg are two different weights, not a partially-correct reading,
so character error rate would flatter a wrong number; CER is the right metric for
the free-text regions phase 4 extracts, and is reported NOT IMPLEMENTED until
then. Flag precision/recall needs a `flagged` field state to measure and the
validation rules of spec section 4 to produce it -- that is phase 5. See
:data:`PENDING_VITALS_METRICS`.

THE OUTCOME TAXONOMY IS THE POINT. A single accuracy percentage hides the
distinction spec section 8 says matters most -- "a system that is 70% accurate
and correctly flags its own 30% is far more useful here than one that is 85%
accurate and silently wrong". So every field lands in one of six outcomes, and
they are grouped two ways:

* correct / incorrect, which gives the accuracy figure; and
* did the system EMIT a value it should not be trusted on, or did it WITHHOLD
  one? A wrong value that reaches a human as if it were read is the dangerous
  failure. A withheld value is the safe one.

That second grouping is *not* flag precision/recall -- there is no flag state
yet, and nothing here is a substitute for one. It is the raw material those
metrics will be computed from in phase 5, reported honestly under its own name.

Ground-truth nulls are scored, not skipped. On 3 of the 7 real slips MAC and Abd
are genuinely not recorded, and "the model correctly reported that nothing is
written there" is a real success that a naive scorer would either drop or count
as a miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..extract.parse import ABSENT
from ..extract.pipeline import ExtractionResult
from .fixtures import Fixture, STATUS_PARTIAL

# Metrics that cannot exist FOR THIS REGION, and why. Two of them were unblocked
# by Phase 4 and now exist elsewhere; the entries stay so that the vitals report
# keeps saying what it does and does not measure, and they point at where the
# real number is rather than claiming it is missing.
PENDING_VITALS_METRICS = {
    "field_character_error_rate": (
        "not applicable to this region and never will be: CER is the metric for "
        "FREE-TEXT fields, and vitals are structured numeric fields where a "
        "one-character error is a different measurement, not a partially correct "
        "one. Implemented for the free-text regions in phase 4 -- see "
        "src/eval/region_metrics.py and `python -m src.eval extract`."
    ),
    "flag_precision": (
        "needs the `flagged` field state and the validation rules of spec "
        "section 4 (phase 5). Phase 3 records a withholding reason per field, "
        "which is the input to this, not the metric."
    ),
    "flag_recall": "same as flag_precision (phase 5)",
    "false_alarm_rate": (
        "a flagged field that turned out to be correctly extracted -- "
        "undefined until fields can be flagged (phase 5)"
    ),
    "missed_error_rate": (
        "an `extracted` field that was wrong -- undefined until fields have "
        "states (phase 5). The `emitted_wrong` count below is the closest "
        "honest measurement available now and is deliberately not called this."
    ),
    "two_pass_agreement_rate": (
        "not measured by THIS command, which is the Phase 3 single-pass table "
        "and is kept unchanged so its 38/42 stays comparable. Phase 4 added a "
        "second vitals prompt (vitals_v2) and reports the agreement rate for "
        "this region alongside the other five: `python -m src.eval extract`."
    ),
}

# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------

CORRECT = "correct"                    # ground truth has a value, and it matched
CORRECT_NULL = "correct_null"          # nothing recorded on the slip, none reported
WRONG = "wrong"                        # a value was reported, and it is not the truth
SPURIOUS = "spurious"                  # nothing recorded, but a value was reported
MISSED_ABSENT = "missed_absent"        # a value is recorded; model said there is none
MISSED_WITHHELD = "missed_withheld"    # a value is recorded; extraction withheld it

OUTCOMES = (CORRECT, CORRECT_NULL, WRONG, SPURIOUS, MISSED_ABSENT, MISSED_WITHHELD)

#: Outcomes counted as correct by the exact-match metric.
CORRECT_OUTCOMES = (CORRECT, CORRECT_NULL)

#: A value reached the output and is not the truth. The dangerous failure: a
#: human reviewing the draft has no signal that it is wrong.
EMITTED_WRONG_OUTCOMES = (WRONG, SPURIOUS)

#: The truth was lost, but nothing false was asserted. The safe failure, and the
#: behaviour spec section 1.3 demands when the system is not confident.
WITHHELD_OUTCOMES = (MISSED_ABSENT, MISSED_WITHHELD)

_OUTCOME_MARK = {
    CORRECT: "ok",
    CORRECT_NULL: "ok-null",
    WRONG: "WRONG",
    SPURIOUS: "SPURIOUS",
    MISSED_ABSENT: "MISSED-abs",
    MISSED_WITHHELD: "MISSED-wh",
}


@dataclass
class FieldScore:
    page_id: str
    name: str
    ground_truth: float | int | None
    predicted: float | int | None
    #: The extraction status from src.extract.parse (ok / absent / out_of_range /
    #: unparseable / missing_key / no_json). Carried through so a miss can be
    #: attributed to the model, the parser or the plausibility range.
    extract_status: str
    outcome: str
    reason: str = ""

    @property
    def correct(self) -> bool:
        return self.outcome in CORRECT_OUTCOMES

    @property
    def mark(self) -> str:
        return _OUTCOME_MARK[self.outcome]

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "field": self.name,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "extract_status": self.extract_status,
            "outcome": self.outcome,
            "correct": self.correct,
            "reason": self.reason,
        }


@dataclass
class PageVitalsScore:
    page_id: str
    ground_truth_status: str
    from_cache: bool
    duration_s: float
    json_ok: bool
    raw_text: str
    raw_text_field: str
    scores: list[FieldScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_correct(self) -> int:
        return sum(1 for s in self.scores if s.correct)

    @property
    def partial_ground_truth(self) -> bool:
        """A null here might mean "not transcribed yet" rather than "not recorded".

        Flagged so the caveat travels with the number instead of living only in a
        config comment.
        """
        return self.ground_truth_status == STATUS_PARTIAL

    def score(self, name: str) -> FieldScore | None:
        return next((s for s in self.scores if s.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "ground_truth_status": self.ground_truth_status,
            "partial_ground_truth": self.partial_ground_truth,
            "from_cache": self.from_cache,
            "duration_s": round(self.duration_s, 3),
            "json_ok": self.json_ok,
            "n_correct": self.n_correct,
            "n_fields": len(self.scores),
            "fields": [s.as_dict() for s in self.scores],
            # The reply is kept as evidence so a wrong score can be traced to what
            # the model actually said, without re-running inference.
            "raw_response": self.raw_text,
            "raw_response_field": self.raw_text_field,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _round(value: float | int | None, decimals: int) -> float | int | None:
    if value is None:
        return None
    r = round(float(value), decimals)
    return int(r) if decimals == 0 else r


def score_field(
    page_id: str,
    name: str,
    truth: Any,
    result: ExtractionResult,
    decimals: int,
    tolerance: float,
) -> FieldScore:
    """Classify one field into one of :data:`OUTCOMES`.

    Both sides are rounded to the field's configured ``decimals`` before
    comparison, so 55 and 55.0 match while 24.8 and 24.82 do not. ``tolerance``
    is 0.0 for exact match; it exists as a config value so that a future
    free-text or derived field can loosen it explicitly rather than by accident.
    """
    read = result.reads.get(name)
    status = read.status if read is not None else "missing_read"
    reason = read.reason if read is not None else "no extraction record for this field"

    gt = _round(truth if isinstance(truth, (int, float)) and not isinstance(truth, bool)
                else None, decimals)
    if truth is not None and gt is None:
        # Ground truth exists but is not a number. A fixture problem, not a model
        # problem, and it must not be silently scored as a null.
        return FieldScore(
            page_id, name, None, None, status, MISSED_WITHHELD,
            f"ground truth {truth!r} is not numeric; fix the fixture",
        )
    pred = _round(result.fields.get(name), decimals)

    if gt is None and pred is None:
        return FieldScore(page_id, name, None, None, status, CORRECT_NULL, reason)
    if gt is None and pred is not None:
        return FieldScore(
            page_id, name, None, pred, status, SPURIOUS,
            f"nothing recorded on the slip, but {pred} was reported",
        )
    if gt is not None and pred is None:
        outcome = MISSED_ABSENT if status == ABSENT else MISSED_WITHHELD
        return FieldScore(page_id, name, gt, None, status, outcome, reason)
    if abs(float(pred) - float(gt)) <= tolerance:
        return FieldScore(page_id, name, gt, pred, status, CORRECT, "")
    return FieldScore(
        page_id, name, gt, pred, status, WRONG,
        f"reported {pred}, ground truth {gt}",
    )


def score_page(
    fixture: Fixture,
    result: ExtractionResult,
    field_names: tuple[str, ...],
    decimals: dict[str, int],
    tolerance: float,
) -> PageVitalsScore:
    vitals = (fixture.fields or {}).get("vitals") or {}
    page = PageVitalsScore(
        page_id=fixture.page_id,
        ground_truth_status=fixture.ground_truth_status,
        from_cache=result.from_cache,
        duration_s=result.duration_s,
        json_ok=result.json_ok,
        raw_text=result.raw_text,
        raw_text_field=result.raw_text_field,
        warnings=list(result.warnings),
    )
    missing_keys = [n for n in field_names if n not in vitals]
    if missing_keys:
        page.warnings.append(
            f"fixture has no ground-truth key(s) {missing_keys} under "
            f"fields.vitals; scored as not recorded"
        )
    for name in field_names:
        page.scores.append(
            score_field(
                fixture.page_id, name, vitals.get(name), result,
                decimals.get(name, 2), tolerance,
            )
        )
    return page


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@dataclass
class FieldSummary:
    name: str
    counts: dict[str, int]
    n_scored: int
    n_correct: int
    #: Fields where the slip actually records a value. The harder subset, and the
    #: one a clinic cares about: getting "no MAC recorded" right is easy.
    n_recorded: int
    n_recorded_correct: int

    @property
    def accuracy(self) -> float | None:
        return None if not self.n_scored else self.n_correct / self.n_scored

    @property
    def recorded_accuracy(self) -> float | None:
        return None if not self.n_recorded else self.n_recorded_correct / self.n_recorded

    @property
    def n_emitted_wrong(self) -> int:
        return sum(self.counts.get(o, 0) for o in EMITTED_WRONG_OUTCOMES)

    @property
    def n_withheld(self) -> int:
        return sum(self.counts.get(o, 0) for o in WITHHELD_OUTCOMES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.name,
            "n_scored": self.n_scored,
            "n_correct": self.n_correct,
            "exact_match_accuracy": self.accuracy,
            "n_recorded_on_slip": self.n_recorded,
            "n_recorded_correct": self.n_recorded_correct,
            "recorded_field_accuracy": self.recorded_accuracy,
            "outcomes": dict(self.counts),
            "n_emitted_wrong": self.n_emitted_wrong,
            "n_withheld": self.n_withheld,
        }


def summarise_fields(pages: list[PageVitalsScore],
                     field_names: tuple[str, ...]) -> list[FieldSummary]:
    out: list[FieldSummary] = []
    for name in field_names:
        scores = [s for p in pages for s in p.scores if s.name == name]
        counts = {o: sum(1 for s in scores if s.outcome == o) for o in OUTCOMES}
        recorded = [s for s in scores if s.ground_truth is not None]
        out.append(
            FieldSummary(
                name=name,
                counts=counts,
                n_scored=len(scores),
                n_correct=sum(1 for s in scores if s.correct),
                n_recorded=len(recorded),
                n_recorded_correct=sum(1 for s in recorded if s.correct),
            )
        )
    return out


def summarise_overall(pages: list[PageVitalsScore]) -> dict[str, Any]:
    all_scores = [s for p in pages for s in p.scores]
    counts = {o: sum(1 for s in all_scores if s.outcome == o) for o in OUTCOMES}
    recorded = [s for s in all_scores if s.ground_truth is not None]
    n = len(all_scores)
    n_emitted_wrong = sum(counts[o] for o in EMITTED_WRONG_OUTCOMES)
    n_withheld = sum(counts[o] for o in WITHHELD_OUTCOMES)
    return {
        "n_pages": len(pages),
        "n_fields_scored": n,
        "n_correct": sum(1 for s in all_scores if s.correct),
        "exact_match_accuracy": (sum(1 for s in all_scores if s.correct) / n) if n else None,
        "n_recorded_on_slip": len(recorded),
        "n_recorded_correct": sum(1 for s in recorded if s.correct),
        "recorded_field_accuracy": (
            sum(1 for s in recorded if s.correct) / len(recorded) if recorded else None
        ),
        "outcomes": counts,
        # The distinction spec section 8 cares about, under its honest name.
        "n_emitted_wrong": n_emitted_wrong,
        "emitted_wrong_rate": (n_emitted_wrong / n) if n else None,
        "n_withheld": n_withheld,
        "withheld_rate": (n_withheld / n) if n else None,
        "note": (
            "n_emitted_wrong counts fields where a value was produced and it is "
            "not the ground truth -- the failure a human reviewer gets no warning "
            "about. n_withheld counts fields where the truth was lost but nothing "
            "false was asserted, which is the behaviour spec section 1.3 requires "
            "when the system is not confident. This is NOT flag precision/recall: "
            "there is no flag state until phase 5."
        ),
        "pages_with_partial_ground_truth": [
            p.page_id for p in pages if p.partial_ground_truth
        ],
    }
