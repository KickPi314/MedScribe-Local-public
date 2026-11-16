"""Accuracy, CER and two-pass agreement for the Phase 4 regions (spec section 8).

Phase 3 could only score six numbers in one region, so ``vitals_metrics.py``
only had to compare floats. Phase 4 extracts free text, dates and two kinds of
list, which spec section 8 asks to be measured two different ways:

    "Metrics: per-field exact-match accuracy, character error rate on free-text
     fields, and separately flag precision/recall"

so this module implements the first two and says plainly, in
:data:`PENDING_REGION_METRICS`, that the third is still phase 5.

THE OUTCOME TAXONOMY IS REUSED, NOT REINVENTED. ``CORRECT / CORRECT_NULL /
WRONG / SPURIOUS / MISSED_ABSENT / MISSED_WITHHELD`` are imported from
``vitals_metrics`` so that a number from the vitals table and a number from the
diagnosis table mean the same thing, and so that the emitted-wrong vs withheld
split -- the distinction spec section 8 cares most about -- is computed once.

CER IS A SECOND MEASUREMENT, NOT A SOFTER ACCURACY. Every field still gets a
discrete outcome; CER is reported alongside it, and only for the pairs where
BOTH ground truth and prediction hold a value. Three deliberate consequences:

* a null-vs-null field is ``CORRECT_NULL``, not "CER 0.0 on two empty strings".
  Scoring it as a perfect character match would inflate the CER of a corpus
  where half the fields are legitimately blank.
* a withheld field is not "CER 1.0". Deleting the whole reference string is
  arithmetically an error rate of 1, but it describes a system that declined to
  answer, not one that transcribed badly. It is counted in the outcome table
  under MISSED_WITHHELD, where it belongs, and excluded from the CER mean --
  which is stated in the report next to the number.
* CER is computed on the same normalised strings the exact-match test uses (see
  ``[extract].text_compare_*`` in config/eval.toml), so the two metrics cannot
  disagree about what "the same string" means.

AGREEMENT IS REPORTED, NOT ACTED ON. :func:`summarise_agreement` counts how
often the two differently-worded passes produced the same value, and crosses
that with correctness to answer the question that actually matters for phase 5:
*when the two passes agree, how often are they both wrong?* That is the raw
material for flag precision/recall. It is NOT flag precision/recall -- there is
no flag state anywhere in this codebase yet -- and the report says so wherever
the number appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable, Sequence

from ..extract.config import (
    TYPE_COMPLAINT_LIST,
    TYPE_DATE,
    TYPE_NUMBER,
    TYPE_STRING_LIST,
    TYPE_TEXT,
)
from ..extract.parse import ABSENT
from ..extract.pipeline import ExtractionResult
from .vitals_metrics import (
    CORRECT,
    CORRECT_NULL,
    EMITTED_WRONG_OUTCOMES,
    MISSED_ABSENT,
    MISSED_WITHHELD,
    OUTCOMES,
    SPURIOUS,
    WITHHELD_OUTCOMES,
    WRONG,
    _OUTCOME_MARK,
)

#: Spec section 8 metrics that Phase 4 still cannot produce, and why.
PENDING_REGION_METRICS = {
    "flag_precision": (
        "needs the `flagged` field state and the validation rules of spec "
        "section 4 (phase 5). Phase 4 produces two of its inputs -- a per-field "
        "withholding reason and a two-pass agreement signal -- and neither is "
        "the metric."
    ),
    "flag_recall": "same as flag_precision (phase 5)",
    "false_alarm_rate": (
        "a flagged field that turned out to be correctly extracted -- undefined "
        "until fields can be flagged (phase 5)"
    ),
    "missed_error_rate": (
        "an `extracted` field that was wrong -- undefined until fields have "
        "states (phase 5). `n_emitted_wrong` below is the closest honest "
        "measurement available now and is deliberately not called this."
    ),
    "formulary_match_rate": (
        "spec section 4.2 fuzzy-matches every product name against a local drug "
        "list. The formulary is phase 5; Phase 4 transcribes product names and "
        "never normalises one to a known entry (spec section 10)."
    ),
}


# ---------------------------------------------------------------------------
# string distance
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Edit distance, iterative two-row DP.

    Written out rather than taken from rapidfuzz (which is already a dependency)
    on purpose: this is the definition the reported CER stands on, it is fifteen
    lines, and a metric whose exact semantics live in a third-party library's
    default arguments is a metric no one can check. rapidfuzz is for the phase 5
    formulary, where speed over a long drug list is the point.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance / max(1, len(reference)).

    Unbounded above (a hypothesis longer than the reference can exceed 1.0),
    which is correct and is left unclipped: clipping would hide a model that
    answered with a sentence where a two-word diagnosis was written.
    """
    return levenshtein(reference, hypothesis) / max(1, len(reference))


def normalise_text(s: Any, casefold: bool, collapse_ws: bool) -> str:
    """The comparison form of a transcribed string.

    Only ever used for COMPARISON. The stored value is always what the model
    said; nothing in the pipeline writes a normalised string back.
    """
    out = "" if s is None else str(s)
    if collapse_ws:
        out = " ".join(out.split())
    if casefold:
        out = out.casefold()
    return out


@dataclass(frozen=True)
class TextCompare:
    """How two transcriptions are compared. From ``[extract]`` in config/eval.toml."""

    casefold: bool
    collapse_whitespace: bool

    def norm(self, s: Any) -> str:
        return normalise_text(s, self.casefold, self.collapse_whitespace)

    def equal(self, a: Any, b: Any) -> bool:
        return self.norm(a) == self.norm(b)

    def cer(self, reference: Any, hypothesis: Any) -> float:
        return cer(self.norm(reference), self.norm(hypothesis))


# ---------------------------------------------------------------------------
# one field's score
# ---------------------------------------------------------------------------


@dataclass
class RegionFieldScore:
    page_id: str
    region: str
    name: str
    ground_truth: Any
    predicted: Any
    extract_status: str
    outcome: str
    reason: str = ""
    #: Character error rate, only where BOTH sides hold a value. None otherwise --
    #: see the module docstring for why a withheld field is not scored as CER 1.
    cer: float | None = None
    #: Length of the reference string, so per-field CER can be aggregated
    #: character-weighted as well as field-weighted.
    ref_chars: int | None = None
    #: Item-level counts for the two list fields.
    items_matched: int | None = None
    items_missed: int | None = None
    items_spurious: int | None = None
    #: Which prescription line crop this came from, where applicable.
    line_index: int | None = None
    #: Which pass produced it ("a" or "b").
    pass_id: str = "a"

    @property
    def correct(self) -> bool:
        return self.outcome in (CORRECT, CORRECT_NULL)

    @property
    def mark(self) -> str:
        return _OUTCOME_MARK[self.outcome]

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "region": self.region,
            "field": self.name,
            "line_index": self.line_index,
            "pass": self.pass_id,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "extract_status": self.extract_status,
            "outcome": self.outcome,
            "correct": self.correct,
            "cer": None if self.cer is None else round(self.cer, 4),
            "ref_chars": self.ref_chars,
            "items_matched": self.items_matched,
            "items_missed": self.items_missed,
            "items_spurious": self.items_spurious,
            "reason": self.reason,
        }


def _outcome_for(truth_present: bool, pred_present: bool, equal: bool,
                 status: str) -> str:
    if not truth_present and not pred_present:
        return CORRECT_NULL
    if not truth_present:
        return SPURIOUS
    if not pred_present:
        return MISSED_ABSENT if status == ABSENT else MISSED_WITHHELD
    return CORRECT if equal else WRONG


def score_number_field(page_id: str, region: str, name: str, truth: Any,
                       predicted: Any, status: str, reason: str,
                       decimals: int, tolerance: float) -> RegionFieldScore:
    """Numeric exact match, identical in behaviour to Phase 3's ``score_field``."""
    def rnd(v: Any) -> float | int | None:
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        r = round(float(v), decimals)
        return int(r) if decimals == 0 else r

    gt, pred = rnd(truth), rnd(predicted)
    if truth is not None and gt is None:
        return RegionFieldScore(page_id, region, name, None, None, status,
                                MISSED_WITHHELD,
                                f"ground truth {truth!r} is not numeric; fix the fixture")
    equal = gt is not None and pred is not None and abs(float(pred) - float(gt)) <= tolerance
    outcome = _outcome_for(gt is not None, pred is not None, equal, status)
    detail = reason
    if outcome == WRONG:
        detail = f"reported {pred}, ground truth {gt}"
    elif outcome == SPURIOUS:
        detail = f"nothing recorded on the slip, but {pred} was reported"
    return RegionFieldScore(page_id, region, name, gt, pred, status, outcome, detail)


def score_text_field(page_id: str, region: str, name: str, truth: Any,
                     predicted: Any, status: str, reason: str,
                     cmp: TextCompare, with_cer: bool = True) -> RegionFieldScore:
    """Free-text exact match plus CER.

    "Exact" is exact after the configured normalisation only -- see
    ``[extract].text_compare_casefold`` in config/eval.toml, which records the
    measured cost of each choice. Spelling is never normalised: the ground truth
    preserves the doctor's "CLINICALY" and "RUTINE", and a scorer that corrected
    them would be grading a different transcript than the one the human verified.
    """
    gt_present = truth is not None and str(truth).strip() != ""
    pred_present = predicted is not None and str(predicted).strip() != ""
    equal = gt_present and pred_present and cmp.equal(truth, predicted)
    outcome = _outcome_for(gt_present, pred_present, equal, status)
    score = RegionFieldScore(page_id, region, name, truth, predicted, status, outcome,
                             reason)
    if gt_present:
        score.ref_chars = len(cmp.norm(truth))
    if with_cer and gt_present and pred_present:
        score.cer = cmp.cer(truth, predicted)
        if outcome == WRONG:
            score.reason = f"CER {score.cer:.3f}: read {predicted!r}, truth {truth!r}"
    elif outcome == SPURIOUS:
        score.reason = f"nothing recorded on the slip, but {predicted!r} was reported"
    return score


def _match_items(ref: Sequence[str], hyp: Sequence[str], cmp: TextCompare,
                 max_cer: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """One-to-one item pairing between two lists. ``(pairs, unmatched_ref, unmatched_hyp)``.

    Exact (normalised) matches are taken first; the remainder is paired greedily
    by lowest CER, and only when that CER is at or below ``max_cer``. A pair over
    the threshold is NOT a match -- it is one missed item and one spurious item,
    which is the honest reading of "the model wrote something else here".

    This pairing is content-based and it is REPORTED, never used to fix up an
    order. Nothing downstream treats "hypothesis item 3 matched reference item 1"
    as evidence that the list was in the right order.
    """
    pairs: list[tuple[int, int, float]] = []
    ref_left = list(range(len(ref)))
    hyp_left = list(range(len(hyp)))
    for i in list(ref_left):
        for j in list(hyp_left):
            if cmp.equal(ref[i], hyp[j]):
                pairs.append((i, j, 0.0))
                ref_left.remove(i)
                hyp_left.remove(j)
                break
    while ref_left and hyp_left:
        best = min(
            ((i, j, cmp.cer(ref[i], hyp[j])) for i in ref_left for j in hyp_left),
            key=lambda t: t[2],
        )
        if best[2] > max_cer:
            break
        pairs.append(best)
        ref_left.remove(best[0])
        hyp_left.remove(best[1])
    return pairs, ref_left, hyp_left


def score_string_list_field(page_id: str, region: str, name: str, truth: Any,
                            predicted: Any, status: str, reason: str,
                            cmp: TextCompare, max_cer: float) -> RegionFieldScore:
    """A list of short strings, scored as an unordered collection.

    ORDER IS NOT SCORED, deliberately: the ground truth for investigations was
    itself flattened out of a two-line margin note by the transcriber, so its
    order carries no information the model could be graded against. The field is
    CORRECT only when every reference item is matched exactly and nothing extra
    was reported -- so an unordered comparison is not a lenient one.
    """
    gt_items = [str(x) for x in truth] if isinstance(truth, list) else []
    pred_items = [str(x) for x in predicted] if isinstance(predicted, list) else []
    gt_present = bool(gt_items)
    pred_present = bool(pred_items)

    pairs, missed, spurious = _match_items(gt_items, pred_items, cmp, max_cer)
    exact = sum(1 for _, _, c in pairs if c == 0.0)
    equal = gt_present and pred_present and not missed and not spurious and exact == len(gt_items)
    outcome = _outcome_for(gt_present, pred_present, equal, status)

    score = RegionFieldScore(page_id, region, name, gt_items or None,
                             pred_items or None, status, outcome, reason)
    score.items_matched = exact
    score.items_missed = len(missed)
    score.items_spurious = len(spurious)
    if gt_present:
        score.ref_chars = sum(len(cmp.norm(x)) for x in gt_items)
    if pairs and gt_present and pred_present:
        # Character error over the matched pairs only, weighted by reference
        # length. Unmatched items are counted as missed/spurious above rather
        # than folded into a CER, which would make "wrote nothing" and "wrote
        # the wrong test" look like the same kind of error.
        num = sum(cmp.cer(gt_items[i], pred_items[j]) * max(1, len(cmp.norm(gt_items[i])))
                  for i, j, _ in pairs)
        den = sum(max(1, len(cmp.norm(gt_items[i]))) for i, j, _ in pairs)
        score.cer = num / den
    if outcome != CORRECT and (missed or spurious):
        score.reason = (
            f"{exact}/{len(gt_items)} matched exactly; missed "
            f"{[gt_items[i] for i in missed]}; extra {[pred_items[j] for j in spurious]}"
        )
    return score


def _fmt_duration(d: tuple[Any, Any]) -> str:
    """A (number, unit) pair as the human reading the mismatch needs to see it.

    "no unit" is spelled out rather than shown as ``None`` because the whole
    point of the pair is that a number without a unit is a different reading
    from the same number in days.
    """
    number, unit = d
    if number is None:
        return "none"
    return f"{number} {unit}" if unit else f"{number} (no unit)"


def score_complaint_list_field(page_id: str, region: str, name: str, truth: Any,
                               predicted: Any, status: str, reason: str,
                               cmp: TextCompare, max_cer: float) -> RegionFieldScore:
    """Complaints: the same unordered comparison on ``text``, plus the duration.

    A matched pair whose durations differ is NOT counted as an exact match: the
    duration is part of the complaint, and a wrong duration is a wrong reading.

    THE DURATION IS THE PAIR (number, unit), COMPARED AS A PAIR. The slips count
    in days on two pages and in hours on a third, so "2" is not a duration until
    it says which. A complaint read with the right number and the wrong unit is
    off by a factor of 24 and is a wrong reading, not a near miss; scoring the
    number alone would report that as correct. The null unit is a distinct value
    here rather than a wildcard matching either one: one real slip circles a
    number with no unit glyph after it, the fixture records that as written
    (duration 1, unit null), and a model that answers "1 day" there has asserted
    something the slip does not say. That page is exactly the one whose unit is
    still to be confirmed with the doctor -- see the fixture's own notes -- and
    treating null as "matches anything" would hide it instead.
    """
    def texts(v: Any) -> tuple[list[str], list[Any]]:
        if not isinstance(v, list):
            return [], []
        out_t, out_d = [], []
        for item in v:
            if isinstance(item, dict):
                out_t.append(str(item.get("text", "")))
                out_d.append((item.get("duration"), item.get("duration_unit")))
            else:
                out_t.append(str(item))
                out_d.append((None, None))
        return out_t, out_d

    gt_t, gt_d = texts(truth)
    pr_t, pr_d = texts(predicted)
    gt_present, pred_present = bool(gt_t), bool(pr_t)
    pairs, missed, spurious = _match_items(gt_t, pr_t, cmp, max_cer)
    exact = sum(1 for i, j, c in pairs if c == 0.0 and gt_d[i] == pr_d[j])
    duration_mismatch = [
        (gt_t[i], gt_d[i], pr_d[j]) for i, j, c in pairs if c == 0.0 and gt_d[i] != pr_d[j]
    ]
    equal = (gt_present and pred_present and not missed and not spurious
             and exact == len(gt_t))
    outcome = _outcome_for(gt_present, pred_present, equal, status)

    score = RegionFieldScore(page_id, region, name, truth, predicted, status, outcome,
                             reason)
    score.items_matched = exact
    score.items_missed = len(missed)
    score.items_spurious = len(spurious)
    if gt_present:
        score.ref_chars = sum(len(cmp.norm(x)) for x in gt_t)
    if pairs and gt_present and pred_present:
        num = sum(cmp.cer(gt_t[i], pr_t[j]) * max(1, len(cmp.norm(gt_t[i])))
                  for i, j, _ in pairs)
        den = sum(max(1, len(cmp.norm(gt_t[i]))) for i, j, _ in pairs)
        score.cer = num / den
    bits = []
    if missed:
        bits.append(f"missed {[gt_t[i] for i in missed]}")
    if spurious:
        bits.append(f"extra {[pr_t[j] for j in spurious]}")
    if duration_mismatch:
        bits.append(
            "duration differs: " + "; ".join(
                f"{t!r} truth={_fmt_duration(a)} read={_fmt_duration(b)}"
                for t, a, b in duration_mismatch
            )
        )
    if bits:
        score.reason = f"{exact}/{len(gt_t)} matched exactly; " + "; ".join(bits)
    return score


def score_region_field(page_id: str, region: str, name: str, ftype: str, truth: Any,
                       result: ExtractionResult, cmp: TextCompare, max_cer: float,
                       decimals: int = 2, tolerance: float = 0.0,
                       pass_id: str = "a") -> RegionFieldScore:
    """Dispatch one field to the scorer its ``[fields.<name>].type`` calls for."""
    read = result.reads.get(name)
    status = read.status if read is not None else "missing_read"
    reason = read.reason if read is not None else "no extraction record for this field"
    predicted = None if read is None else read.value

    if ftype == TYPE_NUMBER:
        score = score_number_field(page_id, region, name, truth, predicted, status,
                                   reason, decimals, tolerance)
    elif ftype == TYPE_DATE:
        # A date is a structured field, not free text: 2025-04-27 and 2025-04-26
        # are two different days, not a one-character near miss, so no CER.
        score = score_text_field(page_id, region, name, truth, predicted, status,
                                 reason, cmp, with_cer=False)
    elif ftype == TYPE_TEXT:
        score = score_text_field(page_id, region, name, truth, predicted, status,
                                 reason, cmp)
    elif ftype == TYPE_STRING_LIST:
        score = score_string_list_field(page_id, region, name, truth, predicted,
                                        status, reason, cmp, max_cer)
    elif ftype == TYPE_COMPLAINT_LIST:
        score = score_complaint_list_field(page_id, region, name, truth, predicted,
                                           status, reason, cmp, max_cer)
    else:  # pragma: no cover - the extract config rejects unknown types
        raise ValueError(f"no scorer for field type {ftype!r}")
    score.line_index = result.line_index
    score.pass_id = pass_id
    return score


# ---------------------------------------------------------------------------
# prescriptions: alignment is a decision, so it is reported
# ---------------------------------------------------------------------------


@dataclass
class PrescriptionAlignment:
    """How one page's line crops line up with its ground-truth prescriptions.

    THE PROBLEM, measured on the real corpus. Phase 2 splits the prescription
    block by ink projection and writes N line crops; ground truth records M
    prescriptions. On the 7 real scans:

        page   line crops   ground-truth prescriptions
        20fd            5            5
        8a83            6            4
        a19dd           5            2
        ac9e            7            5
        b741            5            5
        e6a8            5            5
        fd12            6            4

    and even where the counts match the crops are not the prescriptions: on
    20fd, line_0 is the "for <name>:" heading with the tail of the diagnosis
    running through it, so a positional zip() would score every line against the
    wrong entry and report five wrong products on a page the model may have read
    correctly.

    THE DECISION. Two alignments are computed and both are reported, and neither
    is allowed to hide the other:

    * POSITIONAL (Nth crop <-> Nth entry) is attempted only when Phase 2 trusted
      the split AND the counts match. Otherwise the page is EXCLUDED from
      positional scoring, by name, with the reason -- never silently zipped to
      the shorter list.
    * CONTENT-ANCHORED pairs each crop's transcribed product name with the
      closest ground-truth product under a CER threshold, one-to-one. This is
      computed for every page and reported separately as the weaker measurement
      it is: it can tell you whether a product was read anywhere on the page, and
      it deliberately cannot tell you whether the dose ended up attached to the
      right drug -- the pairing was built from the product names.

    A crop the model reported as "not a medicine line" (all four fields null) is
    counted separately again, as neither a miss nor a spurious entry: it is the
    model correctly declining a heading crop.
    """

    page_id: str
    n_line_crops: int
    n_ground_truth: int
    split_trusted: bool
    positional_eligible: bool
    exclusion_reason: str = ""
    n_null_lines: int = 0
    n_matched: int = 0
    n_missed: int = 0
    n_spurious: int = 0
    pairs: list[tuple[int, int, float]] = dc_field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "n_line_crops": self.n_line_crops,
            "n_ground_truth_prescriptions": self.n_ground_truth,
            "phase2_split_trusted": self.split_trusted,
            "positional_scoring_eligible": self.positional_eligible,
            "exclusion_reason": self.exclusion_reason,
            "n_lines_reported_not_a_medicine": self.n_null_lines,
            "content_anchored": {
                "n_matched": self.n_matched,
                "n_ground_truth_unmatched": self.n_missed,
                "n_extracted_unmatched": self.n_spurious,
                "pairs_line_to_gt_index_cer": [
                    {"line_index": j, "gt_index": i, "product_cer": round(c, 4)}
                    for i, j, c in self.pairs
                ],
                "note": (
                    "content-anchored pairing is built from the PRODUCT NAME only. "
                    "It measures whether a product was read somewhere on the page; "
                    "it cannot measure whether a dose was attached to the right "
                    "drug, because the pairing already assumed the product."
                ),
            },
        }


def align_prescriptions(page_id: str, gt_entries: list[dict[str, Any]],
                        line_results: list[ExtractionResult], split_trusted: bool,
                        split_reason: str, cmp: TextCompare,
                        max_cer: float) -> PrescriptionAlignment:
    products_gt = [str(e.get("product") or "") for e in gt_entries]
    products_pred = [str(r.fields.get("product") or "") for r in line_results]
    null_lines = [i for i, p in enumerate(products_pred) if not p]
    # Lines the model declined are removed from the pairing candidates but are
    # counted and reported; they are not silently dropped.
    idx_map = [i for i, p in enumerate(products_pred) if p]
    pairs, missed, spurious = _match_items(
        products_gt, [products_pred[i] for i in idx_map], cmp, max_cer
    )

    eligible = split_trusted and len(line_results) == len(gt_entries) and bool(gt_entries)
    reason = ""
    if not split_trusted:
        reason = f"Phase 2 did not trust the line split ({split_reason})"
    elif not gt_entries:
        reason = "no prescriptions in the ground truth for this page"
    elif len(line_results) != len(gt_entries):
        reason = (
            f"{len(line_results)} line crop(s) vs {len(gt_entries)} ground-truth "
            f"prescription(s); a positional pairing would score at least one line "
            f"against the wrong entry"
        )

    return PrescriptionAlignment(
        page_id=page_id,
        n_line_crops=len(line_results),
        n_ground_truth=len(gt_entries),
        split_trusted=split_trusted,
        positional_eligible=eligible,
        exclusion_reason=reason,
        n_null_lines=len(null_lines),
        n_matched=len(pairs),
        n_missed=len(missed),
        n_spurious=len(spurious),
        pairs=[(i, idx_map[j], c) for i, j, c in pairs],
    )


def score_prescription_pairs(page_id: str, gt_entries: list[dict[str, Any]],
                             line_results: list[ExtractionResult],
                             alignment: PrescriptionAlignment, cmp: TextCompare,
                             field_names: Sequence[str],
                             pass_id: str = "a") -> list[RegionFieldScore]:
    """Score the four line fields over the CONTENT-ANCHORED pairs.

    Only the paired lines are scored here, and the pairing came from the product
    name, so ``product``'s own CER on these pairs is by construction the CER of
    the pairs that were close enough to pair -- reported, but read alongside
    ``n_ground_truth_unmatched``, which counts the products that were not read at
    all. dose / frequency / form are the interesting columns: they were not used
    to build the pairing.
    """
    by_line = {r.line_index: r for r in line_results}
    out: list[RegionFieldScore] = []
    for gt_i, line_i, _ in alignment.pairs:
        res = by_line.get(line_i)
        if res is None:  # pragma: no cover - pairs come from these results
            continue
        entry = gt_entries[gt_i]
        for name in field_names:
            score = score_text_field(
                page_id, "prescription_line", name, entry.get(name),
                res.fields.get(name),
                res.reads[name].status if name in res.reads else "missing_read",
                res.reads[name].reason if name in res.reads else "",
                cmp,
            )
            score.line_index = line_i
            score.pass_id = pass_id
            out.append(score)
    return out


def score_prescription_positional(page_id: str, gt_entries: list[dict[str, Any]],
                                  line_results: list[ExtractionResult],
                                  cmp: TextCompare, field_names: Sequence[str],
                                  pass_id: str = "a") -> list[RegionFieldScore]:
    """Nth crop against Nth ground-truth entry. Callers must check eligibility first."""
    out: list[RegionFieldScore] = []
    for i, (entry, res) in enumerate(zip(gt_entries, line_results)):
        for name in field_names:
            score = score_text_field(
                page_id, "prescription_line", name, entry.get(name),
                res.fields.get(name),
                res.reads[name].status if name in res.reads else "missing_read",
                res.reads[name].reason if name in res.reads else "",
                cmp,
            )
            score.line_index = res.line_index if res.line_index is not None else i
            score.pass_id = pass_id
            out.append(score)
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@dataclass
class FieldSummary:
    region: str
    name: str
    counts: dict[str, int]
    n_scored: int
    n_correct: int
    n_recorded: int
    n_recorded_correct: int
    cer_values: list[tuple[float, int]] = dc_field(default_factory=list)

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

    @property
    def mean_cer(self) -> float | None:
        """Unweighted mean over the fields where both sides held a value."""
        if not self.cer_values:
            return None
        return sum(c for c, _ in self.cer_values) / len(self.cer_values)

    @property
    def char_weighted_cer(self) -> float | None:
        """Total edit distance / total reference characters.

        Reported next to the unweighted mean because they answer different
        questions: the mean treats a two-character form abbreviation and a
        forty-character diagnosis as equally important, the weighted figure is
        the corpus-level character error rate.
        """
        if not self.cer_values:
            return None
        den = sum(n for _, n in self.cer_values)
        if den <= 0:
            return None
        return sum(c * n for c, n in self.cer_values) / den

    @property
    def n_cer_scored(self) -> int:
        return len(self.cer_values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
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
            "n_cer_scored": self.n_cer_scored,
            "mean_cer": self.mean_cer,
            "char_weighted_cer": self.char_weighted_cer,
            "cer_note": (
                "CER is computed only where ground truth AND prediction both hold "
                "a value. A withheld field is not scored as CER 1.0 -- it is "
                "counted under missed_withheld, because declining to answer is a "
                "different behaviour from transcribing badly."
            ),
        }


def summarise_field(region: str, name: str,
                    scores: Iterable[RegionFieldScore]) -> FieldSummary:
    ss = list(scores)
    counts = {o: sum(1 for s in ss if s.outcome == o) for o in OUTCOMES}
    recorded = [s for s in ss if s.ground_truth is not None]
    return FieldSummary(
        region=region,
        name=name,
        counts=counts,
        n_scored=len(ss),
        n_correct=sum(1 for s in ss if s.correct),
        n_recorded=len(recorded),
        n_recorded_correct=sum(1 for s in recorded if s.correct),
        cer_values=[(s.cer, s.ref_chars or 1) for s in ss if s.cer is not None],
    )


def summarise_scores(scores: Iterable[RegionFieldScore]) -> dict[str, Any]:
    ss = list(scores)
    counts = {o: sum(1 for s in ss if s.outcome == o) for o in OUTCOMES}
    n = len(ss)
    recorded = [s for s in ss if s.ground_truth is not None]
    n_wrong = sum(counts[o] for o in EMITTED_WRONG_OUTCOMES)
    n_withheld = sum(counts[o] for o in WITHHELD_OUTCOMES)
    cers = [(s.cer, s.ref_chars or 1) for s in ss if s.cer is not None]
    return {
        "n_fields_scored": n,
        "n_correct": sum(1 for s in ss if s.correct),
        "exact_match_accuracy": (sum(1 for s in ss if s.correct) / n) if n else None,
        "n_recorded_on_slip": len(recorded),
        "n_recorded_correct": sum(1 for s in recorded if s.correct),
        "recorded_field_accuracy": (
            sum(1 for s in recorded if s.correct) / len(recorded) if recorded else None
        ),
        "outcomes": counts,
        "n_emitted_wrong": n_wrong,
        "emitted_wrong_rate": (n_wrong / n) if n else None,
        "n_withheld": n_withheld,
        "withheld_rate": (n_withheld / n) if n else None,
        "n_cer_scored": len(cers),
        "mean_cer": (sum(c for c, _ in cers) / len(cers)) if cers else None,
        "char_weighted_cer": (
            sum(c * w for c, w in cers) / sum(w for _, w in cers) if cers else None
        ),
        "note": (
            "n_emitted_wrong counts fields where a value was produced and it is "
            "not the ground truth -- the failure a human reviewer gets no warning "
            "about. n_withheld counts fields where the truth was lost but nothing "
            "false was asserted. This is NOT flag precision/recall: there is no "
            "flag state until phase 5."
        ),
    }


# ---------------------------------------------------------------------------
# two-pass agreement
# ---------------------------------------------------------------------------


@dataclass
class AgreementSummary:
    region: str
    name: str
    n: int = 0
    n_agree: int = 0
    n_both_read: int = 0
    n_agree_both_read: int = 0
    n_both_null: int = 0
    n_status_agree: int = 0
    #: Crossed with correctness of PASS A, which is the pass Phase 3 measured.
    n_agree_and_correct: int = 0
    n_agree_and_incorrect: int = 0
    n_disagree_and_correct: int = 0
    n_disagree_and_incorrect: int = 0

    @property
    def rate(self) -> float | None:
        return None if not self.n else self.n_agree / self.n

    @property
    def rate_both_read(self) -> float | None:
        """Agreement restricted to fields where both passes produced a value.

        The stricter and more informative number: two passes that both read
        nothing agree trivially, and on this corpus a lot of fields are
        legitimately blank.
        """
        return None if not self.n_both_read else self.n_agree_both_read / self.n_both_read

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "field": self.name,
            "n_compared": self.n,
            "n_agree": self.n_agree,
            "agreement_rate": self.rate,
            "n_both_passes_read_a_value": self.n_both_read,
            "n_agree_both_read": self.n_agree_both_read,
            "agreement_rate_both_read": self.rate_both_read,
            "n_both_null": self.n_both_null,
            "n_same_status": self.n_status_agree,
            "crossed_with_pass_a_correctness": {
                "agree_and_correct": self.n_agree_and_correct,
                "agree_and_incorrect": self.n_agree_and_incorrect,
                "disagree_and_correct": self.n_disagree_and_correct,
                "disagree_and_incorrect": self.n_disagree_and_incorrect,
            },
        }


def summarise_agreement(region: str, name: str, comparisons: Iterable[Any],
                        correctness: dict[tuple[str, int | None], bool] | None = None
                        ) -> AgreementSummary:
    """Aggregate :class:`src.extract.agreement.TwoPassField` records for one field.

    ``correctness`` maps ``(page_id, line_index) -> was pass A correct`` and is
    optional: the agreement rate stands on its own, and crossing it with
    correctness is a separate question that only makes sense where ground truth
    exists.
    """
    out = AgreementSummary(region=region, name=name)
    for page_id, line_index, cmpf in comparisons:
        out.n += 1
        out.n_agree += int(cmpf.agree)
        out.n_both_read += int(cmpf.both_read)
        out.n_agree_both_read += int(cmpf.agree and cmpf.both_read)
        out.n_both_null += int(cmpf.both_null)
        out.n_status_agree += int(cmpf.status_agree)
        if correctness is None:
            continue
        was_correct = correctness.get((page_id, line_index))
        if was_correct is None:
            continue
        if cmpf.agree:
            out.n_agree_and_correct += int(was_correct)
            out.n_agree_and_incorrect += int(not was_correct)
        else:
            out.n_disagree_and_correct += int(was_correct)
            out.n_disagree_and_incorrect += int(not was_correct)
    return out


__all__ = [
    "AgreementSummary",
    "FieldSummary",
    "PENDING_REGION_METRICS",
    "PrescriptionAlignment",
    "RegionFieldScore",
    "TextCompare",
    "align_prescriptions",
    "cer",
    "levenshtein",
    "normalise_text",
    "score_prescription_pairs",
    "score_prescription_positional",
    "score_region_field",
    "summarise_agreement",
    "summarise_field",
    "summarise_scores",
]
