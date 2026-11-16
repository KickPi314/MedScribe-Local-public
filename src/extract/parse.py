"""Turning a model reply into field values, or into nulls (Phases 3-4).

This module is where spec section 1.3 -- "never guess", "the single most
important behavioural requirement in this document" -- is actually enforced.
The prompt in config/extract.toml asks the model to withhold values it cannot
read; nothing here believes that it did. Every value is re-derived from the raw
reply and must survive four checks:

1. the reply as a whole must be valid JSON (optionally after a brace salvage on
   the envelope only, which cannot alter a digit);
2. the key must be present, under its own name or a configured alias;
3. the value must reduce to a real number after configured unit tokens are
   stripped;
4. the number must fall inside the field's configured plausibility range
   (spec section 4.3, "vitals must parse as numbers within plausible paediatric
   ranges").

A value that fails any of them is stored as ``None`` together with the reason.
It is never repaired, rounded into range, or replaced with the nearest plausible
number. The raw reply is carried alongside the parsed result as *evidence*, for
debugging -- never as a fallback value.

There is one distinction the caller needs and that this module preserves: a
value can be null because the model *asserted the slip has no value there*
(``ABSENT`` -- the correct answer on 3 of the 7 real pages, which record no MAC
or Abd, and on the 5 pages that requested no investigations), or because
something failed (``UNPARSEABLE``, ``OUT_OF_RANGE``, ``MISSING_KEY``,
``NO_JSON``). Both store ``None``, because a withheld value is withheld; but the
eval harness scores them differently, and Phase 5 will turn the failures -- and
only the failures -- into the ``flagged`` field state.

PHASE 4 GENERALISATION. Phase 3 only had to read single numbers out of one flat
object. Phase 4's regions also produce free text (patient name, diagnosis,
clinical notes, and the four prescription-line fields), a date, and two kinds of
JSON array (investigations as strings, complaints as objects). Those are new
*coercions*, not a new parser: the JSON envelope check, the alias-tolerant key
lookup, the ``[parse].null_tokens`` handling and the "null with a recorded
reason" contract are shared by every type, and :func:`read_field` still has one
signature and one dispatch point. What changes per type is only the question
"what does a plausible value look like here":

    number         reduces to a real number inside [min, max]
    text           a JSON string, non-empty after whitespace collapse, within
                   max_chars (a runaway generation is a failed read, not a
                   long answer)
    date           three digit groups that reformat to ISO 8601 -- a SYNTACTIC
                   rewrite of the digits the model reported, never a
                   calculation, never a two-digit year expanded to a century
    string_list    a JSON array of strings, each valid as text
    complaint_list a JSON array of {text, duration, duration_unit} objects, the
                   unit being one of the canonical units named in
                   [fields.complaints.duration_units] -- the slips write both
                   days and hours, so a number alone is not a duration

LISTS ARE ALL-OR-NOTHING, deliberately. If one item of a list fails validation
the WHOLE field is withheld, rather than emitting the items that parsed. A
complaints list is one field in the fixture and it is read as a complete
statement of what the patient presented with; emitting three of four complaints
would assert a complete list that is not complete, which is the same class of
error as inventing a value. The reason names the item that failed, so the human
knows where to look.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..preprocess.config import Config
from .config import (
    TYPE_COMPLAINT_LIST,
    TYPE_DATE,
    TYPE_NUMBER,
    TYPE_STRING_LIST,
    TYPE_TEXT,
    VITALS_FIELDS,
    field_spec,
)

# Why a field ended up null. Recorded per field, per page.
OK = "ok"                          # a number that passed every check
ABSENT = "absent"                  # model says there is no value on the slip
MISSING_KEY = "missing_key"        # model omitted the key entirely
UNPARSEABLE = "unparseable"        # present, but not reducible to a number
OUT_OF_RANGE = "out_of_range"      # a number, but outside the plausible range
NO_JSON = "no_json"                # the whole reply was not usable JSON

# Reasons that mean "extraction failed here", as opposed to "there is genuinely
# nothing here". Phase 5 turns exactly this set into a `flagged` field state.
FAILURE_STATUSES = (MISSING_KEY, UNPARSEABLE, OUT_OF_RANGE, NO_JSON)

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")


@dataclass
class FieldRead:
    """One field's outcome. ``value`` is ``None`` unless ``status == OK``.

    ``value`` is a number for a number field, a string for text and date fields,
    and a list for the two list types -- always the canonical form that will be
    compared with ground truth, never the raw reply.
    """

    name: str
    value: Any = None
    status: str = MISSING_KEY
    raw_value: Any = None          # exactly what the model put under the key
    source_key: str | None = None  # the key it was found under (may be an alias)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def failed(self) -> bool:
        return self.status in FAILURE_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "raw_value": self.raw_value,
            "source_key": self.source_key,
            "reason": self.reason,
        }


@dataclass
class ParsedReply:
    fields: dict[str, FieldRead] = field(default_factory=dict)
    json_ok: bool = False
    json_salvaged: bool = False
    reason: str = ""
    #: The field names this reply was parsed for, in the region's declared
    #: order. Kept so :meth:`values` stays ordered without the module having to
    #: know which region it just parsed.
    field_order: tuple[str, ...] = VITALS_FIELDS

    def values(self) -> dict[str, Any]:
        """The parsed result in the shape of the matching ``Fixture.fields`` slice.

        Same keys, same order, so the eval harness compares key-by-key with no
        translation layer between extraction and ground truth.
        """
        return {n: self.fields[n].value for n in self.field_order if n in self.fields}


# ---------------------------------------------------------------------------
# JSON envelope
# ---------------------------------------------------------------------------


def extract_json_object(text: str, salvage_braces: bool) -> tuple[dict[str, Any] | None, bool, str]:
    """``(object, salvaged, reason)``.

    The salvage retries on the substring from the first ``{`` to the last ``}``,
    which recovers a model that wrapped its object in a code fence or a sentence.
    It is a repair of the *envelope*, not of a value: it cannot change a digit,
    and if the salvaged substring is still not an object every field ends up null
    anyway.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None, False, "model returned no text at all"
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, False, ""
        reason = f"reply parsed as {type(obj).__name__}, not a JSON object"
    except json.JSONDecodeError as exc:
        reason = f"reply is not valid JSON ({exc.msg} at pos {exc.pos})"

    if not salvage_braces:
        return None, False, reason
    i, j = stripped.find("{"), stripped.rfind("}")
    if i == -1 or j <= i:
        return None, False, f"{reason}; no {{...}} substring to salvage either"
    try:
        obj = json.loads(stripped[i : j + 1])
    except json.JSONDecodeError as exc:
        return None, False, f"{reason}; brace salvage also failed ({exc.msg})"
    if not isinstance(obj, dict):
        return None, False, f"{reason}; brace salvage did not yield an object"
    return obj, True, ""


# ---------------------------------------------------------------------------
# one value
# ---------------------------------------------------------------------------


def _lookup(obj: dict[str, Any], name: str, aliases: list[str]) -> tuple[bool, Any, str | None]:
    """Find a field in the model's object, by name or configured alias.

    Matching is case-insensitive and ignores spaces/underscores/hyphens, because
    a key's *spelling* is not a transcribed value -- accepting ``"Wt (kg)"`` for
    ``wt_kg`` invents nothing. It only stops a correctly-read number being thrown
    away because the model renamed its own key.
    """

    def norm(s: str) -> str:
        return re.sub(r"[\s_\-()]+", "", str(s)).lower()

    wanted = {norm(name): name}
    for a in aliases:
        wanted.setdefault(norm(a), name)
    for key, value in obj.items():
        if norm(key) in wanted:
            return True, value, str(key)
    return False, None, None


def _null_tokens(cfg: Config) -> set[str]:
    """Reply strings that mean "there is no value here", from ``[parse]``.

    One definition, shared by every field type, so a model that answers
    ``"not visible"`` is treated identically whether the field is a weight, a
    diagnosis or a list of investigations.
    """
    return {str(t).strip().lower() for t in cfg.get("parse", "null_tokens")}


def _to_number(raw: Any, cfg: Config) -> tuple[float | None, str, str]:
    """``(number, status, reason)`` for one raw value, before the range check."""
    null_tokens = _null_tokens(cfg)
    strip_tokens = [str(t).lower() for t in cfg.get("parse", "strip_tokens")]
    strip_thousands = bool(cfg.get("parse", "strip_thousands_separator"))

    if raw is None:
        return None, ABSENT, "model reported null"
    if isinstance(raw, bool):
        # JSON true/false is never a measurement. Coercing it to 1/0 would be
        # inventing a vital sign out of a type error.
        return None, UNPARSEABLE, f"value is a boolean ({raw!r}), not a number"
    if isinstance(raw, (int, float)):
        return float(raw), OK, ""
    if isinstance(raw, (list, dict)):
        return None, UNPARSEABLE, f"value is a {type(raw).__name__}, not a number"

    s = str(raw).strip()
    if s.lower() in null_tokens:
        return None, ABSENT, f"model reported {raw!r}, a null token"

    # Strip configured unit text from either end, repeatedly (so "39.5 cm." and
    # "100 %" both reduce). Only non-numeric characters the prompt asked the
    # model not to emit are removed; anything left that is not a number stays a
    # failure.
    s = s.rstrip(".").strip()
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for tok in strip_tokens:
            if not tok:
                continue
            if low.endswith(tok) and len(s) > len(tok):
                s = s[: len(s) - len(tok)].strip().rstrip(".").strip()
                changed = True
                break
            if low.startswith(tok) and len(s) > len(tok):
                s = s[len(tok):].strip()
                changed = True
                break
    if strip_thousands:
        s = s.replace(",", "")
    s = s.strip()
    if not s:
        return None, UNPARSEABLE, f"nothing but unit text in {raw!r}"
    if s.lower() in null_tokens:
        return None, ABSENT, f"model reported {raw!r}, a null token"
    if not _NUMBER_RE.match(s):
        # Deliberately strict. A permissive parse ("97.9-98.1" -> 97.9,
        # "1 4 7" -> 147) would be picking one of several readings the model
        # offered, which is guessing.
        return None, UNPARSEABLE, f"{raw!r} does not reduce to a single number (got {s!r})"
    try:
        return float(s), OK, ""
    except ValueError:
        return None, UNPARSEABLE, f"{raw!r} does not parse as a number"


def _to_text(raw: Any, cfg: Config, max_chars: int,
             label: str) -> tuple[str | None, str, str]:
    """``(text, status, reason)`` for one raw value of a free-text field.

    The only editing is whitespace: runs of spaces, tabs and newlines collapse to
    one space and the ends are trimmed. That is the text equivalent of the unit
    stripping ``_to_number`` does -- it cannot change a character the model read,
    it only removes layout the model added. Nothing else is normalised: casing,
    spelling and punctuation are transcription and are preserved exactly, which
    matters because the ground truth itself preserves the doctor's spelling
    ("CLINICALY", "RUTINE").

    A value longer than ``max_chars`` is a failed read, not a long answer: at
    that point the model has stopped transcribing a line and started narrating,
    and the words in it are not on the slip.
    """
    if raw is None:
        return None, ABSENT, "model reported null"
    if isinstance(raw, bool):
        return None, UNPARSEABLE, f"value is a boolean ({raw!r}), not text"
    if isinstance(raw, (list, dict)):
        return None, UNPARSEABLE, f"value is a {type(raw).__name__}, not a string"
    if isinstance(raw, (int, float)):
        # A number where text was asked for is not necessarily wrong (a product
        # can be "P-250"), but a bare int for a name is a shape error worth
        # seeing rather than silently stringifying.
        return None, UNPARSEABLE, f"value is a {type(raw).__name__}, not a string"

    s = re.sub(r"\s+", " ", str(raw)).strip()
    if s.lower() in _null_tokens(cfg):
        return None, ABSENT, f"model reported {raw!r}, a null token"
    if not s:
        return None, ABSENT, "model reported an empty string"
    if len(s) > max_chars:
        return None, OUT_OF_RANGE, (
            f"{label} came back {len(s)} characters long, over the {max_chars}-"
            f"character ceiling; that is a runaway generation, not a transcription. "
            f"Withheld in full rather than truncated -- a truncated value would look "
            f"like a read field"
        )
    return s, OK, ""


def _to_date(raw: Any, cfg: Config, spec: dict[str, Any]) -> tuple[str | None, str, str]:
    """``(iso_date, status, reason)`` from the date as the doctor wrote it.

    The model is asked to report the date EXACTLY as written (see
    [prompts.date_v1] and [prompts.date_v2]); this turns "27/4/2025" into
    "2025-04-27".

    WHY THAT IS NOT A "CONVERSION" of the kind spec section 1.3 forbids: no
    information is added or inferred. The same three digit groups come out in a
    different order with zero padding, which is a syntactic rewrite of the
    characters the model reported -- unlike converting 37.2 C to 99.0 F, which
    invents a value the slip does not contain. Asking the model to do the
    reordering instead was rejected: it makes a 4B VLM do arithmetic on the one
    field where a silent day/month swap is unrecoverable, and it would hide the
    written form, which is the evidence.

    A TWO-DIGIT YEAR IS REFUSED, not expanded. "19/6/26" could be 1926 or 2026
    and choosing is guessing a century.
    """
    if raw is None:
        return None, ABSENT, "model reported null"
    if isinstance(raw, bool):
        return None, UNPARSEABLE, f"value is a boolean ({raw!r}), not a date"
    if isinstance(raw, (list, dict)):
        return None, UNPARSEABLE, f"value is a {type(raw).__name__}, not a date"

    s = re.sub(r"\s+", " ", str(raw)).strip()
    if s.lower() in _null_tokens(cfg) or not s:
        return None, ABSENT, f"model reported {raw!r}, a null token"

    groups = [g for g in re.split(r"[^0-9]+", s) if g]
    if len(groups) != 3:
        return None, UNPARSEABLE, (
            f"{s!r} does not split into exactly three groups of digits (got "
            f"{len(groups)}); a date written in words or partly illegible is "
            f"withheld rather than reconstructed"
        )
    order = str(cfg.get("parse", "date_order")).lower()
    if len(groups[0]) == 4:
        y, m, d = groups                      # already ISO-ordered
    elif len(groups[2]) == 4:
        if order == "dmy":
            d, m, y = groups
        elif order == "mdy":
            m, d, y = groups
        else:
            return None, UNPARSEABLE, f"[parse].date_order {order!r} is not dmy or mdy"
    else:
        return None, UNPARSEABLE, (
            f"{s!r} has no four-digit year. A two-digit year is NOT expanded: "
            f"'26' could be 1926 or 2026 and choosing one would be inventing a "
            f"century (spec section 1.3)"
        )
    try:
        yi, mi, di = int(y), int(m), int(d)
    except ValueError:  # pragma: no cover - the regex split guarantees digits
        return None, UNPARSEABLE, f"{s!r} does not parse as three integers"

    lo, hi = int(spec["min_year"]), int(spec["max_year"])
    if not (lo <= yi <= hi):
        return None, OUT_OF_RANGE, (
            f"year {yi} is outside the plausible range [{lo}, {hi}]; withheld"
        )
    if not (1 <= mi <= 12) or not (1 <= di <= 31):
        # No day/month swap to rescue it. If the groups are in the other order
        # the reading is ambiguous, and picking the arrangement that happens to
        # be valid is exactly the plausible-looking guess spec section 1.3 bans.
        return None, OUT_OF_RANGE, (
            f"{s!r} read as day={di} month={mi} under [parse].date_order="
            f"{order!r} is not a real date; withheld rather than reordered into "
            f"one that is"
        )
    return f"{yi:04d}-{mi:02d}-{di:02d}", OK, ""


def _to_string_list(raw: Any, cfg: Config,
                    spec: dict[str, Any]) -> tuple[list[str] | None, str, str]:
    """``(items, status, reason)`` for a JSON array of short strings.

    An empty array is ``ABSENT``: "nothing was requested on this slip" is the
    right answer on 5 of the 7 real pages and is a success, not a failed read.
    """
    if raw is None:
        return None, ABSENT, "model reported null"
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() in _null_tokens(cfg) or not s:
            return None, ABSENT, f"model reported {raw!r}, a null token"
        # A comma-joined string is NOT split into a list here. Splitting would be
        # this module deciding where one investigation ends and the next begins,
        # which is a reading decision, not a parse.
        return None, UNPARSEABLE, (
            f"{raw!r} is a string, not a JSON array. It is not split on commas: "
            f"deciding where one item ends is a reading decision, not a parse"
        )
    if not isinstance(raw, list):
        return None, UNPARSEABLE, f"value is a {type(raw).__name__}, not a JSON array"

    max_items = int(spec["max_items"])
    if len(raw) > max_items:
        return None, OUT_OF_RANGE, (
            f"{len(raw)} items returned, over the {max_items}-item ceiling; the "
            f"whole list is withheld"
        )
    out: list[str] = []
    for i, item in enumerate(raw):
        text, status, reason = _to_text(
            item, cfg, int(spec["max_item_chars"]), f"{spec['label']} item {i}"
        )
        if status == ABSENT:
            # A null entry inside the array is the model's own placeholder, not
            # an item. Dropping it removes a marker, not a transcribed value.
            continue
        if status != OK:
            return None, UNPARSEABLE, (
                f"item {i} ({item!r}) is not usable ({reason}); the WHOLE list is "
                f"withheld -- emitting the rest would assert a complete list that "
                f"is not complete"
            )
        out.append(str(text))
    if not out:
        return [], ABSENT, "model returned an empty list: nothing requested on this slip"
    return out, OK, ""


def _duration_units(spec: dict[str, Any]) -> dict[str, str]:
    """Accepted spelling -> canonical unit, from ``[fields.<name>.duration_units]``.

    The canonical form (the table key) is what ground truth records; the listed
    spellings are what the doctor's hand and the model actually produce ("d",
    "dy", "HRS"). Mapping one onto the other is the same move ``_lookup`` makes
    for key names: a unit's *spelling* is not a transcribed quantity, so
    accepting "hrs" for "hours" invents nothing. A spelling that is in no list is
    not mapped to anything -- see :func:`_to_complaint_list`.
    """
    out: dict[str, str] = {}
    for canon, spellings in spec["duration_units"].items():
        out[str(canon).strip().lower()] = str(canon)
        for s in spellings:
            out.setdefault(str(s).strip().lower(), str(canon))
    return out


def _to_complaint_list(raw: Any, cfg: Config,
                       spec: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str, str]:
    """``(items, status, reason)`` for ``[{"text", "duration", "duration_unit"}]``.

    Shape matches ``Fixture.fields["complaints"]`` exactly, so the eval harness
    compares entries without a translation layer.

    THE UNIT IS NOT ASSUMED. The slips do not all count in days: one records
    "2 HRS" against two complaints, so a bare number carries no meaning on its
    own and "2" read as 2 days would be a fabricated timescale. A number whose
    unit is written in a spelling this config does not recognise therefore
    withholds the whole list rather than defaulting to days. A number with NO
    unit written next to it is a real reading and is kept with a null unit --
    that is the state of one real slip, where the circled digit has no glyph
    after it -- and it is the eval harness's job to score that as different from
    the same number in days.
    """
    if raw is None:
        return None, ABSENT, "model reported null"
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() in _null_tokens(cfg) or not s:
            return None, ABSENT, f"model reported {raw!r}, a null token"
        return None, UNPARSEABLE, f"{raw!r} is a string, not a JSON array of objects"
    if not isinstance(raw, list):
        return None, UNPARSEABLE, f"value is a {type(raw).__name__}, not a JSON array"

    max_items = int(spec["max_items"])
    if len(raw) > max_items:
        return None, OUT_OF_RANGE, (
            f"{len(raw)} complaints returned, over the {max_items}-item ceiling; "
            f"the whole list is withheld"
        )
    dmin, dmax = int(spec["min_duration"]), int(spec["max_duration"])
    units = _duration_units(spec)
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            # A bare string where an object was asked for is still a complaint
            # with no duration -- the text is transcribed, the missing duration
            # stays missing. Nothing is invented by reading it this way.
            item = {"text": item, "duration": None, "duration_unit": None}
        if not isinstance(item, dict):
            return None, UNPARSEABLE, (
                f"complaint {i} is a {type(item).__name__}, not an object; the whole "
                f"list is withheld"
            )
        found, raw_text, _ = _lookup(item, "text", ["complaint", "symptom", "name"])
        text, status, reason = _to_text(
            raw_text if found else None, cfg, int(spec["max_item_chars"]),
            f"complaint {i} text",
        )
        if status != OK:
            return None, UNPARSEABLE, (
                f"complaint {i} ({item!r}) has no usable text ({reason}); the WHOLE "
                f"list is withheld -- emitting the rest would assert a complete list "
                f"that is not complete"
            )
        found_d, raw_d, _ = _lookup(
            item, "duration", ["duration_value", "duration_number", "how_long"]
        )
        duration: int | None = None
        if found_d:
            number, dstatus, dreason = _to_number(raw_d, cfg)
            if dstatus == OK:
                if not (dmin <= float(number) <= dmax):
                    return None, OUT_OF_RANGE, (
                        f"complaint {i} duration {number:g} is outside the "
                        f"plausible range [{dmin}, {dmax}]; the whole list is withheld"
                    )
                if float(number) != int(number):
                    return None, UNPARSEABLE, (
                        f"complaint {i} duration {number:g} is not a whole number; "
                        f"not rounded -- the whole list is withheld"
                    )
                duration = int(number)
            elif dstatus != ABSENT:
                return None, UNPARSEABLE, (
                    f"complaint {i} duration {raw_d!r} is not usable ({dreason}); the "
                    f"whole list is withheld"
                )
        unit: str | None = None
        found_u, raw_u, _ = _lookup(item, "duration_unit", ["unit", "units", "period"])
        if duration is not None and found_u:
            # Only consulted when a number was read: a unit on its own qualifies
            # nothing, so it is dropped the way _to_string_list drops a null entry
            # -- a marker, not a transcribed value -- rather than failing a list
            # whose complaint texts are perfectly readable.
            utext, ustatus, ureason = _to_text(
                raw_u, cfg, int(spec["max_item_chars"]), f"complaint {i} duration unit"
            )
            if ustatus == OK:
                unit = units.get(str(utext).rstrip(".").strip().lower())
                if unit is None:
                    return None, UNPARSEABLE, (
                        f"complaint {i} duration unit {raw_u!r} is not one of "
                        f"{sorted(set(units.values()))} (or a configured spelling of "
                        f"one); the whole list is withheld rather than the number "
                        f"being kept as a guessed number of days"
                    )
            elif ustatus != ABSENT:
                return None, UNPARSEABLE, (
                    f"complaint {i} duration unit {raw_u!r} is not usable "
                    f"({ureason}); the whole list is withheld"
                )
        out.append({"text": text, "duration": duration, "duration_unit": unit})
    if not out:
        return [], ABSENT, "model returned an empty list: no complaints on this slip"
    return out, OK, ""


def read_field(name: str, obj: dict[str, Any] | None, cfg: Config,
               json_reason: str = "") -> FieldRead:
    """One field, from the model's object to a value or a recorded null.

    The shared half -- envelope, alias-tolerant key lookup, null tokens -- is the
    same for every field type; only the "what does a plausible value look like"
    half is dispatched on ``[fields.<name>].type``.
    """
    spec = field_spec(cfg, name)
    if obj is None:
        return FieldRead(name=name, status=NO_JSON,
                         reason=json_reason or "reply was not usable JSON")

    aliases = [str(a) for a in spec.get("aliases", [])]
    found, raw, key = _lookup(obj, name, aliases)
    if not found:
        return FieldRead(
            name=name,
            status=MISSING_KEY,
            reason=f"model omitted key {name!r} (and aliases {aliases}); "
                   f"keys returned: {sorted(map(str, obj))}",
        )

    ftype = str(spec.get("type", TYPE_NUMBER))
    value: Any
    if ftype == TYPE_NUMBER:
        number, status, reason = _to_number(raw, cfg)
        if status == OK:
            lo, hi = float(spec["min"]), float(spec["max"])
            if not (lo <= number <= hi):
                # Withheld, not corrected. Spec section 4.4: "Flag only -- never
                # correct, never suggest a corrected value." Rounding 100.1 down
                # to a plausible 100 would be exactly the substitution spec
                # section 1.3 forbids.
                status, reason = OUT_OF_RANGE, (
                    f"{number:g} is outside the plausible range [{lo:g}, {hi:g}] for "
                    f"{spec['label']}; value withheld, not corrected"
                )
                value = None
            else:
                decimals = int(spec["decimals"])
                value = round(number, decimals)
                if decimals == 0:
                    value = int(value)
        else:
            value = None
    elif ftype == TYPE_TEXT:
        value, status, reason = _to_text(raw, cfg, int(spec["max_chars"]),
                                         str(spec["label"]))
    elif ftype == TYPE_DATE:
        value, status, reason = _to_date(raw, cfg, spec)
    elif ftype == TYPE_STRING_LIST:
        value, status, reason = _to_string_list(raw, cfg, spec)
    elif ftype == TYPE_COMPLAINT_LIST:
        value, status, reason = _to_complaint_list(raw, cfg, spec)
    else:  # pragma: no cover - load_extract_config rejects unknown types
        value, status, reason = None, UNPARSEABLE, f"unknown field type {ftype!r}"

    if status != OK:
        # An empty list is the model asserting "nothing here", which is a real
        # answer and is stored as such -- as [] rather than None, so the eval
        # harness can tell "reported none" from "read nothing".
        stored = value if (status == ABSENT and isinstance(value, list)) else None
        return FieldRead(name=name, value=stored, status=status, raw_value=raw,
                         source_key=key, reason=reason)
    return FieldRead(name=name, value=value, status=OK, raw_value=raw, source_key=key)


# ---------------------------------------------------------------------------
# the whole reply
# ---------------------------------------------------------------------------


def parse_reply(text: str, cfg: Config, field_names: tuple[str, ...]) -> ParsedReply:
    """Parse one model reply into the named fields.

    Never raises on bad model output: a hostile or empty reply produces one null
    per field, each with a recorded reason. Raising would be wrong here -- an
    unreadable page is a normal outcome that the eval harness has to be able to
    score, not an error in the program.
    """
    obj, salvaged, reason = extract_json_object(
        text, bool(cfg.get("parse", "salvage_braces"))
    )
    out = ParsedReply(json_ok=obj is not None, json_salvaged=salvaged, reason=reason,
                      field_order=tuple(field_names))
    for name in field_names:
        out.fields[name] = read_field(name, obj, cfg, json_reason=reason)
    return out


def parse_vitals(text: str, cfg: Config) -> ParsedReply:
    """Parse one model reply into the six vitals fields (Phase 3's entry point)."""
    return parse_reply(text, cfg, VITALS_FIELDS)
