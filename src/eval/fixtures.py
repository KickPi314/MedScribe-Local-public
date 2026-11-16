"""Ground-truth fixture format and loader (spec section 8).

One JSON file per page, named ``<page_id>.json``, in the fixtures directory.

The field set mirrors the document layout described in spec section 2 so that
when extraction lands (phase 3+) the harness already knows the shape of the
answer. In Phase 1 every value is ``null`` / ``[]`` because **no ground truth
has been transcribed yet**. That is recorded explicitly in
``ground_truth_status``, and the harness refuses to compute accuracy against
un-transcribed fixtures rather than scoring nulls against nulls and printing a
flattering number.

PHI: a fixture stores the path of its source scan, and those filenames contain
patient names, so ``data/fixtures/`` is gitignored. ``page_id`` itself is a
content hash and carries no PHI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

STATUS_EMPTY = "empty"        # nothing transcribed yet
STATUS_PARTIAL = "partial"    # some fields transcribed
STATUS_COMPLETE = "complete"  # fully transcribed, safe to score against
VALID_STATUS = (STATUS_EMPTY, STATUS_PARTIAL, STATUS_COMPLETE)


class FixtureError(RuntimeError):
    pass


def empty_fields() -> dict[str, Any]:
    """The Phase 1 field template. All values null/empty, by design.

    Keys follow spec section 2's layout. Gujarati marginalia is deliberately
    an image-crop reference, not text -- transcribing it is out of scope
    (spec section 4).
    """
    return {
        "patient_id": None,          # "J-#####"
        "patient_name": None,
        "date": None,                # ISO 8601 once transcribed
        "vitals": {
            "wt_kg": None,
            "temp_f": None,
            "spo2_pct": None,
            "hr": None,
            "mac_cm": None,
            "abd_cm": None,
        },
        "complaints": [],            # list[{"text": str, "duration": int|null,
                                     #       "duration_unit": "days"|"hours"|null}]
        "diagnosis": None,
        "clinical_notes": None,
        "investigations": [],        # list[str]
        "prescriptions": [],         # list[{form, product, dose, frequency,
                                     #       gujarati_marginalia_crop}]
    }


@dataclass
class Fixture:
    page_id: str
    source_path: Path
    page_index: int = 0
    ground_truth_status: str = STATUS_EMPTY
    fields: dict[str, Any] = field(default_factory=empty_fields)
    notes: str = ""
    path: Path | None = None

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "page_id": self.page_id,
            "source_path": str(self.source_path),
            "page_index": self.page_index,
            "ground_truth_status": self.ground_truth_status,
            "fields": self.fields,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> "Fixture":
        missing = [
            k
            for k in ("schema_version", "page_id", "source_path", "ground_truth_status", "fields")
            if k not in data
        ]
        if missing:
            raise FixtureError(f"{path}: fixture missing key(s) {missing}")
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise FixtureError(
                f"{path}: fixture schema_version {data['schema_version']} != "
                f"{SCHEMA_VERSION}"
            )
        status = data["ground_truth_status"]
        if status not in VALID_STATUS:
            raise FixtureError(
                f"{path}: ground_truth_status {status!r} not in {VALID_STATUS}"
            )
        return cls(
            page_id=data["page_id"],
            source_path=Path(data["source_path"]),
            page_index=int(data.get("page_index", 0)),
            ground_truth_status=status,
            fields=data["fields"],
            notes=data.get("notes", ""),
            path=path,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        self.path = path
        return path

    # -- queries ---------------------------------------------------------
    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth_status != STATUS_EMPTY

    def resolve_source(self, project_root: Path) -> Path:
        p = self.source_path
        if not p.is_absolute():
            p = project_root / p
        if not p.is_file():
            raise FixtureError(
                f"{self.path}: source scan not found at {p}. Fixtures reference "
                f"scans by path; update source_path if the scans moved."
            )
        return p


def fixture_path_for(fixtures_dir: Path, page_id: str) -> Path:
    return Path(fixtures_dir) / f"{page_id}.json"


def scaffold_fixture(fixtures_dir: Path, page_id: str, source_path: Path,
                     page_index: int = 0, overwrite: bool = False) -> tuple[Path, bool]:
    """Create an empty fixture for a page. Returns (path, created)."""
    path = fixture_path_for(fixtures_dir, page_id)
    if path.exists() and not overwrite:
        return path, False
    fx = Fixture(page_id=page_id, source_path=source_path, page_index=page_index)
    fx.notes = (
        "Ground truth not transcribed yet. Fill in `fields` by reading the "
        "original scan, then set ground_truth_status to 'partial' or 'complete'. "
        "Do not guess a value you cannot read; leave it null."
    )
    fx.save(path)
    return path, True


def load_fixtures(fixtures_dir: str | Path) -> list[Fixture]:
    d = Path(fixtures_dir)
    if not d.is_dir():
        raise FixtureError(
            f"fixtures directory not found: {d}. Create fixtures with "
            f"`python -m src.eval init-fixtures <scan-dir>`."
        )
    out: list[Fixture] = []
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FixtureError(f"{p}: invalid JSON ({exc})") from exc
        out.append(Fixture.from_dict(data, path=p))
    return out


def iter_fixture_pages(fixtures: list[Fixture], project_root: Path) -> Iterator[tuple[Fixture, Path]]:
    for fx in fixtures:
        yield fx, fx.resolve_source(project_root)
