"""Shared fixtures for the TUI tests. SYNTHETIC DATA ONLY.

Everything a test touches is generated here under pytest's ``tmp_path``: two
synthetic "scans", Phase 2 crops and manifests, extraction cache entries built
with the real ``src.extract.cache`` key function, and a ground-truth fixture.
No value in this file comes from a real slip -- the names, IDs, products and
readings are invented placeholders ("SYN-0001", "SYNTH-A", ...).

Two guards run for every test, through a Python audit hook:

* ``no_phi_reads`` fails the test if anything opens or lists a file under
  ``samples/`` or ``data/`` -- the folders that hold real patient data;
* ``network`` records every socket connect, name lookup and urllib request,
  so a test can assert that the default flows send nothing.

The tracked config files (config/*.toml, config/templates/*.png) are read as
they are by the app: they are configuration, not patient data.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.extract import cache as cache_mod  # noqa: E402
from src.extract.config import (  # noqa: E402
    EXTRACT_REGIONS,
    output_dir,
    prompt_spec,
    region_prompt_names,
    region_spec,
)
from src.extract.agreement import agreement_dir  # noqa: E402
from src.preprocess.ingest import page_id_for  # noqa: E402
from src.segment.config import REGION_NAMES  # noqa: E402
from src.tui.workspace import Workspace  # noqa: E402

# ---------------------------------------------------------------------------
# audit-hook guards
# ---------------------------------------------------------------------------

_PHI_ROOTS = tuple(os.path.normcase(str(p)) for p in (PROJECT_ROOT / "samples",
                                                      PROJECT_ROOT / "data"))
_STATE: dict[str, Any] = {"phi": None, "net": None}


def _audit(event: str, args: tuple) -> None:
    phi, net = _STATE["phi"], _STATE["net"]
    if phi is not None and event in ("open", "os.scandir", "os.listdir"):
        target = args[0] if args else None
        if isinstance(target, (str, bytes, os.PathLike)):
            try:
                path = os.path.normcase(os.path.abspath(os.fsdecode(target)))
            except (TypeError, ValueError):
                path = ""
            if any(path == r or path.startswith(r + os.sep) for r in _PHI_ROOTS):
                phi.append((event, path))
    if net is not None and event in ("socket.connect", "socket.getaddrinfo",
                                     "urllib.Request", "socket.sendto"):
        net.append((event, args[1:] if event == "socket.connect" else args[:2]))


sys.addaudithook(_audit)


@pytest.fixture(autouse=True)
def no_phi_reads():
    """Every test: nothing may open or list a file in the real-data folders."""
    seen: list[tuple[str, str]] = []
    _STATE["phi"] = seen
    try:
        yield seen
    finally:
        _STATE["phi"] = None
    assert not seen, f"a test touched the patient-data folders: {seen[:5]}"


@pytest.fixture
def network():
    """Records connect / getaddrinfo / urllib.Request events during the test."""
    events: list[tuple[str, Any]] = []
    _STATE["net"] = events
    try:
        yield events
    finally:
        _STATE["net"] = None


def _templates_available() -> bool:
    """True when the letterhead template PNGs Phase 2 anchors on are present.

    They are clinic letterhead crops, so the public copy of this repo does not
    ship them; a test that runs the real segment stage needs them.
    """
    from src.preprocess.config import ConfigError
    from src.segment.anchors import load_templates
    from src.segment.config import load_segment_config

    try:
        load_templates(load_segment_config())
    except ConfigError:
        return False
    return True


needs_templates = pytest.mark.skipif(
    not _templates_available(),
    reason="needs the letterhead templates in config/templates/ (not shipped in "
           "the public copy; regenerate with `python -m src.segment make-templates`)",
)


OLLAMA_PORT = 11434


def outbound(events: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Events that would be a request to Ollama or to anywhere off this machine.

    asyncio on Windows builds its event-loop wake-up pipe from a loopback
    socket pair, which connects to 127.0.0.1 on an ephemeral port; that is
    process-internal plumbing, not a request, and is the only thing allowed.
    """
    bad = []
    for event, args in events:
        if event == "socket.connect":
            addr = args[0] if args else None
            host, port = (addr[0], addr[1]) if isinstance(addr, tuple) and len(addr) >= 2 else (addr, None)
            if host not in ("127.0.0.1", "::1") or port == OLLAMA_PORT:
                bad.append((event, args))
        else:
            bad.append((event, args))
    return bad


# ---------------------------------------------------------------------------
# synthetic workspace
# ---------------------------------------------------------------------------

#: Synthetic model replies, per (region, pass) or (region, pass, line). Each is
#: chosen to exercise one parser outcome; see the expected-value tests.
REPLIES: dict[tuple, Any] = {
    ("header", "A"): {"patient_id": "SYN-0001", "patient_name": "Synthetic Patient One"},
    ("header", "B"): {"patient_id": "SYN-0001", "patient_name": "Synthetic Patient 0ne"},
    ("date", "A"): {"date": "03/02/2001"},
    ("date", "B"): {"date": "03/02/01"},  # two-digit year: UNPARSEABLE, never expanded
    ("vitals", "A"): {"wt_kg": 10.25, "temp_f": "99.1", "spo2_pct": 97, "hr": 110,
                      "mac_cm": None, "abd_cm": None},
    # wt with a unit (stripped), temp out of range, spo2 with %, hr not one
    # number, mac absent, abd omitted (MISSING_KEY).
    ("vitals", "B"): {"wt_kg": "10.25 kg", "temp_f": 991, "spo2_pct": "97 %",
                      "hr": "1 1 0", "mac_cm": None},
    ("complaints", "A"): {"complaints": [{"text": "symptom alpha", "duration": 2,
                                          "duration_unit": "days"}]},
    ("complaints", "B"): {"complaints": [{"text": "symptom alpha", "duration": 2,
                                          "duration_unit": "d"}]},
    ("diagnosis", "A"): {"diagnosis": "synthetic diagnosis", "clinical_notes": None},
    ("diagnosis", "B"): "the slip is not legible",  # not JSON at all: NO_JSON
    ("investigations", "A"): {"investigations": ["TEST-1", "TEST-2"]},
    ("investigations", "B"): {"investigations": ["TEST-1"]},
    ("prescription_line", "A", 0): {"form": "Sy", "product": "SYNTH-A", "dose": "5 ml",
                                    "frequency": "BD"},
    ("prescription_line", "B", 0): {"form": "Sy", "product": "SYNTH-A", "dose": "5 ml",
                                    "frequency": "BD"},
    ("prescription_line", "A", 1): {"form": "Tab", "product": "SYNTH-B", "dose": "1",
                                    "frequency": "OD"},
    ("prescription_line", "B", 1): {"form": "Tab", "product": "SYNTH-8", "dose": "1",
                                    "frequency": "OD"},
}

#: Synthetic ground truth for the "good" page. hr differs from pass A's 110 on
#: purpose, so the vitals eval has exactly one WRONG field: 5/6.
GROUND_TRUTH: dict[str, Any] = {
    "patient_id": "SYN-0001",
    "patient_name": "Synthetic Patient One",
    "date": "2001-02-03",
    "vitals": {"wt_kg": 10.25, "temp_f": 99.1, "spo2_pct": 97, "hr": 112,
               "mac_cm": None, "abd_cm": None},
    "complaints": [{"text": "symptom alpha", "duration": 2, "duration_unit": "days"}],
    "diagnosis": "synthetic diagnosis",
    "clinical_notes": None,
    "investigations": ["TEST-1", "TEST-2"],
    "prescriptions": [
        {"form": "Sy", "product": "SYNTH-A", "dose": "5 ml", "frequency": "BD",
         "gujarati_marginalia_crop": None},
        {"form": "Tab", "product": "SYNTH-B", "dose": "1", "frequency": "OD",
         "gujarati_marginalia_crop": None},
    ],
}

N_LINES = 2


def _toml_path(p: Path) -> str:
    return json.dumps(p.as_posix())


def _png(rng: np.random.Generator, h: int, w: int, colour: bool = False) -> bytes:
    shape = (h, w, 3) if colour else (h, w)
    ok, buf = cv2.imencode(".png", rng.integers(0, 255, shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _scan(rng: np.random.Generator) -> bytes:
    """A synthetic page: light paper with dark bars. Not a real slip."""
    img = np.full((1100, 800, 3), 235, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, 700)), int(rng.integers(0, 1000))
        cv2.rectangle(img, (x, y), (x + 60, y + 8), (120, 60, 40), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class FakeOllama:
    """Stands in for OllamaClient. Records calls, in order, into ``events``
    (shared by every instance a workspace builds); opens no socket."""

    def __init__(self, cfg, events: list[str], prompts: list[str], *,
                 version: str = "9.9.9-test", fail_version: bool = False) -> None:
        self.base_url = "http://127.0.0.1:11434"
        self.model = str(cfg.get("model", "name"))
        self.events = events
        self.prompts = prompts
        self._version = version
        self._fail_version = fail_version

    def version(self) -> str:
        self.events.append("version")
        if self._fail_version:
            from src.extract.client import OllamaError
            raise OllamaError("cannot reach Ollama (test)")
        return self._version

    def preflight(self) -> list[str]:
        self.events.append("preflight")
        return [f"ollama {self._version} (fake)"]

    def generate(self, prompt: str, image_bytes: bytes, options=None):
        from src.extract.client import ModelReply

        self.events.append("generate")
        self.prompts.append(prompt)
        text = json.dumps({"wt_kg": 10.25, "temp_f": 99.1, "spo2_pct": 97, "hr": 110,
                           "mac_cm": None, "abd_cm": None})
        return ModelReply(text=text, text_field="response",
                          raw={"done_reason": "stop", "model": self.model})


@dataclass
class Synth:
    root: Path
    ws: Workspace
    good: str
    bare: str
    crops: dict[tuple[str, int | None], bytes] = field(default_factory=dict)
    #: Every client the workspace built for the explicit inference action.
    clients: list[FakeOllama] = field(default_factory=list)
    #: Ordered log of every call made on any of those clients.
    events: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    @property
    def cfg(self):
        return self.ws.extract_cfg

    def cache_path(self, region: str, tag: str, line_index: int | None = None) -> Path:
        cfg = self.cfg
        segment_region = str(region_spec(cfg, region)["segment_region"])
        prompt = region_prompt_names(cfg, region)[0 if tag == "A" else 1]
        version = str(prompt_spec(cfg, prompt)["version"])
        key = cache_mod.make_key(self.crops[(region, line_index)], segment_region, cfg, version)
        return cache_mod.path_for(key, cfg)

    def drop_cache(self, region: str, tag: str, line_index: int | None = None) -> Path:
        p = self.cache_path(region, tag, line_index)
        p.unlink()
        return p


def build_synth(root: Path, **fake_kwargs: Any) -> Synth:
    """``fake_kwargs`` go to every :class:`FakeOllama` the workspace builds."""
    rng = np.random.default_rng(20260101)
    scans = root / "scans"
    scans.mkdir(parents=True)
    (scans / "synthetic_a.png").write_bytes(_scan(rng))
    (scans / "synthetic_b.png").write_bytes(_scan(rng))
    good = page_id_for(scans / "synthetic_a.png", 0)
    bare = page_id_for(scans / "synthetic_b.png", 0)

    synth_holder: dict[str, Synth] = {}

    def factory(cfg):
        s = synth_holder["s"]
        c = FakeOllama(cfg, s.events, s.prompts, **fake_kwargs)
        s.clients.append(c)
        return c

    ws = Workspace(
        inputs=[scans],
        extract_overrides=[
            f"inputs.segments_dir={_toml_path(root / 'segments')}",
            f"inputs.fixtures_dir={_toml_path(root / 'fixtures')}",
            f"cache.dir={_toml_path(root / 'cache')}",
            f"output.dir={_toml_path(root / 'out' / 'extract')}",
            f"output.agreement_dir={_toml_path(root / 'out' / 'agreement')}",
        ],
        eval_overrides={"fixtures": {"dir": (root / "fixtures").as_posix()}},
        dumps_root=root / "dumps",
        out_dir=root / "out" / "tui",
        client_factory=factory,
    )
    synth = Synth(root=root, ws=ws, good=good, bare=bare)
    synth_holder["s"] = synth
    cfg = ws.extract_cfg

    # -- Phase 2 output for the good page --------------------------------
    seg_dir = root / "segments" / good
    seg_dir.mkdir(parents=True)
    for region in EXTRACT_REGIONS:
        spec = region_spec(cfg, region)
        indices = list(range(N_LINES)) if spec.get("per_line") else [None]
        for i in indices:
            name = str(spec["crop_file"]).format(index=i) if i is not None else str(spec["crop_file"])
            data = _png(rng, 40 + 3 * len(synth.crops), 120)
            (seg_dir / name).write_bytes(data)
            synth.crops[(region, i)] = data
    manifest = {
        "schema_version": 1, "phase": 2, "stage": "region_segmentation", "page_id": good,
        "frame": {"ok": True, "source": "two_anchor", "reason": None},
        "regions": [{"name": n, "enabled": True, "geometry_ok": True, "clamped": False,
                     "dump": f"{n}.png", "notes": []} for n in REGION_NAMES],
        "prescription_lines": {
            "info": {"split_ok": True, "lines": N_LINES},
            "bands": [],
            "dumps": [f"prescription_block_line_{i}.png" for i in range(N_LINES)],
        },
        "warnings": [],
    }
    (seg_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (seg_dir / "prescription_block.png").write_bytes(_png(rng, 60, 120))

    # -- cache entries, keyed exactly as extract_region keys them ----------
    for (region, i), data in synth.crops.items():
        spec = region_spec(cfg, region)
        segment_region = str(spec["segment_region"])
        for tag, prompt in zip(("A", "B"), region_prompt_names(cfg, region)):
            version = str(prompt_spec(cfg, prompt)["version"])
            reply = REPLIES[(region, tag, i)] if i is not None else REPLIES[(region, tag)]
            text = reply if isinstance(reply, str) else json.dumps(reply)
            key = cache_mod.make_key(data, segment_region, cfg, version)
            cache_mod.store(key, cfg, {
                "page_id": good, "region_key": region, "prompt_name": prompt,
                "line_index": i, "crop_path": f"synthetic/{region}", "crop_shape_hw": None,
                "raw_response": text, "raw_response_field": "response",
                "model_meta": {"model": ws.model, "done_reason": "stop",
                               "total_duration": 1_000_000_000, "load_duration": 0,
                               "prompt_eval_count": 300, "prompt_eval_duration": 200_000_000,
                               "eval_count": 20, "eval_duration": 500_000_000},
                "warnings": [],
            })

    # -- ground truth for the good page only ------------------------------
    fx_dir = root / "fixtures"
    fx_dir.mkdir(parents=True)
    (fx_dir / f"{good}.json").write_text(json.dumps({
        "schema_version": 1, "page_id": good, "source_path": (scans / "synthetic_a.png").as_posix(),
        "page_index": 0, "ground_truth_status": "complete", "fields": GROUND_TRUTH,
        "notes": "synthetic test fixture",
    }), encoding="utf-8")

    # Every path the app will use must be inside the temporary directory.
    used = [ws.segments_root, ws.fixtures_root, ws.eval_fixtures_root, ws.cache_root,
            ws.dumps_root, ws.out_dir, output_dir(cfg), agreement_dir(cfg), *ws.inputs]
    for p in used:
        assert Path(p).resolve().is_relative_to(root.resolve()), p
    return synth


@pytest.fixture
def synth(tmp_path: Path) -> Synth:
    return build_synth(tmp_path / "ws")
