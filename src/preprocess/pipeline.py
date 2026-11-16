"""Pipeline orchestration + intermediate-image dumping.

Runs the enabled steps in the configured order, optionally writing every
stage's output to ``data/dumps/<page_id>/<index>_<step>.png`` so a human can
flip through and judge quality -- which is the whole point of Phase 1.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config, PROJECT_ROOT
from .ingest import PageImage
from .steps import STEP_FUNCS, to_gray

DEFAULT_DUMP_ROOT = PROJECT_ROOT / "data" / "dumps"


@dataclass
class StageOutput:
    index: int
    name: str
    enabled: bool
    image: np.ndarray | None
    info: dict[str, Any]
    duration_s: float
    dump_path: Path | None = None


@dataclass
class PipelineResult:
    page_id: str
    source_kind: str
    effective_dpi: tuple[float, float] | None
    source_shape: tuple[int, int]
    final: np.ndarray
    stages: list[StageOutput] = field(default_factory=list)
    ctx: dict[str, Any] = field(default_factory=dict)
    total_duration_s: float = 0.0

    def stage(self, name: str) -> StageOutput | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def stage_image(self, name: str) -> np.ndarray | None:
        s = self.stage(name)
        return None if s is None else s.image


def _write_png(path: Path, img: np.ndarray, max_long_edge: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = img
    if max_long_edge:
        h, w = out.shape[:2]
        if max(h, w) > max_long_edge:
            scale = max_long_edge / float(max(h, w))
            out = cv2.resize(
                out,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError(f"failed to encode PNG for {path.name}")
    path.write_bytes(buf.tobytes())


def _contact_sheet(stages: list[StageOutput], max_long_edge: int) -> np.ndarray | None:
    """Stack every dumped stage side by side, scaled to a common height."""
    imgs = [s for s in stages if s.image is not None]
    if not imgs:
        return None
    target_h = max_long_edge
    tiles = []
    for s in imgs:
        im = s.image
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        scale = target_h / float(h)
        tile = cv2.resize(
            im, (max(int(round(w * scale)), 1), target_h), interpolation=cv2.INTER_AREA
        )
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 28), (240, 240, 240), -1)
        cv2.putText(
            tile,
            f"{s.index}_{s.name}",
            (4, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
        tiles.append(np.full((target_h, 6, 3), 128, dtype=np.uint8))
    return np.hstack(tiles[:-1])


def run_pipeline(
    page: PageImage,
    cfg: Config,
    dump_dir: Path | None = None,
    keep_stage_images: bool = True,
) -> PipelineResult:
    """Run the configured pipeline over one page.

    ``dump_dir`` -- if given, every stage output is written there as PNG plus a
    ``manifest.json`` describing the run. Pass ``None`` to skip disk writes.
    """
    src = page.image
    ctx: dict[str, Any] = {
        "page_id": page.page_id,
        "source_gray": to_gray(src),
        "source_shape": src.shape[:2],
    }

    result = PipelineResult(
        page_id=page.page_id,
        source_kind=page.source_kind,
        effective_dpi=page.effective_dpi,
        source_shape=(src.shape[0], src.shape[1]),
        final=src,
    )

    dump_max = int(cfg.get("dump", "max_long_edge"))
    stages: list[StageOutput] = []

    # index 0 is always the untouched source, so the operator can see what the
    # pipeline actually received.
    src_stage = StageOutput(0, "source", True, src, {"kind": page.source_kind}, 0.0)
    stages.append(src_stage)

    working = src
    t_total = time.perf_counter()
    for i, name in enumerate(cfg.step_order(), start=1):
        enabled = cfg.step_enabled(name)
        if not enabled:
            stages.append(
                StageOutput(i, name, False, None, {"skipped": "disabled in config"}, 0.0)
            )
            continue
        t0 = time.perf_counter()
        out, info = STEP_FUNCS[name](working, cfg, ctx)
        dt = time.perf_counter() - t0
        working = out
        stages.append(StageOutput(i, name, True, out, info, dt))

    result.total_duration_s = time.perf_counter() - t_total
    result.final = working
    result.stages = stages
    result.ctx = {k: v for k, v in ctx.items() if not isinstance(v, np.ndarray)}
    result.ctx["_source_gray_shape"] = list(ctx["source_gray"].shape)

    if dump_dir is not None:
        dump_dir = Path(dump_dir)
        for s in stages:
            if s.image is None:
                continue
            path = dump_dir / f"{s.index:02d}_{s.name}.png"
            _write_png(path, s.image, dump_max)
            s.dump_path = path
        if bool(cfg.get("dump", "contact_sheet")):
            sheet = _contact_sheet(
                stages, int(cfg.get("dump", "contact_sheet_max_long_edge"))
            )
            if sheet is not None:
                _write_png(dump_dir / "contact_sheet.png", sheet)
        _write_manifest(dump_dir, page, cfg, result)

    if not keep_stage_images:
        for s in result.stages:
            s.image = None

    return result


def _manifest_dict(page: PageImage, cfg: Config, result: PipelineResult) -> dict:
    return {
        "schema_version": 1,
        "phase": 1,
        "page_id": page.page_id,
        # Deliberately no source filename: filenames contain patient names.
        "source_kind": page.source_kind,
        "source_page_index": page.page_index,
        "source_shape_hw": list(result.source_shape),
        "effective_dpi": (
            [round(result.effective_dpi[0], 1), round(result.effective_dpi[1], 1)]
            if result.effective_dpi
            else None
        ),
        "ingest_meta": page.meta,
        "final_shape_hw": list(result.final.shape[:2]),
        "total_duration_s": round(result.total_duration_s, 3),
        "stages": [
            {
                "index": s.index,
                "name": s.name,
                "enabled": s.enabled,
                "duration_s": round(s.duration_s, 4),
                "output_shape_hw": None if s.image is None else list(s.image.shape[:2]),
                "info": s.info,
                "dump": None if s.dump_path is None else s.dump_path.name,
            }
            for s in result.stages
        ],
        "config": cfg.as_dict(),
    }


def _write_manifest(dump_dir: Path, page: PageImage, cfg: Config, result: PipelineResult) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "manifest.json").write_text(
        json.dumps(_manifest_dict(page, cfg, result), indent=2), encoding="utf-8"
    )


def dump_dir_for(page_id: str, root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_DUMP_ROOT) / page_id
