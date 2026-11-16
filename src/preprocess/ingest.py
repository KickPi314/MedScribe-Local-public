"""Ingestion: PDF / JPG / PNG / BMP / TIFF -> in-memory BGR numpy array.

Production input is a mix of PDF and raw camera/scanner images, so everything
downstream sees the same thing: a uint8 BGR ``np.ndarray``.

PHI note: source filenames contain real patient names. The identifier used for
every generated artefact (``page_id``) is derived from the file *content* hash,
never from the filename, so no patient name ever reaches ``data/dumps/``, a log
line, or a metrics table.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

try:  # PyMuPDF is only needed for PDF input.
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES

_HASH_CHARS = 12


@dataclass
class PageImage:
    """One page of one source document, as pixels."""

    page_id: str
    image: np.ndarray  # uint8, BGR, HxWx3
    page_index: int
    source_path: Path
    source_kind: str  # "pdf-embedded" | "pdf-render" | "image"
    effective_dpi: tuple[float, float] | None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    def describe(self) -> str:
        """PHI-safe one-line description (no filename)."""
        h, w = self.shape
        dpi = (
            f"{self.effective_dpi[0]:.0f}x{self.effective_dpi[1]:.0f}dpi"
            if self.effective_dpi
            else "dpi=unknown"
        )
        return f"{self.page_id} {w}x{h}px {dpi} via {self.source_kind}"


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def page_id_for(path: str | Path, page_index: int = 0) -> str:
    """Content-derived, PHI-free page identifier.

    Stable across runs and across renames, which also means dump directories
    and cached results survive the operator reorganising their scan folder.
    """
    return f"{file_hash(path)[:_HASH_CHARS]}_p{page_index}"


def _imdecode(data: bytes) -> np.ndarray:
    """Decode bytes to BGR. Used instead of cv2.imread so that non-ASCII
    filenames (which cv2.imread silently fails on under Windows) work."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode image data")
    return arr


def _maybe_downscale(img: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, bool]:
    if not max_long_edge:
        return img, False
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img, False
    scale = max_long_edge / float(long_edge)
    out = cv2.resize(
        img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )
    return out, True


def _pdf_pages(
    path: Path,
    page_id_base: str,
    prefer_embedded: bool,
    min_area_frac: float,
    render_dpi: int,
) -> Iterator[tuple[int, np.ndarray, str, tuple[float, float] | None, dict[str, Any]]]:
    if fitz is None:  # pragma: no cover
        raise RuntimeError("PyMuPDF (pymupdf) is required to read PDF input")
    doc = fitz.open(path)
    try:
        for pno, page in enumerate(doc):
            rect = page.rect
            page_w_in = rect.width / 72.0
            page_h_in = rect.height / 72.0
            used_embedded = False

            if prefer_embedded and page.rotation == 0:
                images = page.get_images(full=True)
                if len(images) == 1:
                    xref = images[0][0]
                    info = doc.extract_image(xref)
                    iw, ih = info["width"], info["height"]
                    # Bounding box of the placed image, as a fraction of page area.
                    try:
                        bbox = page.get_image_bbox(page.get_image_info()[0])
                        cover = (bbox.width * bbox.height) / (
                            rect.width * rect.height
                        )
                    except Exception:
                        cover = 1.0
                    if cover >= min_area_frac:
                        img = _imdecode(info["image"])
                        dpi = (iw / page_w_in, ih / page_h_in)
                        yield (
                            pno,
                            img,
                            "pdf-embedded",
                            dpi,
                            {
                                "embedded_px": [iw, ih],
                                "page_pt": [round(rect.width, 1), round(rect.height, 1)],
                                "page_cover_frac": round(float(cover), 3),
                                "embedded_format": info.get("ext"),
                            },
                        )
                        used_embedded = True

            if not used_embedded:
                zoom = render_dpi / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 1:
                    img = cv2.cvtColor(buf, cv2.COLOR_GRAY2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
                yield (
                    pno,
                    img.copy(),
                    "pdf-render",
                    (float(render_dpi), float(render_dpi)),
                    {
                        "page_pt": [round(rect.width, 1), round(rect.height, 1)],
                        "page_rotation": page.rotation,
                        "render_dpi": render_dpi,
                    },
                )
    finally:
        doc.close()


def iter_pages(path: str | Path, cfg) -> Iterator[PageImage]:
    """Yield every page of ``path`` as a :class:`PageImage`.

    ``cfg`` is a :class:`~src.preprocess.config.Config`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported input type {suffix!r}; supported: "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )

    max_long_edge = int(cfg.get("ingest", "max_long_edge"))
    base = file_hash(p)[:_HASH_CHARS]

    if suffix in PDF_SUFFIXES:
        pages = _pdf_pages(
            p,
            base,
            bool(cfg.get("ingest", "prefer_embedded_image")),
            float(cfg.get("ingest", "embedded_min_page_area_frac")),
            int(cfg.get("ingest", "render_dpi")),
        )
    else:

        def _single():
            img = _imdecode(p.read_bytes())
            yield 0, img, "image", None, {}

        pages = _single()

    for page_index, img, kind, dpi, meta in pages:
        img, resized = _maybe_downscale(img, max_long_edge)
        if resized and dpi is not None:
            h, w = img.shape[:2]
            dpi = (dpi[0] * w / meta.get("embedded_px", [w, h])[0], dpi[1])
        meta = dict(meta)
        meta["downscaled"] = resized
        yield PageImage(
            page_id=f"{base}_p{page_index}",
            image=img,
            page_index=page_index,
            source_path=p,
            source_kind=kind,
            effective_dpi=dpi,
            meta=meta,
        )


def discover_inputs(root: str | Path) -> list[Path]:
    """All supported files under ``root`` (or ``[root]`` if it is a file)."""
    r = Path(root)
    if r.is_file():
        return [r]
    if not r.is_dir():
        raise FileNotFoundError(r)
    out = [
        f
        for f in sorted(r.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return out
