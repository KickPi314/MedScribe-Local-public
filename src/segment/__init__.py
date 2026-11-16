"""Region segmentation (Phase 2, spec section 3.4).

Downstream of ``src.preprocess``: it consumes a Phase 1 pipeline result and
does not change it. The job is to turn one page of pixels into a small set of
named crops -- header strip, vitals column, complaints block, diagnosis line,
investigations margin, prescription block, and one crop per prescription line --
so that Phase 3 can run a narrow prompt per region instead of sending whole
pages to a CPU-bound VLM.

The geometry is anchored on the printed letterhead (template matching on the
logo and the doctor/registration block) rather than on whole-page fractions,
because Phase 1 measured that the printed header's position drifts by up to
3.4% of page height across the real corpus -- see config/preprocess.toml under
[crop] for the unresolved problem this stage exists to fix.

"Never guess" (spec section 1.3) applies to anchoring: when template match
confidence is below the configured threshold the anchor is reported as NOT
found, with a reason, and no region box is emitted from it.
"""

from .config import DEFAULT_SEGMENT_CONFIG_PATH, load_segment_config
from .anchors import AnchorFrame, AnchorMatch, build_frame, match_anchors
from .regions import RegionCrop, crop_regions
from .lines import LineBand, split_lines
from .pipeline import SegmentResult, segment_page

__all__ = [
    "DEFAULT_SEGMENT_CONFIG_PATH",
    "load_segment_config",
    "AnchorFrame",
    "AnchorMatch",
    "build_frame",
    "match_anchors",
    "RegionCrop",
    "crop_regions",
    "LineBand",
    "split_lines",
    "SegmentResult",
    "segment_page",
]
