"""Terminal UI over the existing pipeline -- IN PROGRESS.

The plan: a keyboard-driven, full-screen front end for the stages that exist today
(spec section 9 phases 1-4): see the input pages and their page_ids, run or
replay preprocess -> segment -> extract for one page, browse one page's
extraction region by region (both passes, each field's parse status, the
withholding reason), and view the eval tables ``python -m src.eval`` computes.

It is a FRONT END, not a second pipeline. Every number on screen comes from
calling the existing modules -- ``src.preprocess``, ``src.segment``,
``src.extract`` (including ``agreement.py``'s two-pass comparison) and the
``src.eval`` CLI itself -- and nothing here re-derives what they compute.

WHAT IS IN THIS COPY. Only ``workspace.py``, the non-UI half (paths, page
discovery, cache replay, stage runs, the cache-only eval runner), which is
testable without a terminal (tests/test_tui_workspace.py). The Textual screens
that sit on top of it are still being built and are not included yet, so
there is no ``python -m src.tui`` entry point in this copy.

NETWORK POSTURE (spec section 1.1). By default the TUI opens no socket at all:

* extraction is a pure replay of the spec section 3.5 cache. The client handed
  to ``extract_region`` in replay mode is :class:`workspace.ReplayOnlyClient`,
  which raises instead of sending anything, and ``allow_inference`` is false;
* the eval runner calls ``python -m src.eval`` in-process against a generated
  copy of config/eval.toml whose only change is
  ``run_extraction_if_missing = false``, so it neither preflights Ollama nor
  re-infers a cache miss (a miss fails loudly instead);
* ``--refresh`` and every refresh-style option are never passed, anywhere.

The one exception is an explicit, confirmed action that infers the passes
missing from the cache (``plan_inference`` + ``replay_page(infer=True)`` here;
the confirmation lives in the screens). It first calls Ollama's
``/api/version`` through the existing loopback-only client and shows the
version (a silent Ollama auto-update has changed model outputs before), and
only sends inference requests after the operator confirms.
Cached passes are reused, never re-sent.

NOT HERE, deliberately: formulary matching, flagging and the field state
machine (phase 5), the real review UI with image crops (phase 6), export and
the audit log (phase 7), packaging (phase 8). No passes' disagreement is
resolved and no value is picked or corrected; both readings are shown.

PHI: the screens show patient data at runtime (that is their purpose), but
this package writes nothing patient-derived outside ``data/out/`` (the
derived eval config and the eval metrics JSON go to ``data/out/tui/``) and
keeps no log file.
"""

__all__ = ["workspace"]
