"""The TUI's non-UI layer, on synthetic data: discovery, replay, eval, stages."""

from __future__ import annotations

import json
import tomllib

import pytest

from conftest import GROUND_TRUTH, N_LINES, build_synth, needs_templates, outbound
from src.extract.config import prompt_spec
from src.extract.parse import ABSENT, MISSING_KEY, NO_JSON, OK, OUT_OF_RANGE, UNPARSEABLE
from src.preprocess.ingest import iter_pages
from src.tui import workspace as W


def _crop(replay, region, line=None):
    return next(c for c in replay.crops if c.region == region and c.line_index == line)


# ---------------------------------------------------------------------------
# discovery and status
# ---------------------------------------------------------------------------


def test_discover_pages_uses_the_pipelines_page_ids(synth):
    pages, problems = W.discover_pages(synth.ws.inputs)
    assert problems == []
    assert [p.page_id for p in pages] == [synth.good, synth.bare]
    # the ids are the ones ingestion itself assigns
    for p in pages:
        decoded = [img.page_id for img in iter_pages(p.source, synth.ws.preprocess_cfg)]
        assert decoded == [p.page_id]
        assert p.n_pages == 1 and p.page_index == 0


def test_discover_pages_reports_a_missing_input(synth, tmp_path):
    pages, problems = W.discover_pages([tmp_path / "nowhere"])
    assert pages == [] and problems and "input not found" in problems[0]


def test_status_readers(synth):
    ws = synth.ws
    seg = W.read_segment_status(ws, synth.good)
    assert seg.present and seg.anchored and seg.split_ok
    assert (seg.regions_ok, seg.regions_total, seg.n_lines) == (7, 7, N_LINES)
    assert not W.read_segment_status(ws, synth.bare).present
    assert not W.read_dump_status(ws, synth.good).present
    gt, problem = W.read_fixture_statuses(ws)
    assert gt == {synth.good: "complete"} and problem == ""


# ---------------------------------------------------------------------------
# cache replay
# ---------------------------------------------------------------------------


def test_replay_reads_every_outcome_from_the_cache(synth, network):
    r = W.replay_page(synth.ws, synth.good)
    assert r.n_passes == 2 * (6 + N_LINES) and r.n_passes_present == r.n_passes
    assert all(pv.result.from_cache for c in r.crops for _, pv in c.passes())

    v = _crop(r, "vitals")
    a, b = v.a.result.reads, v.b.result.reads
    assert (a["wt_kg"].status, a["wt_kg"].value) == (OK, 10.25)
    assert (b["wt_kg"].status, b["wt_kg"].value) == (OK, 10.25)  # unit token stripped
    assert b["temp_f"].status == OUT_OF_RANGE and b["temp_f"].value is None
    assert "outside the plausible range" in b["temp_f"].reason
    assert b["hr"].status == UNPARSEABLE and b["hr"].value is None
    assert b["abd_cm"].status == MISSING_KEY
    assert a["mac_cm"].status == ABSENT and b["mac_cm"].status == ABSENT
    cmp = v.comparison
    assert cmp["wt_kg"].agree and cmp["wt_kg"].both_read
    assert not cmp["temp_f"].agree
    assert cmp["abd_cm"].both_null and not cmp["abd_cm"].status_agree

    d = _crop(r, "date")
    assert d.a.result.reads["date"].value == "2001-02-03"
    assert d.b.result.reads["date"].status == UNPARSEABLE
    assert "four-digit year" in d.b.result.reads["date"].reason
    dx = _crop(r, "diagnosis")
    assert {f.status for f in dx.b.result.reads.values()} == {NO_JSON}
    assert _crop(r, "complaints").comparison["complaints"].agree  # "d" == "days"
    assert not _crop(r, "prescription_line", 1).comparison["product"].agree

    # page totals, as the pages screen shows them
    assert (r.n_agree_read, r.n_diff, r.n_both_null, r.n_withheld) == (11, 7, 3, 6)
    assert outbound(network) == []


def test_replay_never_refreshes_and_never_allows_inference(synth, monkeypatch):
    calls = []
    real_two_pass, real_extract = W.run_two_pass, W.extract_region

    def spy_two_pass(*args, **kwargs):
        calls.append(("run_two_pass", kwargs))
        return real_two_pass(*args, **kwargs)

    def spy_extract(*args, **kwargs):
        calls.append(("extract_region", kwargs))
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(W, "run_two_pass", spy_two_pass)
    monkeypatch.setattr(W, "extract_region", spy_extract)
    synth.drop_cache("vitals", "B")
    for pid in (synth.good, synth.bare):
        W.replay_page(synth.ws, pid)
    assert calls
    for name, kw in calls:
        assert kw.get("refresh") is False, (name, kw)
        assert kw.get("allow_inference") is False, (name, kw)
    assert synth.clients == []  # the inference client was never even built


def test_replay_only_client_refuses_to_send(synth):
    client = W.ReplayOnlyClient(synth.cfg)
    for method in (client.generate, client.version, client.preflight):
        with pytest.raises(W.OllamaError, match="no request was sent"):
            method()


def test_a_missing_pass_is_reported_per_pass(synth):
    synth.drop_cache("vitals", "B")
    r = W.replay_page(synth.ws, synth.good)
    v = _crop(r, "vitals")
    assert v.a.ok and not v.b.ok
    assert "no cached extraction" in v.b.error
    assert v.state == "B not cached" and v.comparison is None
    plan = W.plan_inference(r)
    version_b = str(prompt_spec(synth.cfg, "vitals_v2")["version"])
    assert plan.missing == [("vitals", "B", version_b)]
    assert plan.blocked == []


def test_an_unsegmented_page_reports_setup_errors(synth):
    r = W.replay_page(synth.ws, synth.bare)
    presc = next(rv for rv in r.regions if rv.per_line)
    assert W.classify_error(presc.error) == "no manifest"
    assert {c.state for c in r.crops} == {"no crop"}
    plan = W.plan_inference(r)
    assert plan.missing == [] and plan.blocked


def test_replay_is_read_only_on_disk(synth):
    before = sorted(p for p in synth.root.rglob("*"))
    W.replay_page(synth.ws, synth.good)
    W.replay_page(synth.ws, synth.bare)
    assert sorted(p for p in synth.root.rglob("*")) == before


# ---------------------------------------------------------------------------
# the explicit inference path
# ---------------------------------------------------------------------------


def test_ollama_version_goes_through_the_client(synth):
    version, url = W.ollama_version(synth.ws)
    assert (version, url) == ("9.9.9-test", "http://127.0.0.1:11434")
    assert synth.events == ["version"]


def test_inference_sends_only_the_missing_pass(synth):
    synth.drop_cache("vitals", "B")
    client = synth.ws.client_factory(synth.cfg)
    r = W.replay_page(synth.ws, synth.good, infer=True, client=client, write_output=True)
    assert synth.events == ["generate"]
    assert synth.prompts == [str(prompt_spec(synth.cfg, "vitals_v2")["text"]).strip()]
    v = _crop(r, "vitals")
    assert v.a.result.from_cache and not v.b.result.from_cache
    assert synth.cache_path("vitals", "B").is_file()  # stored, so it replays next time
    assert all(pv.result.from_cache for c in r.crops if c is not v for _, pv in c.passes())


def test_inference_requires_an_explicit_client(synth):
    with pytest.raises(ValueError):
        W.replay_page(synth.ws, synth.good, infer=True)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def test_cache_only_eval_config_changes_only_the_two_flags(synth):
    path = W.write_cache_only_eval_config(synth.ws)
    assert path.parent == synth.ws.out_dir
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# GENERATED by python -m src.tui")
    derived = tomllib.loads(text)
    with open(W.DEFAULT_EVAL_CONFIG_PATH, "rb") as fh:
        source = tomllib.load(fh)
    assert source["vitals"]["run_extraction_if_missing"] is True  # the real default
    source["fixtures"]["dir"] = (synth.root / "fixtures").as_posix()
    source["vitals"]["run_extraction_if_missing"] = False
    source["extract"]["run_extraction_if_missing"] = False
    assert derived == source


def test_dump_toml_round_trips_config_shaped_data():
    data = {
        "a": {"s": 'quote " and \\ back', "i": 3, "f": 0.5, "e": 1e-05, "b": False,
              "l": [1, 2.5, "x"], "u": "ünïcode", "nested": {"k": [["x"], []]}},
        "b": {"c": {"d": {"deep": True}}},
        "weird key": {"x.y": 1},
    }
    assert tomllib.loads(W.dump_toml(data)) == data


def test_run_eval_vitals_is_the_cli_cache_only(synth, network):
    run = W.run_eval(synth.ws, "vitals")
    assert run.ok, run.error
    assert run.argv[:2] == ["--eval-config", str(synth.ws.out_dir / "eval_cache_only.toml")]
    assert "--refresh" not in run.argv
    o = run.metrics["overall"]
    assert (o["n_correct"], o["n_fields_scored"], o["n_emitted_wrong"]) == (5, 6, 1)
    assert "[6] Overall" in run.output and "preflight" not in run.output
    assert run.json_path == synth.ws.out_dir / "eval_vitals.json"
    assert synth.clients == [] and outbound(network) == []


def test_run_eval_extract(synth, network):
    run = W.run_eval(synth.ws, "extract")
    assert run.ok, run.error
    agreement = run.metrics["two_pass_agreement"]["overall"]
    assert (agreement["n_compared"], agreement["n_agree"]) == (21, 14)
    assert (agreement["n_both_read"], agreement["n_agree_both_read"]) == (14, 11)
    assert run.metrics["per_region_overall"]["vitals"]["pass_a"]["n_correct"] == 5
    assert outbound(network) == []


def test_run_eval_fails_loudly_on_a_cache_miss_instead_of_inferring(synth, network):
    synth.drop_cache("vitals", "A")
    run = W.run_eval(synth.ws, "vitals")
    assert not run.ok and run.exit_code == 2
    assert "no cached extraction" in run.output
    assert not run.json_path.exists()
    assert synth.clients == [] and outbound(network) == []


# ---------------------------------------------------------------------------
# stage runs (the real Phase 1/2 code on a synthetic scan)
# ---------------------------------------------------------------------------


@needs_templates
def test_stage_runs_on_a_synthetic_scan(synth):
    ws = synth.ws
    page = next(p for p in W.discover_pages(ws.inputs)[0] if p.page_id == synth.bare)
    pre = W.run_preprocess_stage(ws, page)
    assert pre.status == "ok", pre.error
    assert (ws.dumps_root / synth.bare / "manifest.json").is_file()
    assert W.read_dump_status(ws, synth.bare).present
    seg = W.run_segment_stage(ws, page)
    # a synthetic page has no letterhead: Phase 2 refuses to anchor it, and the
    # stage reports that rather than failing
    assert seg.status == "warn" and "UNANCHORED" in seg.summary
    assert any("not anchored" in line for line in seg.lines)
    status = W.read_segment_status(ws, synth.bare)
    assert status.present and not status.anchored


def test_crop_diff_detects_changed_bytes(synth):
    seg_dir = synth.ws.segments_root / synth.good
    before = W.crop_hashes(seg_dir)
    assert len(before) == 7 + N_LINES
    (seg_dir / "vitals_column.png").write_bytes(b"different bytes")
    d = W.diff_crops(before, W.crop_hashes(seg_dir))
    assert d["changed"] == ["vitals_column.png"] and len(d["unchanged"]) == 6 + N_LINES


def test_summarise_replay(synth):
    rep = W.summarise_replay(W.replay_page(synth.ws, synth.good))
    assert rep.status == "ok" and "16/16 passes from cache" in rep.summary
    rep = W.summarise_replay(W.replay_page(synth.ws, synth.bare))
    assert rep.status == "warn" and any("no manifest" in ln.lower() or "manifest" in ln
                                        for ln in rep.lines)


# ---------------------------------------------------------------------------
# views: what each cell says
# ---------------------------------------------------------------------------


def test_synthetic_ground_truth_is_synthetic():
    # A guard on the guard: the fixture data must stay obviously invented.
    blob = json.dumps(GROUND_TRUTH)
    assert "SYN-0001" in blob and "SYNTH-A" in blob


def test_build_synth_honours_fake_kwargs(tmp_path):
    s = build_synth(tmp_path / "ws2", version="1.2.3-test")
    assert W.ollama_version(s.ws)[0] == "1.2.3-test"
