# MedScribe Local

[![tests](https://github.com/KickPi314/medscribe-local-public/actions/workflows/tests.yml/badge.svg)](https://github.com/KickPi314/medscribe-local-public/actions/workflows/tests.yml)

MedScribe Local reads a pediatrician's handwritten prescription slips and turns them into structured fields (patient ID, date, vitals, complaints, diagnosis, investigations, prescription lines) that a person then checks. I'm building it with a pediatrician in my hometown in India. Everything runs on the clinic's own machine with a local vision model, so patient data never leaves the device.

**Where it's at:** the extraction pipeline and the eval harness work, and a terminal UI is in progress. Matching drug names against the clinic's drug list, a review UI and export are still to be implemented. This is also only one of the two stages we're working on. The other is a Gujarati-to-English speech-to-text transcriber, which is still in its early-early stages and isn't in this repo.

> Quick note: this is the public copy of a private working repo, scrubbed before I put it up. Patient scans, ground truth, model outputs, my build notes and the clinic's letterhead aren't here. Identifying details in comments and prompt examples (patient names and IDs, real visit dates, the doctor's name and registration numbers) were swapped for placeholders. Every number below was measured in the private repo on real slips.

## What does it do?

It works in stages:

- **OpenCV preprocessing** lifts the blue/purple ballpoint ink off the pink pre-printed letterhead, deskews, applies CLAHE and binarizes.
- **A template segmenter** finds the printed letterhead and cuts each page into 7 fixed regions from config, then splits the prescription block into one crop per line.
- **Qwen3-VL running in Ollama** (`qwen3-vl:4b`) reads each region with a narrow, single-purpose prompt.
- **A strict parser** withholds a value (leaves it empty, with a reason) instead of guessing when the reply is malformed, unparseable or implausible.
- **Two separate passes** read every region with two differently worded prompts and record whether they agree.
- **An eval harness** scores all of this against ground truth the doctor verified, from real scans.

The output is a draft for a person to verify. The tool only transcribes; it doesn't advise, diagnose, correct or interpret anything.

## How does the pipeline work?

```
scan (PDF or image)
  │
  ▼
src/preprocess   separate the ink from the pink letterhead → deskew → CLAHE → binarize
  │              → crop header/footer bands
  ▼
src/segment      template-match the printed letterhead to fix page coordinates, then crop
  │              7 regions: header_strip, date_row, vitals_column, complaints_block,
  │              diagnosis_line, investigations_margin, prescription_block (split into lines)
  ▼
src/extract      one narrow prompt per region → qwen3-vl:4b via Ollama on 127.0.0.1
  │              → strict parser that withholds a value instead of guessing
  │              → each region read twice (pass A / pass B) → per-field agreement
  │              → replies cached by (crop hash, region, model, prompt version)
  ▼
src/eval         exact match, character error rate, two-pass agreement and inference
                 timing against ground truth the doctor verified
```

Every preprocessing step can be switched on or off in config, and each stage can dump its intermediate images to disk so you can look at them by eye (which is how I've found most of the bugs). Region boxes, model name, prompts, sampling options and plausibility ranges all live in `config/*.toml`, not in code, and most values have a comment saying how I measured them or why I picked them.

## How well does it work?

Measured on **7 real slips**, which is every slip I have so far. At this size one page moves most numbers by about 14 points, so treat these as a baseline, not a verdict on the approach. `qwen3-vl:4b` on Ollama 0.40.14, pass A (the primary prompt), exact match against ground truth:

| Region | Exact match |
|---|---|
| vitals (weight, temperature, SpO2, heart rate, ...) | **38/42** |
| header (patient ID, name) | 4/14 |
| date | 1/7 |
| diagnosis | 3/14 |
| complaints | 0/7 |
| investigations | 0/7 |
| prescription lines | 5/44 |

- **Two-pass agreement:** 87/247 fields. Both passes can agree on the same wrong value, so agreement is a signal for review, not a confidence score.
- **Speed:** about 1.3 s per model call (median, warm) and about 30 s of model time per page, on an RTX 4060 laptop GPU. The clinic's machine is CPU-only, so it will be several times slower there.
- **Flag precision/recall** (spec §8) is to be implemented. It needs the `flagged` field state that Phase 5 adds.

And don't get me wrong, this is nowhere near done. The vitals work, but the free-text regions, the header, and speed on the clinic's CPU-only machine are all still open problems. On hard handwriting a 4B model swaps in real drug names that aren't on the slip (`amoxil` came back as `amoxicillin`). The scorer counts those as wrong, and catching them is the job of the next phase, which checks drug names against the clinic's own drug list.

## Why is it built this way?

**Withhold, don't guess.** [`src/extract/parse.py`](src/extract/parse.py) re-derives every value from the raw reply. If a value isn't valid JSON, can't be reduced to a number, or falls outside the field's plausible range, it's stored as `null` with a reason, never repaired. One example: the model once read a printed `%` as a digit and gave back an SpO2 of 100.1. Rounding that to 100 would look right and still be made up, so it gets withheld. Lists are all-or-nothing, because three out of four complaints looks exactly like a complete list.

**Two passes, no winner.** [`src/extract/agreement.py`](src/extract/agreement.py) records whether the two readings agree and deliberately has no "final value". Picking between them is the reviewer's job, not the code's.

**Check the labels, not just the model.** Several fields first scored near zero, and it turned out the ground truth was wrong, not the model. The doctor had written patient IDs inside the transcription template's `(e.g. ...)` slot, so they got transcribed as empty. Circled durations had been recorded as missing. And one slip wrote a duration in hours, so the schema went from `duration_days` to `duration` + `duration_unit`, and a missing unit is stored as unknown instead of assumed to be days.

**What an Ollama update broke.** Ollama auto-updated itself (0.32.6 to 0.40.14) and the JSON output broke with zero code changes. The model started writing a hidden reasoning block that ate the whole token budget. An empty-reasoning prefill fixed it, and vitals still scored 38/42 afterwards, but only 40 of 158 raw replies matched the old runtime byte for byte. So I treat the runtime version as part of the experiment. Replies are cached by crop hash, region, model and prompt version, so every number can be replayed and runtime changes show up as diffs.

**Zero network.** The only connection the client makes is to Ollama on loopback (`127.0.0.1`). `config/extract.toml` holds an allowlist of loopback hosts, and loading the config fails if `[ollama].host` is anything else (tested in [`tests/`](tests/)). The Ollama client uses only the standard library (`urllib`), and dependencies are kept to NumPy, OpenCV and PyMuPDF, so the dependency tree stays small enough to actually audit for outbound calls. The model gets downloaded once, during setup, and that's it.

## What's in here?

```
src/preprocess/          Phase 1: ingest (PDF rasterizing via PyMuPDF), preprocessing steps, intermediate dumps
src/segment/             Phase 2: letterhead anchoring, region geometry, prescription line splitting
src/extract/             Phases 3-4: Ollama client, prompts, parser, result cache, two-pass agreement
src/eval/                eval harness: fixture format and loader, segmentation, vitals and per-region metrics
src/tui/                 terminal UI, in progress: only its non-UI layer (workspace.py) is in this copy
config/preprocess.toml   preprocessing steps and parameters
config/segment.toml      anchor templates and region boxes (in letterhead-anchored units)
config/extract.toml      model, Ollama host (loopback only), prompts, sampling, field rules
config/eval.toml         fixture location and scoring rules
tests/                   unit tests on synthetic data (no scans, no Ollama)
medscribe-local-spec.md  requirements and architecture
```

## What's done and what's next?

The build follows the phases in [`medscribe-local-spec.md`](medscribe-local-spec.md) §9.

| Phase | Scope | Status |
|---|---|---|
| 1 | Preprocessing pipeline, intermediate-image dump, eval harness skeleton | Done |
| 2 | Region segmentation from a config-driven template | Done |
| 3 | Single-region VLM extraction (vitals), measured against ground truth | Done |
| 4 | Remaining regions and two-pass agreement | Done |
| – | Terminal UI over phases 1-4 | In progress (see below) |
| 5 | Formulary, validation and flagging | To be implemented; waiting on the clinic's drug list |
| 6 | Review UI | To be implemented |
| 7 | Export and audit log | To be implemented |
| 8 | Packaging and setup documentation | To be implemented |

### Terminal UI (in progress)

I'm building a keyboard-driven terminal UI so the pipeline can be used without memorizing CLI flags: a list of input pages with their status, a page view that goes region by region showing both passes side by side with each field's parse status and why a value was withheld, a screen to run preprocess → segment → extract for one page, and the eval tables.

What's in this copy is the non-UI layer, [`src/tui/workspace.py`](src/tui/workspace.py): page discovery, cache replay, per-page stage runs and a cache-only eval runner, tested in [`tests/test_tui_workspace.py`](tests/test_tui_workspace.py). It keeps the zero-network rule: by default it only replays the cache, with a client that refuses to send anything, and the eval runs with `run_extraction_if_missing = false`, so a cache miss fails loudly instead of quietly calling the model. Inferring missing passes is an explicit action that checks the Ollama version first and only sends the passes the cache doesn't have.

The screens on top of it are still being built and aren't in this copy yet, so there's no `python -m src.tui` to run here.

## How do I install it?

### What you'll need

- **Python 3.11 or newer.** Config files are parsed with the standard-library `tomllib`.
- **Ollama**, installed from [ollama.com](https://ollama.com).
- **The model**, pulled once (this is the only step that needs the network):

  ```powershell
  ollama pull qwen3-vl:4b
  ```

- **Optional: an NVIDIA GPU.** Ollama uses it automatically if it's there. Everything also runs on CPU, just more slowly.

### 1. Clone the repository

```powershell
git clone https://github.com/KickPi314/medscribe-local-public.git
cd medscribe-local-public
```

### 2. Create a virtual environment and install the dependencies

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The dependencies are NumPy, OpenCV and PyMuPDF. To run the tests, which need no scans and no Ollama:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

One test runs the real segment stage, so it's skipped until you've made the letterhead templates in step 4.

Heads up for Windows: a bare `python` can resolve to the Microsoft Store stub instead of a real interpreter. Call the venv's interpreter directly (`.venv\Scripts\python.exe`) for every command below. On macOS/Linux, use `.venv/bin/python` wherever these instructions show `.venv\Scripts\python.exe`.

### 3. Add input scans and ground truth

Scans and ground truth aren't in the repository (see [Patient data](#what-about-patient-data)). You supply them locally:

| Folder | Contents |
|---|---|
| `samples/` | Input scans, PDF or image, one page per visit |
| `data/fixtures/` | Ground-truth fixtures, one JSON file per page, named `<page_id>.json` |

A `page_id` is a hash of the page's content, so it carries no patient information. The fixture format, including every field and its type, is documented in [`src/eval/fixtures.py`](src/eval/fixtures.py). To create empty fixtures for your scans:

```powershell
.venv\Scripts\python.exe -m src.eval init-fixtures samples
```

This writes one fixture per page with every field empty and `ground_truth_status` set to `"empty"`. Fill in the fields from the original slips and set the status to `"complete"`. The harness only scores against complete fixtures, so it never compares empty ground truth with empty output. Use `--overwrite` to regenerate existing fixtures.

Fill these in carefully. Several of my fields first scored near zero because the ground truth was wrong, not the model. If a field scores suspiciously low, check the label before blaming the model.

The extraction and eval commands find their pages from `data/fixtures/`, so fixtures need to exist before step 7.

### 4. Make the letterhead templates

The segmenter anchors every page on crops of the printed letterhead, and this copy doesn't ship them, because they identify the clinic. Set `[templates].source_page_id` in `config/segment.toml` to the `page_id` of your scan with the crispest letterhead (the fixture filenames from step 3), then:

```powershell
.venv\Scripts\python.exe -m src.segment make-templates samples
```

Look at the generated PNGs in `config/templates/` by eye: they should contain printed letterhead only, no handwriting. The template boxes and all the region boxes in `config/segment.toml` were measured on one clinic's letterhead, so slips from anywhere else also need their boxes re-measured.

### 5. Check that Ollama is running

```powershell
curl.exe http://127.0.0.1:11434/api/version
```

(On macOS/Linux, plain `curl`.) If this fails, start Ollama with `ollama serve`, or launch the desktop app. Then confirm the model and inputs from MedScribe's side:

```powershell
.venv\Scripts\python.exe -m src.extract probe
```

### 6. Preprocess and segment

```powershell
# Optional: inspect preprocessing; writes each stage's image to data/dumps/<page_id>/
.venv\Scripts\python.exe -m src.preprocess run samples --dump-intermediates

# Check letterhead anchoring confidence per page
.venv\Scripts\python.exe -m src.segment anchors samples

# Crop the regions and prescription lines into data/segments/<page_id>/ (needed by extraction)
.venv\Scripts\python.exe -m src.segment run samples --dump-crops
```

### 7. Extract and evaluate

```powershell
# Read every region through both prompts and show the two passes side by side
.venv\Scripts\python.exe -m src.extract regions

# Score all regions: exact match, character error rate, two-pass agreement, timing
.venv\Scripts\python.exe -m src.eval extract

# Score the vitals region only
.venv\Scripts\python.exe -m src.eval vitals
```

Model replies are cached under `data/out/extract_cache/`, so repeat runs read from the cache and finish almost instantly. The eval commands still need Ollama to be reachable, though. Useful flags:

- `--region NAME` limits `src.extract regions` or `src.eval extract` to one region (for example `--region prescription_line`).
- `--page-id PAGE_ID` limits `src.extract regions` or `src.extract vitals` to one page.
- `--show-raw` prints the raw model replies.
- `--set-extract 'model.name="other-model:tag"'` scores a different Ollama model with `src.eval` (or `--set` with `src.extract`), so you can compare models without editing config.
- `--refresh` ignores the cache and re-runs inference. A full `src.eval extract --refresh` is 162 model calls.

Every command has `--help`.

### Before you `--refresh`

- **Check the Ollama version.** As above, an Ollama update changed the readings even with fixed sampling settings, and the cache key doesn't include the Ollama version. Check `ollama --version` before running anything with `--refresh`, and note it next to any results you record. On a deployment machine, turning off the app's update check also keeps Ollama itself from contacting the internet.
- **Keep `OLLAMA_HOST` on loopback.** Leave it unset or set it to `127.0.0.1`. Never `0.0.0.0`, which would expose the model server to the local network.

## What about patient data?

Real scans and everything made from them are patient data and stay on the clinic's machine. These paths are gitignored and must never be committed:

- `samples/`: the input scans
- `data/fixtures/`: ground truth, which also records source file paths
- `data/dumps/`: preprocessing intermediates
- `data/segments/`: region and line crops
- `data/out/`: model replies, the extraction cache, per-page results and metrics

`config/templates/` is gitignored in this copy too. The private repo's history is not part of this copy, and the tests here use synthetic data only: the test setup fails any test that opens a file under `samples/` or `data/`.

## How is it evaluated?

The eval harness only uses real scans. I don't use synthetic handwriting for evaluation because it overstates accuracy on real documents (spec §8). Other `src.eval` subcommands cover earlier phases: `run` (preprocessing proxy metrics), `segment` (anchor confidence and region sanity) and `channels` (ranks channel-separation methods).

## How I built it

Most of the implementation was written with Claude Code. I made the design decisions, worked with the doctor on the ground truth, and did the failure analysis.
