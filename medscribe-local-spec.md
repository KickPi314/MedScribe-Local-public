# MedScribe Local — Engineering Brief

Build a fully offline desktop application that assists a human in transcribing handwritten paediatric prescription notes into structured text. Read this entire brief before writing code.

---

## 1. Non-negotiable constraints

These are requirements, not preferences. Do not design around them or propose alternatives.

1. **Zero network at runtime.** No API calls, no telemetry, no CDN-hosted assets, no update checks, no cloud fallback of any kind. The app must function correctly with the network adapter disabled. Model downloads happen once, out-of-band, during setup only.
2. **CPU-only inference.** Target machine has no discrete GPU (integrated graphics only). Every architectural decision must assume CPU inference. Do not assume CUDA is available anywhere.
3. **Never guess.** When the system is not confident, it must flag the field for human review and leave it empty. It must never emit a plausible-looking value in place of an uncertain one. This is the single most important behavioural requirement in this document.
4. **Not a clinical decision tool.** The app transcribes; it does not advise, diagnose, suggest, correct, or interpret. No dose recommendations, no interaction checking, no diagnostic suggestions. Output is a draft for human verification.

---

## 2. What the documents look like

Single-clinic paediatric prescription slips, scanned at ~600dpi, one page per patient visit.

- **Pre-printed letterhead** in pink/magenta ink: clinic name and logo (top left), doctor name and registration numbers (top right), address and phone footer (bottom).
- **Handwritten content** in blue/purple ballpoint, over and around the letterhead.
- **Consistent layout:**
  - Line 1: patient ID (format `J-#####`), then patient full name.
  - Right column, top-down: date, then vitals in stable order — Wt (kg), T (°F), SpO2 (%), HR, sometimes MAC and Abd (cm).
  - Left/centre body: presenting complaints (`c/o ...` lines, often with a circled digit for duration in days), diagnosis line (prefixed with a delta symbol or `D:`), sometimes clinical notes for follow-up cases.
  - Right margin: requested investigations, arrow-prefixed (CBC, SGPT, CRP, etc).
  - Lower body: `for <first name>:` then a numbered/ticked prescription list. Each line is roughly `Sy./Tab/Inj <product name> — <dose> × <frequency>`.
  - **Gujarati script** appears as trailing marginalia on most prescription lines. This is administration instruction (timing, with/after food).
- Scan quality is poor: faint ink, low contrast against the tinted letterhead, mild skew, occasional bleed-through.

Sample content: product names recurring across pages include Pantop, Ansec, Levolin, Rinogel, Respicure-D, Histafree, Tusdex, Felispur-DS, Azee, ATM-200, P-250, P-500, Tonoferon. Dose units are almost always ml of syrup; frequency is BD/TID/OD.

---

## 3. Architecture

### 3.1 Pipeline

```
PDF/image in
  → preprocess (OpenCV, CPU, fast)
  → region segmentation (template-anchored)
  → per-region VLM extraction (narrow prompt per region)
  → validation & confidence derivation
  → human review UI
  → export
```

### 3.2 Stack

- Python 3.11+
- **Inference:** Ollama as a documented local prerequisite. Default model `qwen3-vl:4b` (Q4 quantised). Model name must be a config value, not hardcoded — the operator will want to try DeepSeek-OCR and other small VLMs and compare.
- **Imaging:** OpenCV, NumPy, PyMuPDF (PDF rasterisation).
- **Fuzzy matching:** rapidfuzz.
- **Storage:** SQLite, local file, no ORM needed.
- **UI:** FastAPI serving a local-only web UI (bind `127.0.0.1` explicitly, never `0.0.0.0`), vanilla JS, all assets vendored locally. Browser UI is chosen over native toolkits because the review workflow needs image crops displayed next to editable fields — that is awkward in Tkinter and trivial in HTML.

### 3.3 Preprocessing (do this well — it matters more than model choice)

The single largest available accuracy gain is lifting faint blue-purple ink off a pink letterhead. Implement, in order:

1. **Channel separation.** The letterhead and the ink occupy different colour regions. Extract in a colour space that separates them (test LAB and HSV; also test simple per-channel subtraction). Isolate the handwriting, suppress the letterhead.
2. **Deskew** via Hough lines or minimum-area rectangle on text contours.
3. **Adaptive contrast** — CLAHE, tuned on the real scans, not defaults.
4. **Binarise** with an adaptive/Sauvola method, not a global threshold.
5. **Crop** the printed header and footer bands out entirely.

Every step must be individually toggleable, and the app must be able to dump intermediate images to disk for inspection. The operator needs to tune this against real scans.

### 3.4 Region segmentation

Do **not** send whole pages to the VLM. On CPU this is the difference between a workable tool and an unusable one — image-token count scales with area, and prefill dominates runtime.

Anchor on the printed letterhead (template matching on the logo and the registration-number block) to establish page coordinates, then crop fixed relative regions: header strip, vitals column, complaints block, diagnosis line, investigations margin, and the prescription block. Split the prescription block into individual line crops by horizontal projection profile.

Then run one narrow, single-purpose prompt per region. "Read the vitals column, output JSON with keys wt/temp/spo2/hr" beats "transcribe this page" on both accuracy and speed.

Region geometry must live in an editable config file. The template will drift; hardcoding it guarantees a rewrite.

### 3.5 Processing model

Assume minutes per page. Design accordingly:

- Persistent job queue in SQLite. Survives restart, resumes where it stopped.
- Per-region progress reporting, not a spinner.
- Batch mode intended to be started and left alone.
- Cache results per (image hash, region, model, prompt version) so re-runs and re-tuning don't recompute everything.

---

## 4. Confidence and the failure mode

**Do not ask the model how confident it is, and do not use self-reported confidence scores for anything.** Small VLMs are poorly calibrated and will report high confidence on hallucinated drug names. Derive confidence from independent signals instead:

1. **Two-pass agreement.** Run each region twice with differently-worded prompts. Disagreement between passes is your strongest error signal. Cost is 2×, worth it.
2. **Formulary match.** Fuzzy-match every extracted product name against the local drug list (§5). Score below threshold → flag.
3. **Format validation.** Patient ID must match `J-\d{4,5}`. Vitals must parse as numbers within plausible paediatric ranges. Dates must parse. Frequency must be in a known set. Any parse failure → flag.
4. **Dose sanity check.** Where the product is in the formulary with a known concentration and mg/kg range, check the extracted dose against the recorded weight. Out of range → flag. **Flag only — never correct, never suggest a corrected value.**

Every field carries a state: `confirmed` (validated and human-approved), `extracted` (passed all checks, awaiting review), or `flagged` (failed a check, value withheld). Flagged fields render empty in the UI with the reason shown, next to the image crop, for the human to fill in.

Exports must clearly mark unverified and flagged fields. A flagged field must never silently export as a value.

### Gujarati marginalia — explicitly out of scope

Do not attempt to transcribe the Gujarati text. Off-the-shelf local models will not read fast cursive Gujarati reliably, and a wrong administration instruction is a safety problem. Instead: detect the marginalia region, crop it, and attach the **image crop** to the prescription line in the UI and in exports (as an embedded image or a referenced file). The human reads the original. This is a feature, not a limitation.

---

## 5. Local drug formulary

A local, editable data file (JSON or SQLite table) holding known products. Per entry: canonical name, aliases/common misreadings, form (syrup/tablet/injection/drop), optional concentration, optional mg/kg range, optional notes.

- Ships with a small seed list, then **grows through use**: when a human corrects a flagged product name, offer to add that spelling as an alias. Over a few hundred pages this becomes the highest-value asset in the app.
- Must be editable directly by the operator, in a plain-text format, without going through the UI.
- Fuzzy matching must be conservative. A weak match is a flag, not a substitution. Never normalise a low-confidence read to the nearest formulary entry.

---

## 6. Review UI

Optimise for speed of human verification, since that is the actual bottleneck in the workflow.

- Two-pane layout: the preprocessed page image (zoomable, pannable) beside the extracted fields.
- Selecting a field highlights its source region on the image; the relevant crop displays inline beside the field.
- Flagged fields are visually distinct, sorted to the top, and show the reason ("no formulary match", "dose 3.1× expected for 5.99 kg", "passes disagreed: Pantop / Pentop").
- Full keyboard navigation — tab between fields, one key to accept, one to jump to next flag. A mouse-only review UI will not get used.
- A page cannot be marked complete while flagged fields remain unresolved. Allow an explicit "unreadable — leave blank" resolution, recorded as such.
- Persistent, non-dismissible footer: transcription is a draft requiring verification against the original document before any clinical use.

---

## 7. Data handling

- All data stays in a local application directory. No cloud-sync folders (Dropbox/OneDrive/Google Drive) as default paths — detect and warn if the chosen path is inside one.
- Audit log table: what file was processed, when, by which model and prompt version, what was flagged, what a human changed. Append-only.
- No crash reporters, no analytics, no bundled SDK that phones home. Audit the dependency tree for outbound calls.
- Support at-rest encryption of the SQLite database, or clearly document that it is unencrypted so the operator can rely on OS-level disk encryption instead.

---

## 8. Evaluation harness

Build this early — before tuning anything.

- A ground-truth fixture format: page image + expected field values, stored locally.
- Metrics: per-field exact-match accuracy, character error rate on free-text fields, and separately **flag precision/recall** — i.e. does the system flag the fields it actually got wrong? A system that is 70% accurate and correctly flags its own 30% is far more useful here than one that is 85% accurate and silently wrong.
- Track: how often a flagged field turned out to be correctly extracted (false alarm), and how often an `extracted` field was wrong (missed error — the dangerous case).
- **Evaluate on real scans only.** Do not generate synthetic handwriting for evaluation. Fine-tuned VLMs have been shown to score well above 85% on synthetic validation sets and then collapse below 45% on real handwritten documents. Synthetic evaluation will lie to you.
- CLI command to run the eval suite and print a comparison table across models and prompt versions.

---

## 9. Build order

Do not build this end-to-end in one pass. Deliver in testable phases:

1. Preprocessing pipeline + intermediate-image dump + eval harness skeleton.
2. Region segmentation with the config-driven template.
3. Single-region VLM extraction, one region type, measured against ground truth.
4. Remaining regions + two-pass agreement.
5. Formulary, validation, and flagging.
6. Review UI.
7. Export + audit log.
8. Packaging and setup documentation.

Stop after phase 1 and report the preprocessing results before continuing.

---

## 10. Explicitly do not

- Do not add a cloud API fallback, even behind a flag, even "for comparison."
- Do not invent patient names, IDs, dates, drug names, or doses under any circumstance.
- Do not normalise an uncertain drug name to the nearest formulary entry.
- Do not attempt Gujarati transcription.
- Do not add diagnosis suggestions, dose recommendations, or interaction checks.
- Do not use model self-reported confidence as a gate.
- Do not silently drop a field that failed extraction — flag it.
- Do not assume a GPU exists.
