# Session parameters and reproduction notes

Recorded 7 August 2026. These are the details that are not recoverable from the
code and would otherwise be lost.

## Capture session

| Parameter | Value |
|---|---|
| Date | 7 August 2026, afternoon |
| Camera | iPhone 13 (`iPhone14,5`) rear module, over Apple Continuity Camera |
| Resolution | 1920×1080 |
| Camera height | 155 cm above floor |
| Camera-to-subject distance | 175 cm (subject against a flat wall, used as a physical stop) |
| Illumination | Indoor, fluorescent ceiling lighting, single condition |
| Median detected face width | ~130 px |
| Subjects | 9, all former students of the institution |
| Presentations | 268 total: 89 bona fide, 59 print, 54 cut-out mask, 66 replay |
| Per presentation | 2 s burst at 10 fps (~16 frames) |

Continuity Camera must have automatic reframing, portrait blur and studio
lighting **disabled**. Automatic reframing in particular changes the framing
between presentations and silently destroys the fixed geometry: the capture
still succeeds and the defect is only visible at analysis time.

Camera indices shift: with the iPhone connected it takes index 0 and the
built-in camera moves to 1. If the phone drops mid-session, `--cam 0` silently
becomes the laptop camera. Confirm the preview before each subject.

## Attack artefacts

All three species were built from the target's **enrolled archive photograph**,
so the results are an upper bound on attack success: they correspond to an
adversary who has obtained the enrolment reference. An artefact built from an
independent photograph would be expected to score lower. This is the single
most important caveat in the study.

- Print: laser print mounted flat on card
- Cut-out mask: same print, eye regions removed (ovals ~2.5 × 1.5 cm), held
  against the attacker's face
- Replay: archive photograph at full screen brightness on a tablet

Print scale was set anthropometrically: interpupillary distance 6.3 cm on the
artefact. Archive photographs are 224×224 px, so artefacts are interpolated;
at 175 cm the camera resolves the face at ~120 px, so the source resolution is
adequate but not generous.

## Operating points

Thresholds calibrated on the institutional corpus (`resultados/arcface_calib.json`):

| Target FMR | FaceNet | SFace | ArcFace |
|---|---|---|---|
| 1e-2 | 0.628 | 0.399 | 0.296 |
| 1e-3 | 0.715 | 0.474 | 0.368 |
| 1e-4 | 0.774 | 0.535 | 0.430 |
| 1e-5 | 0.819 | 0.591 | 0.597 |

τ = 0.430 (FMR 1e-4) is used as the operating point: the strictest setting at
which the genuine distribution retains margin. At τ = 0.597 the worst genuine
score clears by only 0.031, and one presentation falls below.

## Decision policy

A presentation is scored by the **maximum** over its frames, for both the
matcher and the PAD module. This is the realistic gate policy, since the system
keeps reading as the subject approaches and opens on the first frame that
passes, and it is the conservative choice against an attack.

## Data

Session captures (~1.7 GB) and archive photographs are **not** in this
repository and never will be. They are scheduled for deletion on
**31 October 2026**, after the conference.

Two folders are needed to re-run the analysis and are not vendored:

- `plantillas/`: one archive photograph per student code, named `<code>.jpg`
- `sesion/<code>/<species>/p<NN>/f<NNN>.jpg`: the captures

## Known gaps

- The virtual-credential screenshot species was never collected; it is the most
  natural species to add next.
- No public-benchmark control was run, so demographic composition and
  preprocessing normalisation remain confounded as causes of the calibration
  offset.
- The 2,739 duplicate-identity figure was not manually verified against a
  sample, so it rests on the same class of model that produced it.
- The credential (QR/HMAC) path is described in the paper but not evaluated
  beyond signature verification latency.
