# Securing Campus Access — code and aggregate results

Code and aggregate statistics accompanying the SIMBig 2026 submission
*Securing Campus Access with Biometric Authentication, Liveness Detection, and
Secure QR Codes: Calibration and Evaluation on 105,796 Real Student
Photographs*.

**No biometric data is published here.** No student photograph, no session
capture and no face embedding is included in this repository, and none will be.
Consent for institutional research use does not extend to public
redistribution, and biometric data is irrevocable once released. What is
published is the code needed to reproduce the pipeline and the aggregate score
distributions needed to reproduce the calibration analysis.

## Layout

```
scripts/     capture and analysis pipeline
resultados/  aggregate statistics (no student codes, no per-subject records)
NOTES.md     session parameters, operating points, known gaps
```

## Scripts

| Script | Purpose |
|---|---|
| `camaras.py` | Enumerate cameras visible to OpenCV |
| `capturar.py` | Record presentations (ISO/IEC 30107-3), one burst per presentation |
|  `revisar.py` | Per-presentation quality check: face width in px, sharpness, framing |
| `verificar.py` | Confirm each presentation shows the labelled identity |
| `puntuar.py` | Matcher scoring: genuine comparisons and IAPMR per PAI species |
| `pad_score.py` | PAD evaluation with MiniFASNet: APCER, BPCER, ACER |
| `latencia.py` | Per-stage latency on deployment hardware |
| `foto.py` | Self-timer capture for setup documentation |
| `figuras.py` | Regenerate the paper figures and the HMAC timing |

## Aggregate results

| File | Contents |
|---|---|
| `arcface_calib.json` | Impostor score histogram and FMR as a function of threshold |
| `compare.json` | Expected false candidates vs. gallery size, three models |
| `matcher_agregado.json` | Acceptance rates per species and threshold, with Wilson 95% intervals |
| `pad_agregado.json` | APCER per species, BPCER, ACER |
| `resultados_latencia.json` | Per-stage latency, p50/p95 |

Per-presentation records are deliberately excluded: they are keyed by student
code and are therefore linkable to individuals.

## Reproducing

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Models are not vendored. Download separately:

- YuNet `face_detection_yunet_2023mar.onnx` — [opencv_zoo](https://github.com/opencv/opencv_zoo)
- ArcFace `buffalo_l` (`w600k_r50.onnx`) — [insightface](https://github.com/deepinsight/insightface) releases
- MiniFASNet — [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

Expected layout: `modelos/yunet.onnx`, `modelos/buffalo_l/w600k_r50.onnx`,
`pad/` (the cloned anti-spoofing repository).

## Ethics

The institutional corpus was made available for coursework at the institution
and the analysis grew out of that coursework under the supervision of the course
instructor. That is academic supervision, not a determination by the
institution's data-protection function; the paper states this limitation
explicitly rather than claiming broader authorisation. Under Peruvian Ley
N.º 29733, biometric data is sensitive personal data.

The nine session participants gave written informed consent covering capture,
research use, retention and — separately and optionally — publication of their
image. All nine granted the publication authorisation. Consent to appear in a
figure is not consent to release a biometric dataset, and is not treated as
such. Session captures are deleted on 31 October 2026.
