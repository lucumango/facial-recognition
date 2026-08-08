#!/usr/bin/env python3
"""Scoring offline de la sesion de captura.

Toma las presentaciones grabadas por capturar.py, las compara contra las
plantillas de archivo, y reporta por umbral de operacion:

  IAPMR   por especie de PAI (print, mask)  -- ataques que el matcher acepta
  FRR     sobre bona fide                   -- legitimos que rechaza
  TPR     genuino (= 1 - FRR)

Politica de decision a nivel presentacion: se acepta la presentacion si
CUALQUIER frame supera el umbral. Es el comportamiento realista de una
puerta, que sigue leyendo mientras la persona se acerca.

Uso:
    ./venv/bin/python puntuar.py
    ./venv/bin/python puntuar.py --sesion sesion --out resultados.json
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

# plantilla de 5 puntos de ArcFace sobre 112x112
ARCFACE_5PT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# umbrales ArcFace calibrados sobre el corpus de archivo (arcface_calib.json)
TAUS = {"1e-2": 0.296, "1e-3": 0.368, "1e-4": 0.430, "1e-5": 0.597}

SPECIES = ("bonafide", "print", "mask", "screen")


class Pipeline:
    def __init__(self, det_path, rec_path):
        self.det = cv2.FaceDetectorYN.create(det_path, "", (320, 320),
                                             score_threshold=0.6)
        self.rec = ort.InferenceSession(rec_path,
                                        providers=["CPUExecutionProvider"])
        self.rec_in = self.rec.get_inputs()[0].name

    def detect(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return None
        # la cara mas grande: en una puerta, la persona que se esta presentando
        return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]

    def align(self, img, face):
        pts = face[4:14].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_5PT, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(img, M, (112, 112), borderValue=0)

    def embed(self, chip):
        x = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[None]
        v = self.rec.run(None, {self.rec_in: x})[0][0]
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def embed_image(self, img):
        f = self.detect(img)
        if f is None:
            return None
        chip = self.align(img, f)
        if chip is None:
            return None
        return self.embed(chip)


def load_templates(pipe, dirs):
    gallery = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.jpg")) +
                        glob.glob(os.path.join(d, "*.jpeg")) +
                        glob.glob(os.path.join(d, "*.png"))):
            code = os.path.splitext(os.path.basename(p))[0].upper().lstrip("U")
            img = cv2.imread(p)
            if img is None:
                print(f"  ! no se pudo leer {p}")
                continue
            e = pipe.embed_image(img)
            if e is None:
                print(f"  ! sin cara detectada en la plantilla {p}")
                continue
            gallery[code] = e
    return gallery


def score_presentation(pipe, pdir, tmpl):
    """Devuelve (score_max, score_mean, n_frames_con_cara, n_frames)."""
    frames = sorted(glob.glob(os.path.join(pdir, "f*.jpg")))
    scores = []
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            continue
        e = pipe.embed_image(img)
        if e is None:
            continue
        scores.append(float(np.dot(e, tmpl)))
    if not scores:
        return None, None, 0, len(frames)
    return max(scores), float(np.mean(scores)), len(scores), len(frames)


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95%. Con n chico, mejor que el normal."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sesion", default="sesion")
    ap.add_argument("--plantillas", nargs="+",
                    default=["plantillas", "plantillas_grupo2"])
    ap.add_argument("--out", default="resultados.json")
    args = ap.parse_args()

    pipe = Pipeline("modelos/yunet.onnx", "modelos/buffalo_l/w600k_r50.onnx")

    print("Cargando plantillas de archivo...")
    gallery = load_templates(pipe, args.plantillas)
    print(f"  {len(gallery)} plantillas: {', '.join(sorted(gallery))}\n")

    codes = sorted(d for d in os.listdir(args.sesion)
                   if os.path.isdir(os.path.join(args.sesion, d)))
    presentations = []
    sin_plantilla = []

    for code in codes:
        if code not in gallery:
            sin_plantilla.append(code)
            continue
        tmpl = gallery[code]
        for sp in SPECIES:
            base = os.path.join(args.sesion, code, sp)
            if not os.path.isdir(base):
                continue
            for pdir in sorted(glob.glob(os.path.join(base, "p*"))):
                smax, smean, nf, nt = score_presentation(pipe, pdir, tmpl)
                presentations.append({
                    "code": code, "species": sp,
                    "presentation": os.path.basename(pdir),
                    "score_max": smax, "score_mean": smean,
                    "frames_con_cara": nf, "frames_total": nt,
                })
                st = f"{smax:.3f}" if smax is not None else "SIN CARA"
                print(f"{code} {sp:9s} {os.path.basename(pdir)}  "
                      f"max={st}  ({nf}/{nt} frames)")

    if sin_plantilla:
        print(f"\n! codigos capturados sin plantilla de archivo: "
              f"{', '.join(sin_plantilla)}")

    # ---- metricas por umbral -------------------------------------------
    results = {"taus": TAUS, "por_umbral": {}, "n_sujetos": len(
        {p['code'] for p in presentations})}

    print("\n" + "=" * 62)
    for name, tau in TAUS.items():
        row = {}
        for sp in SPECIES:
            sub = [p for p in presentations if p["species"] == sp]
            n = len(sub)
            # una presentacion sin cara detectada NO cuenta como aceptada
            acc = sum(1 for p in sub
                      if p["score_max"] is not None and p["score_max"] >= tau)
            lo, hi = wilson(acc, n)
            row[sp] = {"n": n, "aceptadas": acc,
                       "tasa": acc / n if n else None,
                       "ic95": [lo, hi]}
        bf = row["bonafide"]
        row["FRR_bonafide"] = (1 - bf["tasa"]) if bf["tasa"] is not None else None
        attack_rates = [row[s]["tasa"] for s in ("print", "mask", "screen")
                        if row[s]["n"] > 0 and row[s]["tasa"] is not None]
        # ISO: la especie peor, no el promedio
        row["IAPMR_global"] = max(attack_rates) if attack_rates else None
        results["por_umbral"][name] = row

        print(f"\ntau = {tau:.3f}  (FMR objetivo {name})")
        for sp in SPECIES:
            r = row[sp]
            if r["n"] == 0:
                continue
            print(f"  {sp:9s} {r['aceptadas']:3d}/{r['n']:3d} = "
                  f"{100*r['tasa']:6.2f}%   IC95 "
                  f"[{100*r['ic95'][0]:.1f}, {100*r['ic95'][1]:.1f}]")
        if row["IAPMR_global"] is not None:
            print(f"  -> IAPMR global (peor especie) = "
                  f"{100*row['IAPMR_global']:.2f}%")
        if row["FRR_bonafide"] is not None:
            print(f"  -> FRR bona fide               = "
                  f"{100*row['FRR_bonafide']:.2f}%")

    results["presentaciones"] = presentations
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
