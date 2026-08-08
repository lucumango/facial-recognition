#!/usr/bin/env python3
"""Evaluacion PAD zero-shot con MiniFASNet (Silent-Face-Anti-Spoofing).

Los dos modelos preentrenados se aplican tal cual, sin reentrenar sobre
nuestros datos. Eso es a proposito: la pregunta es si un PAD comercial
listo para usar protege un despliegue real, no si se puede ajustar a el.

Metricas ISO/IEC 30107-3:
  APCER  por especie = ataques que el PAD clasifica como bona fide
  BPCER               = bona fide que el PAD clasifica como ataque
  ACER                = (APCER_peor + BPCER) / 2

Politica a nivel presentacion: se toma el maximo P(real) sobre los frames.
Es la puerta realista -- al atacante le basta que un frame pase.

Uso:
    ./venv/bin/python pad_score.py
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pad"))
from src.anti_spoof_predict import AntiSpoofPredict  # noqa: E402
from src.generate_patches import CropImage          # noqa: E402
from src.utility import parse_model_name            # noqa: E402

SPECIES = ("bonafide", "print", "mask", "screen")
MODEL_DIR = "pad/resources/anti_spoof_models"


class PAD:
    """Ensamble de los dos MiniFASNet, como en el demo original."""

    def __init__(self, model_dir=MODEL_DIR):
        self.cropper = CropImage()
        self.models = []
        model_dir = os.path.abspath(model_dir)
        # el repo carga su detector Caffe con rutas relativas a pad/;
        # entramos ahi solo para construir y volvemos enseguida
        aqui = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pad"))
        try:
            for name in sorted(os.listdir(model_dir)):
                if not name.endswith(".pth"):
                    continue
                h, w, _, scale = parse_model_name(name)
                m = AntiSpoofPredict(0)
                m._load_model(os.path.join(model_dir, name))
                m.model.eval()
                self.models.append((name, m, h, w, scale))
        finally:
            os.chdir(aqui)
        if not self.models:
            raise SystemExit(f"No hay modelos .pth en {model_dir}")
        print(f"PAD: {len(self.models)} modelos -> "
              f"{', '.join(n for n, *_ in self.models)}")

    def p_real(self, img, bbox):
        """P(bona fide) sumada sobre el ensamble. bbox = [x, y, w, h]."""
        total = np.zeros((1, 3))
        for _, m, h, w, scale in self.models:
            patch = self.cropper.crop(**{
                "org_img": img, "bbox": bbox, "scale": scale,
                "out_w": w, "out_h": h, "crop": scale is not None,
            })
            x = torch.from_numpy(np.transpose(patch, (2, 0, 1))).unsqueeze(0)
            x = x.float().to(m.device)
            with torch.no_grad():
                total += F.softmax(m.model.forward(x), dim=1).cpu().numpy()
        return float(total[0][1] / len(self.models))


def wilson(k, n, z=1.96):
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
    ap.add_argument("--out", default="resultados_pad.json")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="umbral de P(real) para clasificar como bona fide")
    args = ap.parse_args()

    det = cv2.FaceDetectorYN.create("modelos/yunet.onnx", "", (320, 320),
                                    score_threshold=0.6)
    pad = PAD()

    rows = []
    for code in sorted(os.listdir(args.sesion)):
        for sp in SPECIES:
            for pdir in sorted(glob.glob(os.path.join(args.sesion, code, sp, "p*"))):
                vals = []
                for fp in sorted(glob.glob(os.path.join(pdir, "f*.jpg"))):
                    img = cv2.imread(fp)
                    if img is None:
                        continue
                    hh, ww = img.shape[:2]
                    det.setInputSize((ww, hh))
                    _, faces = det.detect(img)
                    if faces is None or len(faces) == 0:
                        continue
                    f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
                    bbox = [int(v) for v in f[:4]]
                    vals.append(pad.p_real(img, bbox))
                if not vals:
                    continue
                rows.append({"code": code, "species": sp,
                             "presentation": os.path.basename(pdir),
                             "p_real_max": max(vals),
                             "p_real_mean": float(np.mean(vals)),
                             "n_frames": len(vals)})
                print(f"{code} {sp:9s} {os.path.basename(pdir)}  "
                      f"P(real)max={max(vals):.3f}")

    # ---- metricas ISO --------------------------------------------------
    res = {"theta": args.theta, "por_especie": {}, "presentaciones": rows}
    print("\n" + "=" * 62)
    print(f"theta = {args.theta}  (P(real) >= theta  =>  clasificado bona fide)\n")

    apcers = []
    for sp in SPECIES:
        sub = [r for r in rows if r["species"] == sp]
        n = len(sub)
        if n == 0:
            continue
        como_real = sum(1 for r in sub if r["p_real_max"] >= args.theta)
        if sp == "bonafide":
            k, tasa, etiqueta = n - como_real, (n - como_real) / n, "BPCER"
        else:
            k, tasa, etiqueta = como_real, como_real / n, "APCER"
            apcers.append(tasa)
        lo, hi = wilson(k, n)
        res["por_especie"][sp] = {"n": n, "metrica": etiqueta, "tasa": tasa,
                                  "ic95": [lo, hi]}
        print(f"  {sp:9s} {etiqueta:6s} {k:3d}/{n:3d} = {100*tasa:6.2f}%   "
              f"IC95 [{100*lo:.1f}, {100*hi:.1f}]")

    if apcers and "bonafide" in res["por_especie"]:
        apcer = max(apcers)                      # ISO: la peor especie
        bpcer = res["por_especie"]["bonafide"]["tasa"]
        res["APCER_global"] = apcer
        res["BPCER"] = bpcer
        res["ACER"] = (apcer + bpcer) / 2
        print(f"\n  APCER global (peor especie) = {100*apcer:.2f}%")
        print(f"  BPCER                       = {100*bpcer:.2f}%")
        print(f"  ACER                        = {100*res['ACER']:.2f}%")

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
