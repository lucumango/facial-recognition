#!/usr/bin/env python3
"""Latencia por etapa sobre el hardware de despliegue.

Mide deteccion, alineamiento, extraccion de embedding, comparacion 1:N y PAD
sobre frames reales de la sesion, no sobre imagenes sinteticas.

Uso:
    ./venv/bin/python latencia.py --n 200
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.argv_backup = sys.argv
sys.argv = ["x"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "puntuar.py")).read().split("def main")[0])
sys.argv = sys.argv_backup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pad"))
from src.anti_spoof_predict import AntiSpoofPredict   # noqa: E402
from src.generate_patches import CropImage            # noqa: E402
from src.utility import parse_model_name              # noqa: E402


def pct(v):
    a = np.array(v) * 1000.0
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "media": float(a.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--galeria", type=int, default=105796,
                    help="tamano de galeria a simular para la busqueda 1:N")
    ap.add_argument("--out", default="resultados_latencia.json")
    args = ap.parse_args()

    pipe = Pipeline("modelos/yunet.onnx", "modelos/buffalo_l/w600k_r50.onnx")  # noqa: F821

    # PAD
    cropper = CropImage()
    modelos = []
    aqui = os.getcwd()
    mdir = os.path.abspath("pad/resources/anti_spoof_models")
    os.chdir("pad")
    try:
        for name in sorted(os.listdir(mdir)):
            if name.endswith(".pth"):
                h, w, _, scale = parse_model_name(name)
                mm = AntiSpoofPredict(0)
                mm._load_model(os.path.join(mdir, name))
                mm.model.eval()
                modelos.append((mm, h, w, scale))
    finally:
        os.chdir(aqui)

    frames = glob.glob("sesion/*/bonafide/p*/f*.jpg")
    random.seed(0)
    frames = random.sample(frames, min(args.n, len(frames)))
    print(f"{len(frames)} frames, galeria simulada de {args.galeria} plantillas")

    galeria = np.random.randn(args.galeria, 512).astype(np.float32)
    galeria /= np.linalg.norm(galeria, axis=1, keepdims=True)

    t = {k: [] for k in ("deteccion", "alineamiento", "embedding",
                         "busqueda", "pad", "total")}

    for fp in frames:
        img = cv2.imread(fp)
        h, w = img.shape[:2]
        t0 = time.perf_counter()

        pipe.det.setInputSize((w, h))
        _, faces = pipe.det.detect(img)
        t1 = time.perf_counter()
        if faces is None or len(faces) == 0:
            continue
        f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]

        chip = pipe.align(img, f)
        t2 = time.perf_counter()
        if chip is None:
            continue

        e = pipe.embed(chip)
        t3 = time.perf_counter()

        _ = np.argmax(galeria @ e)
        t4 = time.perf_counter()

        bbox = [int(v) for v in f[:4]]
        total_pad = np.zeros((1, 3))
        for mm, hh, ww, scale in modelos:
            patch = cropper.crop(org_img=img, bbox=bbox, scale=scale,
                                 out_w=ww, out_h=hh, crop=scale is not None)
            x = torch.from_numpy(np.transpose(patch, (2, 0, 1))).unsqueeze(0)
            with torch.no_grad():
                total_pad += F.softmax(mm.model.forward(x.float()), dim=1).numpy()
        t5 = time.perf_counter()

        t["deteccion"].append(t1 - t0)
        t["alineamiento"].append(t2 - t1)
        t["embedding"].append(t3 - t2)
        t["busqueda"].append(t4 - t3)
        t["pad"].append(t5 - t4)
        t["total"].append(t5 - t0)

    res = {k: pct(v) for k, v in t.items() if v}
    res["_n_frames"] = len(t["total"])
    res["_galeria"] = args.galeria

    print(f"\n{'etapa':<14}{'p50 (ms)':>10}{'p95 (ms)':>10}{'media':>10}")
    print("-" * 44)
    for k in ("deteccion", "alineamiento", "embedding", "busqueda", "pad", "total"):
        if k in res:
            print(f"{k:<14}{res[k]['p50']:>10.1f}{res[k]['p95']:>10.1f}"
                  f"{res[k]['media']:>10.1f}")

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
