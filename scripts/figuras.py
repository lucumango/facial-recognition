#!/usr/bin/env python3
"""Genera las figuras del paper.

Reune el codigo que produjo cada figura, para que sean reproducibles y no
haya que reconstruirlas a mano.

Uso:
    ./venv/bin/python figuras.py --que galeria
    ./venv/bin/python figuras.py --que especies --code 202111177
    ./venv/bin/python figuras.py --que contactos
    ./venv/bin/python figuras.py --que hmac
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

SALIDA = "figuras"

# FMR medida por modelo a tau = 0.363 (ver resultados/arcface_calib.json)
MODELOS_FMR = [("FaceNet", 0.2760), ("SFace", 0.0243), ("ArcFace", 0.00121)]
GALERIA_OPERATIVA = 4106


def _pipeline():
    """Carga YuNet + ArcFace reutilizando las clases de puntuar.py."""
    aqui = os.path.dirname(os.path.abspath(__file__))
    argv, sys.argv = sys.argv, ["x"]
    ns = {}
    exec(open(os.path.join(aqui, "puntuar.py")).read().split("def main")[0], ns)
    sys.argv = argv
    return ns


def fig_galeria():
    """FPIR frente al tamano de galeria, por modelo (Figura 4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.logspace(1, 6, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for (nombre, fmr), ls in zip(MODELOS_FMR, ["-", "--", "-."]):
        ax.loglog(N, 1 - (1 - fmr) ** N, ls, lw=1.6,
                  color="black" if nombre == "ArcFace" else None,
                  label=f"{nombre} (FMR={100*fmr:.3g}%)")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.axvline(GALERIA_OPERATIVA, color="grey", lw=0.8, ls=":")
    ax.annotate(f"operational gallery\nN={GALERIA_OPERATIVA:,}",
                xy=(GALERIA_OPERATIVA, 2e-3), fontsize=7, ha="center",
                color="grey")
    ax.set_xlabel("Gallery size $N$ (identities)")
    ax.set_ylabel("FPIR: P(at least one false candidate)")
    ax.set_ylim(1e-4, 2)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=.3, which="both", lw=.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{SALIDA}/fig2_gallery.{ext}", dpi=160)
    print(f"-> {SALIDA}/fig2_gallery.pdf")


def fig_especies(code, presentacion="p02"):
    """Las cuatro especies como las recibe la camara (Figura 5)."""
    ns = _pipeline()
    pipe = ns["Pipeline"]("modelos/yunet.onnx", "modelos/buffalo_l/w600k_r50.onnx")
    etiquetas = [("bonafide", "Bona fide"), ("print", "Print"),
                 ("mask", "Cut-out mask"), ("screen", "Replay")]
    paneles = []
    for sp, etq in etiquetas:
        fps = sorted(glob.glob(f"sesion/{code}/{sp}/{presentacion}/f*.jpg"))
        if not fps:
            raise SystemExit(f"faltan frames en sesion/{code}/{sp}/{presentacion}")
        img = cv2.imread(fps[len(fps) // 2])
        f = pipe.detect(img)
        if f is None:
            raise SystemExit(f"sin cara en {sp}")
        x, y, w, h = [int(v) for v in f[:4]]
        m = int(h * 0.45)
        crop = cv2.resize(img[max(0, y - m):y + h + m, max(0, x - m):x + w + m],
                          (420, 420))
        cv2.rectangle(crop, (0, 378), (420, 420), (0, 0, 0), -1)
        cv2.putText(crop, etq, (12, 408), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    (255, 255, 255), 2)
        paneles.append(crop)
    cv2.imwrite(f"{SALIDA}/especies_camara.jpg", np.hstack(paneles),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"-> {SALIDA}/especies_camara.jpg  ({code})")


def fig_contactos():
    """Hoja de contactos por codigo. Herramienta interna de verificacion.

    NO va al paper: publicar el rostro junto al codigo de alumno es
    exactamente la vinculacion que convierte un conjunto de fotos en una
    base de datos biometrica.
    """
    ns = _pipeline()
    pipe = ns["Pipeline"]("modelos/yunet.onnx", "modelos/buffalo_l/w600k_r50.onnx")
    tiles = []
    for c in sorted(os.listdir("sesion")):
        fps = sorted(glob.glob(f"sesion/{c}/bonafide/p01/f*.jpg"))
        if not fps:
            continue
        img = cv2.imread(fps[len(fps) // 2])
        f = pipe.detect(img)
        if f is None:
            continue
        x, y, w, h = [int(v) for v in f[:4]]
        m = int(h * 0.55)
        crop = cv2.resize(img[max(0, y - m):y + h + m, max(0, x - m):x + w + m],
                          (300, 300))
        cv2.rectangle(crop, (0, 258), (300, 300), (0, 0, 0), -1)
        cv2.putText(crop, c, (8, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 3:
        tiles.append(np.zeros((300, 300, 3), np.uint8))
    filas = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite("quien_es_quien.jpg", np.vstack(filas))
    print("-> quien_es_quien.jpg  (uso interno, no publicar)")


def medir_hmac(n=20000):
    """Latencia de verificacion de la credencial firmada."""
    import hashlib
    import hmac
    import json
    import time

    key = b"0" * 32
    payload = json.dumps({"sub": "202299999", "iat": 1786000000,
                          "exp": 1786000300, "nonce": "a3f9c2"}).encode()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        mac = hmac.new(key, payload, hashlib.sha256).digest()
        hmac.compare_digest(mac, mac)
        t.append(time.perf_counter() - t0)
    a = np.array(t) * 1000
    print(f"HMAC-SHA256: p50 {np.percentile(a,50):.4f} ms   "
          f"p95 {np.percentile(a,95):.4f} ms   (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--que", required=True,
                    choices=("galeria", "especies", "contactos", "hmac"))
    ap.add_argument("--code", default="202111177")
    args = ap.parse_args()
    os.makedirs(SALIDA, exist_ok=True)
    {"galeria": fig_galeria,
     "especies": lambda: fig_especies(args.code),
     "contactos": fig_contactos,
     "hmac": medir_hmac}[args.que]()


if __name__ == "__main__":
    main()
