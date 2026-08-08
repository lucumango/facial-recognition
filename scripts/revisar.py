#!/usr/bin/env python3
"""Revisa una presentacion ya capturada: tamano de cara en px, nitidez, encuadre.

Uso:
    ./venv/bin/python revisar.py sesion/TEST/bonafide/p01
"""
import glob
import os
import sys

import cv2
import numpy as np

pdir = sys.argv[1] if len(sys.argv) > 1 else "sesion/TEST/bonafide/p01"
det = cv2.FaceDetectorYN.create("modelos/yunet.onnx", "", (320, 320),
                                score_threshold=0.6)

frames = sorted(glob.glob(os.path.join(pdir, "f*.jpg")))
if not frames:
    sys.exit(f"No hay frames en {pdir}")

anchos, nitidez, fuera, sin_cara = [], [], 0, 0
for fp in frames:
    img = cv2.imread(fp)
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        sin_cara += 1
        continue
    f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
    x, y, fw, fh = f[:4]
    anchos.append(fw)
    chip = img[max(0, int(y)):int(y + fh), max(0, int(x)):int(x + fw)]
    if chip.size:
        nitidez.append(cv2.Laplacian(cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F).var())
    if x < 0 or y < 0 or x + fw > w or y + fh > h:
        fuera += 1

print(f"{pdir}")
print(f"  frames            {len(frames)}  (sin cara: {sin_cara})")
if not anchos:
    sys.exit("  NINGUN frame con cara detectada -- revisa iluminacion/encuadre")
print(f"  ancho de cara     mediana {np.median(anchos):.0f} px  "
      f"(min {min(anchos):.0f}, max {max(anchos):.0f})")
print(f"  nitidez (Laplac.) mediana {np.median(nitidez):.0f}")
print(f"  cara cortada por el borde en {fuera} frames")

a = np.median(anchos)
print()
if a < 112:
    print("  MAL: la cara mide menos que la entrada del modelo (112 px).")
    print("       Sube la resolucion o acerca la marca -- y anota la distancia nueva.")
elif a < 150:
    print("  JUSTO: funciona, pero sin margen. Aceptable si no puedes cambiarlo.")
else:
    print("  BIEN: tamano de cara suficiente.")
if np.median(nitidez) < 100:
    print("  OJO: nitidez baja. Puede ser desenfoque o movimiento.")
if fuera:
    print("  OJO: la cara toca el borde en algunos frames. Sube o baja el tripode.")
