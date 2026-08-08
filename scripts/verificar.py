#!/usr/bin/env python3
"""Verifica que cada presentacion de un codigo muestre a la persona correcta.

Detecta los dos errores tipicos de la sesion:
  - presentaciones vacias (nadie/nada presentado)
  - artefacto de otra persona (foto o tablet equivocada)

Uso:
    ./venv/bin/python verificar.py 202299998
    ./venv/bin/python verificar.py            # todos los codigos
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.argv_backup = sys.argv
sys.argv = ["x"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "puntuar.py")).read().split("def main")[0])
sys.argv = sys.argv_backup

pipe = Pipeline("modelos/yunet.onnx", "modelos/buffalo_l/w600k_r50.onnx")  # noqa: F821
gal = load_templates(pipe, ["plantillas", "plantillas_grupo2"])            # noqa: F821

codes = sys.argv[1:] or sorted(os.listdir("sesion"))
problemas = 0

for code in codes:
    print(f"\n=== {code} ===")
    for sp in ("bonafide", "print", "mask", "screen"):
        for pdir in sorted(glob.glob(f"sesion/{code}/{sp}/p*")):
            mejor, quien = -1.0, None
            for fp in sorted(glob.glob(pdir + "/f*.jpg"))[::4]:
                e = pipe.embed_image(cv2.imread(fp))
                if e is None:
                    continue
                for k, v in gal.items():
                    s = float(np.dot(e, v))
                    if s > mejor:
                        mejor, quien = s, k
            etq = os.path.basename(pdir)
            if quien is None:
                print(f"  {sp:9s} {etq}  VACIA -- ninguna cara detectada")
                problemas += 1
            elif quien != code:
                print(f"  {sp:9s} {etq}  ES DE {quien} ({mejor:.3f})")
                problemas += 1
            elif mejor < 0.4:
                print(f"  {sp:9s} {etq}  score bajo {mejor:.3f} -- revisar")
                problemas += 1

print(f"\n{'OK, nada raro' if not problemas else f'{problemas} presentaciones a revisar'}")
