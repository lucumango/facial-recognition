#!/usr/bin/env python3
"""Foto suelta con temporizador, para documentar el montaje.

La MacBook hace de fotografo desde un costado mientras el iPhone sigue
montado en el tripode como camara de captura.

Uso:
    ./venv/bin/python foto.py --out figuras/montaje.jpg
    ./venv/bin/python foto.py --out figuras/ataque.jpg --delay 15
"""
import argparse
import os
import subprocess
import time

import cv2


def hablar(t):
    try:
        subprocess.Popen(["say", "-r", "220", t],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="ruta del jpg de salida")
    ap.add_argument("--cam", type=int, default=1,
                    help="1 = camara del MacBook (0 suele ser el iPhone)")
    ap.add_argument("--delay", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la camara {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    win = "encuadre  [ESPACIO=disparar con temporizador  q=salir]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"Camara {args.cam}. Encuadra, ESPACIO para disparar en "
          f"{args.delay:.0f} s, q para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            t0 = time.time()
            dicho = None
            while True:
                resta = args.delay - (time.time() - t0)
                if resta <= 0:
                    break
                ent = int(resta) + 1
                if ent != dicho and ent <= 5:
                    hablar(str(ent))
                    dicho = ent
                ok, frame = cap.read()
                if ok:
                    v = frame.copy()
                    cv2.putText(v, str(ent), (40, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 6)
                    cv2.imshow(win, v)
                cv2.waitKey(1)
            hablar("ya")
            for _ in range(6):          # dejar que estabilice exposicion
                ok, frame = cap.read()
            cv2.imwrite(args.out, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"guardada  {args.out}  {frame.shape[1]}x{frame.shape[0]}")
            hablar("foto tomada")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
