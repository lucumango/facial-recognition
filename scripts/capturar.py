#!/usr/bin/env python3
"""Captura de presentaciones para el estudio PAD (ISO/IEC 30107-3).

No corre reconocimiento: solo graba frames crudos y metadatos.
El scoring (APCER/BPCER/genuinos/latencia) se hace despues, offline.

Uso:
    ./venv/bin/python capturar.py --code 202216148 --species bonafide
    ./venv/bin/python capturar.py --code 202216148 --species print
    ./venv/bin/python capturar.py --code 202216148 --species mask

Teclas:
    ESPACIO  graba una presentacion (rafaga de ~2 s)
    d        descarta la ultima presentacion grabada
    q / ESC  salir
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import cv2

DEVNULL = subprocess.DEVNULL


def hablar(texto):
    """Voz por el parlante del Mac: se oye desde el otro lado del cuarto."""
    try:
        subprocess.Popen(["say", "-r", "220", texto],
                         stdout=DEVNULL, stderr=DEVNULL)
    except Exception:
        pass

SPECIES = ("bonafide", "print", "mask", "screen")
BURST_SECONDS = 2.0
BURST_FPS = 10


def next_presentation(base):
    n = 0
    if os.path.isdir(base):
        for d in os.listdir(base):
            if d.startswith("p") and d[1:].isdigit():
                n = max(n, int(d[1:]))
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="codigo del alumno objetivo (u ej. 202216148)")
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default="sesion")
    ap.add_argument("--distance-cm", type=float, default=None,
                    help="distancia camara-sujeto en cm (se pide si no se pasa)")
    ap.add_argument("--holder", default="", help="quien sostiene el artefacto (print/mask)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="segundos de cuenta regresiva antes de cada rafaga")
    ap.add_argument("--auto", type=int, default=0,
                    help="graba N presentaciones seguidas sin volver al teclado")
    ap.add_argument("--gap", type=float, default=4.0,
                    help="segundos entre presentaciones en modo --auto (sal y vuelve)")
    args = ap.parse_args()

    # en modo auto siempre hace falta margen para llegar a la pared
    if args.auto and args.delay == 0.0:
        args.delay = 6.0

    code = args.code.upper().lstrip("U")
    base = os.path.join(args.out, code, args.species)
    os.makedirs(base, exist_ok=True)

    if args.distance_cm is None:
        args.distance_cm = float(input("Distancia camara-sujeto en cm: ").strip())

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la camara {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camara {args.cam} abierta a {w}x{h}")

    n = next_presentation(base)
    last_dir = None
    win = f"{code} / {args.species}  [ESPACIO=grabar  d=descartar  q=salir]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def mostrar(texto="", color=(0, 255, 0)):
        """Lee un frame y lo pinta con la guia y un mensaje. Devuelve la tecla."""
        ok, frame = cap.read()
        if not ok:
            return None
        view = frame.copy()
        gx, gy = int(w * 0.30), int(h * 0.10)
        cv2.rectangle(view, (gx, gy), (w - gx, h - int(h * 0.05)), color, 2)
        cv2.putText(view, texto or f"proxima: p{n:02d}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
        cv2.imshow(win, view)
        return cv2.waitKey(1) & 0xFF

    def cuenta_regresiva(seg):
        """Cuenta atras con voz, sin congelar la vista previa."""
        t0 = time.time()
        dicho = None
        while True:
            resta = seg - (time.time() - t0)
            if resta <= 0:
                break
            ent = int(resta) + 1
            if ent != dicho and ent <= 3:
                hablar(str(ent))
                dicho = ent
            mostrar(f"p{n:02d} en {ent}", (0, 200, 255))
        hablar("ya")

    def grabar():
        """Graba una rafaga completa. Devuelve el directorio."""
        pdir = os.path.join(base, f"p{n:02d}")
        os.makedirs(pdir, exist_ok=True)
        t0 = time.time()
        i = 0
        stamps = []
        while time.time() - t0 < BURST_SECONDS:
            ok, f = cap.read()
            if not ok:
                continue
            ts = time.time()
            cv2.imwrite(os.path.join(pdir, f"f{i:03d}.jpg"), f,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            stamps.append(ts)
            i += 1
            mostrar("GRABANDO", (0, 0, 255))
            time.sleep(max(0.0, 1.0 / BURST_FPS - (time.time() - ts)))
        with open(os.path.join(pdir, "meta.json"), "w") as fh:
            json.dump({
                "code": code,
                "species": args.species,
                "presentation": n,
                "n_frames": i,
                "distance_cm": args.distance_cm,
                "holder": args.holder,
                "camera_index": args.cam,
                "resolution": [w, h],
                "delay_s": args.delay,
                "modo": "auto" if args.auto else "manual",
                "started_at": datetime.fromtimestamp(t0).isoformat(),
                "frame_timestamps": stamps,
            }, fh, indent=2)
        print(f"grabada p{n:02d}  ({i} frames)")
        return pdir

    while True:
        k = mostrar()
        if k is None:
            print("frame perdido")
            continue
        if k in (ord("q"), 27):
            break
        if k == ord("d") and last_dir:
            os.system(f'rm -rf "{last_dir}"')
            print(f"descartada {last_dir}")
            n -= 1
            last_dir = None
            continue
        if k == ord(" "):
            repeticiones = args.auto if args.auto else 1
            for j in range(repeticiones):
                if args.delay > 0:
                    cuenta_regresiva(args.delay)
                last_dir = grabar()
                n += 1
                if j < repeticiones - 1:
                    hablar("sal y vuelve a entrar")
                    t0 = time.time()
                    while time.time() - t0 < args.gap:
                        mostrar("SAL Y VUELVE A ENTRAR", (0, 200, 255))
            if args.auto:
                hablar("listo")
                print(f"--- {repeticiones} presentaciones grabadas ---")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Listo. {n - 1} presentaciones en {base}")


if __name__ == "__main__":
    main()
