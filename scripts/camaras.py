#!/usr/bin/env python3
"""Lista que camaras ve OpenCV. Correr ANTES de empezar la sesion."""
import cv2

for i in range(5):
    c = cv2.VideoCapture(i)
    ok, f = c.read()
    print(i, ok, f.shape if ok else None)
    c.release()
