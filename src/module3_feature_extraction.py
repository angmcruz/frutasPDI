import cv2
import numpy as np
import config


def calcular_promedios(roi_bgr, roi_hsv, roi_lab, mascara):
    b, g, r = cv2.split(roi_bgr)
    h, s, v = cv2.split(roi_hsv)
    l, a, b_lab = cv2.split(roi_lab)

    return {
        # Promedios
        "R_promedio": cv2.mean(r, mask=mascara)[0],
        "G_promedio": cv2.mean(g, mask=mascara)[0],
        "B_promedio": cv2.mean(b, mask=mascara)[0],
    
        "H_promedio": cv2.mean(h, mask=mascara)[0],
        "S_promedio": cv2.mean(s, mask=mascara)[0],
        "V_promedio": cv2.mean(v, mask=mascara)[0],
        
        "L_promedio": cv2.mean(l, mask=mascara)[0],
        "a_promedio": cv2.mean(a, mask=mascara)[0],
        "b_promedio": cv2.mean(b_lab, mask=mascara)[0],
    }


def calcular_histogramas(roi_hsv, mascara, bins=config.HIST_BINS):
    histogramas = {}
    for indice_canal, nombre_canal in zip(range(3), ("H", "S", "V")):
        hist = cv2.calcHist([roi_hsv], [indice_canal], mascara, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histogramas[nombre_canal] = hist
    return histogramas


def calcular_histogramas_rgb(roi_bgr, mascara, bins=config.HIST_BINS):
    histogramas = {}
    # Para RGB, los canales están en orden BGR en OpenCV
    canales_bgr = [("B", 0), ("G", 1), ("R", 2)]
    for nombre_canal, indice in canales_bgr:
        hist = cv2.calcHist([roi_bgr], [indice], mascara, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histogramas[nombre_canal] = hist
    return histogramas


def calcular_histogramas_lab(roi_lab, mascara, bins=config.HIST_BINS):
    histogramas = {}
    for indice_canal, nombre_canal in zip(range(3), ("L", "a", "b")):
        hist = cv2.calcHist([roi_lab], [indice_canal], mascara, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histogramas[nombre_canal] = hist
    return histogramas


def calcular_porcentajes_color(roi_hsv, mascara):
    total_pixeles_fruta = cv2.countNonZero(mascara)
    if total_pixeles_fruta == 0:
        return {f"pct_{color}": 0.0 for color in config.COLOR_RANGES_HSV}

    porcentajes = {}
    for color, (inferior, superior) in config.COLOR_RANGES_HSV.items():
        mascara_color = cv2.inRange(roi_hsv, np.array(inferior), np.array(superior))
        mascara_color = cv2.bitwise_and(mascara_color, mascara)  # solo dentro de la fruta
        pixeles_color = cv2.countNonZero(mascara_color)
        porcentajes[f"pct_{color}"] = round(100 * pixeles_color / total_pixeles_fruta, 2)

    return porcentajes


def extraer_caracteristicas(roi_bgr, roi_mask):
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

    caracteristicas = {}
    # Promedios de todos los espacios de color
    caracteristicas.update(calcular_promedios(roi_bgr, roi_hsv, roi_lab, roi_mask))
    # Porcentajes por rango de color (basado en HSV)
    caracteristicas.update(calcular_porcentajes_color(roi_hsv, roi_mask))
    # Histogramas de los tres espacios de color
    caracteristicas["histogramas"] = {
        "RGB": calcular_histogramas_rgb(roi_bgr, roi_mask),
        "HSV": calcular_histogramas(roi_hsv, roi_mask),
        "LAB": calcular_histogramas_lab(roi_lab, roi_mask)
    }

    return caracteristicas