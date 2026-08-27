import cv2
import numpy as np
import config


def crear_mascara_fruta(imagen_hsv):
    fondo = cv2.inRange(
        imagen_hsv,
        np.array(config.BACKGROUND_HSV_LOWER),
        np.array(config.BACKGROUND_HSV_UPPER),
    )
    mascara_fruta = cv2.bitwise_not(fondo)
    return mascara_fruta


def limpiar_mascara(mascara):
    kernel_morfologico = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
    mascara_limpia = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_morfologico)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_OPEN, kernel_morfologico)
    kernel_erosion = np.ones((5, 5), np.uint8)
    mascara_limpia = cv2.erode(mascara_limpia, kernel_erosion, iterations=2)
    return mascara_limpia


def obtener_bounding_box(mascara):
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return None
    contorno_mayor = max(contornos, key=cv2.contourArea)
    return cv2.boundingRect(contorno_mayor)


def segmentar_fruta(imagen_bgr, imagen_hsv):
    """pipeline completo del Módulo 2 para una imagen ya 
    preprocesada
    """
    mascara = crear_mascara_fruta(imagen_hsv)
    mascara = limpiar_mascara(mascara)

    bbox = obtener_bounding_box(mascara)
    if bbox is None:
        return {"mascara": mascara, "bbox": None, "roi": None, "roi_mask": None}

    x, y, w, h = bbox
    roi = imagen_bgr[y:y + h, x:x + w]
    roi_mask = mascara[y:y + h, x:x + w]

    return {"mascara": mascara, "bbox": bbox, "roi": roi, "roi_mask": roi_mask}
