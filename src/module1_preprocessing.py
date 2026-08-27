import cv2
import config

def cargar_imagen(ruta_imagen):
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        print(f"No se pudo leer la imagen: {ruta_imagen}")
    return imagen_bgr

def redimensionar(imagen, tamano=config.STANDARD_SIZE):
    return cv2.resize(imagen, tamano, interpolation=cv2.INTER_AREA)

def suavizar(imagen, kernel=config.GAUSSIAN_KERNEL):
    return cv2.GaussianBlur(imagen, kernel, 0)

def ecualizar_histograma(imagen_bgr):
    """
    mejora el contraste de la imagen compensando la iluminación irregular.
    Se realiza en el espacio LAB para ecualizar solo la luminancia (canal L)
    y no distorsionar los colores originales (canales a y b)
    """
    imagen_lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)
    
    l, a, b = cv2.split(imagen_lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ecualizado = clahe.apply(l)
    
    lab_ecualizado = cv2.merge((l_ecualizado, a, b))
    
    imagen_bgr_ecualizada = cv2.cvtColor(lab_ecualizado, cv2.COLOR_LAB2BGR)
    
    return imagen_bgr_ecualizada


def convertir_espacios_color(imagen_bgr):
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    imagen_lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)
    return {"bgr": imagen_bgr, "hsv": imagen_hsv, "lab": imagen_lab}



def preprocesar_imagen(ruta_imagen):
    imagen_bgr = cargar_imagen(ruta_imagen)
    if imagen_bgr is None:
        return None

    imagen_bgr = ecualizar_histograma(imagen_bgr)
    imagen_bgr = redimensionar(imagen_bgr)
    imagen_bgr = suavizar(imagen_bgr)

    return convertir_espacios_color(imagen_bgr)
