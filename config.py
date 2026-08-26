import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed_images")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")

FRUITS = ["banano"]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp")

# MODULO 1 - Preprocesamiento
STANDARD_SIZE = (300, 300)
GAUSSIAN_KERNEL = (3, 3)

# MODULO 2 - Segmentación
BACKGROUND_HSV_LOWER = (0, 0, 150)
BACKGROUND_HSV_UPPER = (180, 60, 255)
MORPH_KERNEL_SIZE = (7, 7)

# MODULO 3 - Extracción de Características
HIST_BINS = 32  # Número de bins para histogramas

# Rangos de color HSV para clasificación de madurez de Banano
COLOR_RANGES_HSV = {
    "verde": (
        (35, 40, 40),    # Verde claro / medio
        (85, 255, 255)   # Verde oscuro
    ),
    "amarillo": (
        (16, 40, 40),    # Amarillo claro
        (34, 255, 255)   # Amarillo vibrante
    ),
    "naranja": (
        (5, 40, 40),     # Naranja / Tono bronce
        (15, 255, 255)   # Naranja oscuro
    ),
    "marron": (
        (0, 30, 20),     # Marrón / Manchas oscuras
        (12, 255, 160)   # Marrón oscuro
    )
}

# MODULO 4 - Clasificación de Madurez del Banano (Umbrales)
BANANO_THRESHOLDS = {
    "INMADURO": {
        "PCT_VERDE_MIN": 40.0,
        "HUE_VERDE_MIN": 34.0
    },
    "SOBREMADURO": {
        "PCT_MARRON_NARANJA_MIN": 50.0,
        "PCT_AMARILLO_MAX": 10.0,
        "B_LAB_MAX": 170.0
    }
}