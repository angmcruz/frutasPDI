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

# Rangos de color HSV para clasificación de madurez de Banano (sin solapamiento)
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
        (6, 40, 40),     # Naranja / Tono bronce (H: 6 a 15)
        (15, 255, 255)   # Naranja oscuro
    ),
    "marron": (
        (0, 30, 20),     # Marrón / Manchas oscuras (H: 0 a 5)
        (5, 255, 160)    # Marrón oscuro
    )
}

# MODULO 4 - Clasificación de Madurez del Banano (Umbrales por Espacio de Color)
BANANO_THRESHOLDS = {
    "HSV": {
        "INMADURO": {
            "PCT_VERDE_MIN": 40.0,
            "HUE_VERDE_MIN": 34.0,
            "PCT_VERDE_SECUNDARIO_MIN": 15.0,
            "HUE_BASE_CONF": 30.0,
        },
        "SOBREMADURO": {
            "PCT_MARRON_NARANJA_MIN": 50.0,
            "PCT_VERDE_MAX": 5.0,
            "PCT_AMARILLO_MAX": 10.0,
        },
        "MADURO": {
            "HUE_OPTIMO": 22.0,
            "HUE_TOLERANCIA": 35.0,
        },
    },
    "RGB": {
        "INMADURO": {
            "PCT_G_MIN": 40.0,
        },
        "SOBREMADURO": {
            "PCT_R_MIN": 35.0,
        },
        "MADURO": {
            "RATIO_RG_B": 1.5,
        },
    },
    "LAB": {
        "INMADURO": {
            "A_MAX": 80.0,
            "B_MAX": 150.0,
        },
        "SOBREMADURO": {
            "A_MIN": 120.0,
            "B_MAX": 170.0,
        },
        "MADURO": {
            "B_MIN": 150.0,
        },
    },
}