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

# MODULO 3 - Extracción de características
HIST_BINS = 32  


COLOR_RANGES_HSV = {
    "verde": (
        (35, 40, 40),    
        (85, 255, 255)   
    ),
    "amarillo": (
        (16, 40, 40),    
        (34, 255, 255)   
    ),
    "naranja": (
        (6, 40, 40),     
        (15, 255, 255)   
    ),
    "marron": (
        (0, 30, 20),     
        (5, 255, 160)    
    )
}

# MODULO 4 
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
        "SOBREMADURO": {
            "PCT_B_MIN": 15.0,   
        },
    },
    
    "LAB": {
        "INMADURO": {
            "A_MAX": 118.0,      # a* < 118 => verde
        },
        "SOBREMADURO": {
            "A_MIN": 136.0,      # a* >= 136 
            "B_MAX": 175.0,      # y b* < 175 
        },
        "MADURO": {
            "B_REF": 150.0,      # amarillo
        },
    },
}