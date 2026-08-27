
import os
import cv2

def listar_imagenes(carpeta, extensiones):
    """Devuelve la lista de rutas completas a imágenes válidas dentro
    de una carpeta (no entra a subcarpetas). Si la carpeta no existe,
    devuelve una lista vacía en vez de lanzar un error.
    """
    if not os.path.isdir(carpeta):
        return []
    return [
        os.path.join(carpeta, nombre)
        for nombre in sorted(os.listdir(carpeta))
        if nombre.lower().endswith(extensiones)
    ]

def crear_carpeta_si_no_existe(ruta):
    """Crea una carpeta (y las carpetas padre necesarias) si no existe."""
    os.makedirs(ruta, exist_ok=True)

def guardar_imagen(ruta, imagen):
    """Guarda una imagen en disco. Avisa por consola si algo salió mal
    (por ejemplo, si la imagen es None)."""
    if imagen is None:
        print(f"  [!] Se intentó guardar una imagen vacía en: {ruta}")
        return False
    ok = cv2.imwrite(ruta, imagen)
    if not ok:
        print(f"  [!] No se pudo guardar la imagen en: {ruta}")
    return ok
