# Clasificador de Frutas por Madurez con HSV/LAB OpenCv

Clasificador automático de madurez de **banano** a partir del color, usando técnicas clásicas de Procesamiento Digital de Imágenes sin deep learning

Dado una foto con fondo controlado, el sistema segmenta la fruta, extrae características de color y la clasifica en 3 niveles.

## Requisitos

- Python 3.10+

## Uso

​```bash
python dashboard.py
​```

Abrir una imagen → **Procesar** → ver resultados y análisis. Se puede exportar la imagen anotada y guardar las métricas en CSV.

## Estructura

​```
├── src/                    # módulos del pipeline (1–4) y utilidades
├── config.py               # rutas, umbrales y rangos de color
├── dashboard.py            # dashboard unificado
├── dashboard_modulo1-4.py  # modulos
├── data_organizada/        # dataset por fruta/estado de madurez
└── outputs/                # imágenes procesadas y results.csv
​```

## Notas

- Solo soporta banano; la estructura permite agregar más frutas ampliando `config.py` y los rangos de color.
- Diseñado para imágenes con fondo claro y controlado.
 
