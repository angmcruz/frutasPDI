# module4_classification.py
import cv2
import numpy as np
import config


def clasificar_madurez_por_espacio(caracteristicas, espacio_color="HSV"):
    if espacio_color == "RGB":
        return clasificar_madurez_rgb(caracteristicas)
    elif espacio_color == "HSV":
        return clasificar_madurez_hsv(caracteristicas)
    elif espacio_color == "LAB":
        return clasificar_madurez_lab(caracteristicas)
    else:
        return {
            "estado": "Desconocido", 
            "etiqueta": "Espacio no soportado",
            "confianza": 0, 
            "regla_activada": "Espacio no soportado",
            "color_bgr": (128, 128, 128),
            "color_hex": "#808080"
        }

def clasificar_madurez_hsv(caracteristicas):
    pct_verde = caracteristicas.get("pct_verde", 0.0)
    pct_amarillo = caracteristicas.get("pct_amarillo", 0.0)
    pct_marron = caracteristicas.get("pct_marron", 0.0)
    pct_naranja = caracteristicas.get("pct_naranja", 0.0)
    h_prom = caracteristicas.get("H_promedio", 0.0)
    pct_marron_oscuro = pct_marron + pct_naranja

    t_hsv = config.BANANO_THRESHOLDS["HSV"]
    t_inm = t_hsv["INMADURO"]
    t_sob = t_hsv["SOBREMADURO"]
    t_mad = t_hsv["MADURO"]

    # INMADURO
    if pct_verde >= t_inm["PCT_VERDE_MIN"] or (h_prom >= t_inm["HUE_VERDE_MIN"] and pct_verde > t_inm["PCT_VERDE_SECUNDARIO_MIN"]):
        confianza = min(99.0, max(75.0, pct_verde + (h_prom - t_inm["HUE_BASE_CONF"])))
        return {
            "estado": "Inmaduro",
            "etiqueta": "Inmaduro (Verde)",
            "color_bgr": (46, 204, 113),
            "color_hex": "#2ECC71",
            "regla_activada": f"HSV: %Verde={pct_verde:.1f}%, H={h_prom:.1f}°",
            "confianza": round(confianza, 1)
        }
    # SOBREMADURO
    elif (pct_marron_oscuro >= t_sob["PCT_MARRON_NARANJA_MIN"] and pct_verde < t_sob["PCT_VERDE_MAX"]) or (pct_amarillo < t_sob["PCT_AMARILLO_MAX"] and pct_verde < t_sob["PCT_VERDE_MAX"]):
        confianza = min(99.0, max(70.0, pct_marron_oscuro))
        return {
            "estado": "Sobremaduro",
            "etiqueta": "Sobremaduro (Marrón)",
            "color_bgr": (42, 42, 165),
            "color_hex": "#A52A2A",
            "regla_activada": f"HSV: %Marrón={pct_marron_oscuro:.1f}%",
            "confianza": round(confianza, 1)
        }
    # MADURO
    else:
        confianza = min(99.0, max(80.0, pct_amarillo + (t_mad["HUE_TOLERANCIA"] - abs(h_prom - t_mad["HUE_OPTIMO"]))))
        return {
            "estado": "Maduro",
            "etiqueta": "Maduro (Amarillo)",
            "color_bgr": (0, 215, 255),
            "color_hex": "#FFD700",
            "regla_activada": f"HSV: %Amarillo={pct_amarillo:.1f}%, H={h_prom:.1f}°",
            "confianza": round(confianza, 1)
        }

def clasificar_madurez_rgb(caracteristicas):
    r_prom = caracteristicas.get("R_promedio", 0.0)
    g_prom = caracteristicas.get("G_promedio", 0.0)
    b_prom = caracteristicas.get("B_promedio", 0.0)
    total = r_prom + g_prom + b_prom
    if total == 0:
        return {
            "estado": "Desconocido",
            "etiqueta": "Sin datos",
            "confianza": 0,
            "regla_activada": "Sin datos",
            "color_bgr": (128, 128, 128),
            "color_hex": "#808080"
        }

    pct_r = (r_prom / total) * 100
    pct_g = (g_prom / total) * 100
    pct_b = (b_prom / total) * 100

    t_sob = config.BANANO_THRESHOLDS["RGB"]["SOBREMADURO"]

    # INMADURO
    if pct_g > pct_r and pct_g > pct_b:
        confianza = min(97.0, 70.0 + (pct_g - pct_r) * 1.5)
        return {
            "estado": "Inmaduro",
            "etiqueta": "Inmaduro (Verde)",
            "color_bgr": (46, 204, 113),
            "color_hex": "#2ECC71",
            "regla_activada": f"RGB: %G={pct_g:.1f} > %R={pct_r:.1f} (domina verde)",
            "confianza": round(confianza, 1)
        }
    # SOBREMADURO
    elif pct_b >= t_sob["PCT_B_MIN"] and pct_r >= pct_g:
        confianza = min(95.0, 70.0 + (pct_b - t_sob["PCT_B_MIN"]) * 1.2)
        return {
            "estado": "Sobremaduro",
            "etiqueta": "Sobremaduro (Marrón)",
            "color_bgr": (42, 42, 165),
            "color_hex": "#A52A2A",
            "regla_activada": f"RGB: %B={pct_b:.1f} alto, %R={pct_r:.1f} > %G={pct_g:.1f}",
            "confianza": round(confianza, 1)
        }
    # MADURO 
    else:
        confianza = min(95.0, 70.0 + (pct_r - pct_b) * 0.5)
        return {
            "estado": "Maduro",
            "etiqueta": "Maduro (Amarillo)",
            "color_bgr": (0, 215, 255),
            "color_hex": "#FFD700",
            "regla_activada": f"RGB: %R={pct_r:.1f}, %B={pct_b:.1f} (amarillo)",
            "confianza": round(confianza, 1)
        }


def clasificar_madurez_lab(caracteristicas):
    a_prom = caracteristicas.get("a_promedio", 0.0)
    b_prom = caracteristicas.get("b_promedio", 0.0)
    l_prom = caracteristicas.get("L_promedio", 0.0)

    t_lab = config.BANANO_THRESHOLDS["LAB"]
    t_inm = t_lab["INMADURO"]
    t_sob = t_lab["SOBREMADURO"]
    t_mad = t_lab["MADURO"]

    # INMADURO 
    if a_prom < t_inm["A_MAX"]:
        confianza = min(97.0, 75.0 + (t_inm["A_MAX"] - a_prom) * 1.2)
        return {
            "estado": "Inmaduro",
            "etiqueta": "Inmaduro (Verde)",
            "color_bgr": (46, 204, 113),
            "color_hex": "#2ECC71",
            "regla_activada": f"LAB: a*={a_prom:.1f} < {t_inm['A_MAX']:.0f} (verde)",
            "confianza": round(confianza, 1)
        }
    # SOBREMADURO 
    elif a_prom >= t_sob["A_MIN"] and b_prom < t_sob["B_MAX"]:
        confianza = min(97.0, 75.0 + (a_prom - t_sob["A_MIN"]) * 1.0 + (t_sob["B_MAX"] - b_prom) * 0.3)
        return {
            "estado": "Sobremaduro",
            "etiqueta": "Sobremaduro (Marrón)",
            "color_bgr": (42, 42, 165),
            "color_hex": "#A52A2A",
            "regla_activada": f"LAB: a*={a_prom:.1f} alto, b*={b_prom:.1f} bajo (marrón)",
            "confianza": round(confianza, 1)
        }
    # MADURO 
    else:
        confianza = min(95.0, 70.0 + (b_prom - t_mad["B_REF"]) * 0.4)
        return {
            "estado": "Maduro",
            "etiqueta": "Maduro (Amarillo)",
            "color_bgr": (0, 215, 255),
            "color_hex": "#FFD700",
            "regla_activada": f"LAB: b*={b_prom:.1f} (amarillo), a*={a_prom:.1f}",
            "confianza": round(confianza, 1)
        }


def generar_visualizacion_comparativa(imagen_bgr, bbox, resultados_por_espacio):
    h, w = imagen_bgr.shape[:2]
    imagen_anotada = imagen_bgr.copy()
    
    if bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(imagen_anotada, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
  
    panel_height = 130
    panel_y = h - panel_height
    overlay = imagen_anotada.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, imagen_anotada, 0.3, 0, imagen_anotada)
    cv2.line(imagen_anotada, (0, panel_y), (w, panel_y), (255, 255, 255), 1)
    cv2.putText(imagen_anotada, "COMPARACION DE ESPACIOS DE COLOR", (10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    espacios = ["RGB", "HSV", "LAB"]
    x_positions = [10, 200, 390]
    width_espacio = 180

    for i, espacio in enumerate(espacios):
        if espacio in resultados_por_espacio:
            resultado = resultados_por_espacio[espacio]
            color = resultado.get("color_bgr", (200, 200, 200))
            estado = resultado.get("estado", "Desconocido")
            confianza = resultado.get("confianza", 0)
            regla = resultado.get("regla_activada", "")
            x = x_positions[i]

            
            cv2.rectangle(imagen_anotada, (x, panel_y + 32), (x + width_espacio, panel_y + 120), (40, 40, 40), -1)
            cv2.rectangle(imagen_anotada, (x, panel_y + 32), (x + width_espacio, panel_y + 120), (80, 80, 80), 1)
            
            cv2.rectangle(imagen_anotada, (x + 5, panel_y + 38), (x + 25, panel_y + 58), color, -1)
            
            cv2.putText(imagen_anotada, espacio, (x + 30, panel_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(imagen_anotada, estado, (x + 5, panel_y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(imagen_anotada, f"Conf: {confianza:.1f}%", (x + 5, panel_y + 96),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            
            if len(regla) > 25:
                regla = regla[:25] + "..."
            cv2.putText(imagen_anotada, regla, (x + 5, panel_y + 114),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
   
    mejor_espacio = max(resultados_por_espacio.items(), key=lambda x: x[1].get("confianza", 0))
    
    if mejor_espacio:
        espacio_nombre, mejor_resultado = mejor_espacio
        color_mejor = mejor_resultado.get("color_bgr", (0, 255, 0))
        cv2.rectangle(imagen_anotada, (w - 300, panel_y + 32), (w - 10, panel_y + 120), (30, 60, 30), -1)
        cv2.rectangle(imagen_anotada, (w - 300, panel_y + 32), (w - 10, panel_y + 120), (0, 200, 0), 1)
        cv2.putText(imagen_anotada, "MEJOR:", (w - 290, panel_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(imagen_anotada, f"{espacio_nombre}", (w - 290, panel_y + 74),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(imagen_anotada, f"{mejor_resultado['estado']}", (w - 290, panel_y + 96),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_mejor, 1)
        cv2.putText(imagen_anotada, f"Conf: {mejor_resultado['confianza']:.1f}%", (w - 290, panel_y + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return imagen_anotada
