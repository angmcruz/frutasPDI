import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from module1_preprocessing import preprocesar_imagen
    from module2_segmentation import (
        crear_mascara_fruta,
        limpiar_mascara,
        obtener_bounding_box
    )
    from module3_feature_extraction import extraer_caracteristicas
    from module4_classification import (
        clasificar_madurez_por_espacio,
        generar_visualizacion_comparativa
    )
    import config
    print("Modulos importados correctamente")
except Exception as e:
    print(f"Error al importar modulos: {e}")
    sys.exit(1)


def obtener_evidencia_espacio(caracteristicas, espacio):
    """
    Retorna los valores clave que usa cada espacio de color para clasificar
    """
    if espacio == "RGB":
        r = caracteristicas.get("R_promedio", 0)
        g = caracteristicas.get("G_promedio", 0)
        b = caracteristicas.get("B_promedio", 0)
        total = r + g + b
        if total > 0:
            pct_r = (r / total) * 100
            pct_g = (g / total) * 100
            pct_b = (b / total) * 100
        else:
            pct_r = pct_g = pct_b = 0
            
        if pct_g > pct_r and pct_g > pct_b:
            dominio = "Verde"
        elif pct_r > pct_g and pct_r > pct_b:
            dominio = "Rojo"
        else:
            dominio = "Amarillo/Mezcla"
            
        return {
            "titulo": "EVIDENCIA RGB",
            "lineas": [
                f"- R: {r:.1f} ({pct_r:.1f}%)",
                f"- G: {g:.1f} ({pct_g:.1f}%)",
                f"- B: {b:.1f} ({pct_b:.1f}%)",
                f"- Dominancia: {dominio}"
            ]
        }
    
    elif espacio == "HSV":
        h = caracteristicas.get("H_promedio", 0)
        s = caracteristicas.get("S_promedio", 0)
        v = caracteristicas.get("V_promedio", 0)
        pct_verde = caracteristicas.get("pct_verde", 0)
        pct_amarillo = caracteristicas.get("pct_amarillo", 0)
        pct_marron = caracteristicas.get("pct_marron", 0)
        pct_naranja = caracteristicas.get("pct_naranja", 0)
        
        if h < 10:
            tono = "Rojo/Naranja"
        elif h < 35:
            tono = "Amarillo"
        elif h < 85:
            tono = "Verde"
        else:
            tono = "Otros"
        
        return {
            "titulo": "EVIDENCIA HSV",
            "lineas": [
                f"- H: {h:.1f} -> {tono}",
                f"- S: {s:.1f}",
                f"- V: {v:.1f}",
                f"- Verde: {pct_verde:.1f}% | Amarillo: {pct_amarillo:.1f}%",
                f"- Marron: {pct_marron:.1f}% | Naranja: {pct_naranja:.1f}%"
            ]
        }
    
    elif espacio == "LAB":
        l = caracteristicas.get("L_promedio", 0)
        a = caracteristicas.get("a_promedio", 0)
        b_lab = caracteristicas.get("b_promedio", 0)
        
        if a < 80:
            a_interpretacion = "Verde"
        elif a < 120:
            a_interpretacion = "Neutro"
        else:
            a_interpretacion = "Rojo"
            
        if b_lab < 130:
            b_interpretacion = "Azul"
        elif b_lab < 170:
            b_interpretacion = "Neutro"
        else:
            b_interpretacion = "Amarillo"
        
        if a < 80 and b_lab < 150:
            color_dominante = "Verde"
        elif a > 120 and b_lab < 170:
            color_dominante = "Rojo/Marron"
        else:
            color_dominante = "Amarillo/Mezcla"
        
        return {
            "titulo": "EVIDENCIA LAB",
            "lineas": [
                f"- L*: {l:.1f}",
                f"- a*: {a:.1f} -> {a_interpretacion}",
                f"- b*: {b_lab:.1f} -> {b_interpretacion}",
                f"- Dominancia: {color_dominante}"
            ]
        }
    
    return {"titulo": "Sin datos", "lineas": ["- No disponible"]}


class DashboardModulo4:
    def __init__(self, root):
        self.root = root
        self.root.title("Modulo 4 - Comparacion de Espacios de Color")
        self.root.geometry("1400x900")
        self.root.configure(bg='#e6f3ff')
        
        self.imagen_original = None
        self.imagen_bgr = None
        self.imagen_hsv = None
        self.roi = None
        self.roi_mask = None
        self.bbox = None
        self.caracteristicas = None
        self.resultados_por_espacio = {}
        self.imagen_anotada = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Frame superior
        frame_top = tk.Frame(self.root, bg='#ff8c42', height=70)
        frame_top.pack(fill=tk.X)
        frame_top.pack_propagate(False)
        
        tk.Label(frame_top, text="MODULO 4: COMPARACION DE ESPACIOS DE COLOR", 
                font=('Arial', 16, 'bold'), bg='#ff8c42', fg='white').pack(side=tk.LEFT, padx=20)
        
        btn_cargar = tk.Button(frame_top, text="Cargar Imagen", command=self.cargar,
                              font=('Arial', 11), bg='#ff8c42', fg='white', 
                              relief=tk.FLAT, padx=15, pady=5)
        btn_cargar.pack(side=tk.RIGHT, padx=5)
        
        btn_clasificar = tk.Button(frame_top, text="Comparar", command=self.comparar,
                                  font=('Arial', 11), bg='#ff8c42', fg='white',
                                  relief=tk.FLAT, padx=15, pady=5)
        btn_clasificar.pack(side=tk.RIGHT, padx=5)
        
        btn_guardar = tk.Button(frame_top, text="Guardar Resultado", command=self.guardar_resultado,
                               font=('Arial', 11), bg='#ff8c42', fg='white',
                               relief=tk.FLAT, padx=15, pady=5)
        btn_guardar.pack(side=tk.RIGHT, padx=5)
        
        # Informacion
        self.info = tk.Label(frame_top, text="Sin imagen cargada", 
                           bg='#ff8c42', fg='white', font=('Arial', 10))
        self.info.pack(side=tk.RIGHT, padx=20)
        
        # Frame principal - 2 columnas
        frame_principal = tk.Frame(self.root, bg='#e6f3ff')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Columna izquierda: Imagen (40%)
        frame_imagen = tk.Frame(frame_principal, bg='#e6f3ff')
        frame_imagen.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        contenedor_imagen = tk.Frame(frame_imagen, bg='white', relief=tk.RAISED, bd=2)
        contenedor_imagen.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_original = tk.Label(contenedor_imagen, bg='white')
        self.lbl_original.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(contenedor_imagen, text="Imagen Original / Comparacion", 
                font=('Arial', 11, 'bold'), bg='white').pack(pady=5)
        
        self.info_original = tk.Label(contenedor_imagen, text="", 
                                    font=('Arial', 9), bg='white', fg='#7f8c8d')
        self.info_original.pack(pady=5)
        
        # Columna derecha: Tarjetas verticales (60%)
        frame_tarjetas_container = tk.Frame(frame_principal, bg='#e6f3ff')
        frame_tarjetas_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        contenedor_tarjetas = tk.Frame(frame_tarjetas_container, bg='white', relief=tk.RAISED, bd=2)
        contenedor_tarjetas.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(contenedor_tarjetas, text="COMPARACION DE ESPACIOS DE COLOR", 
                font=('Arial', 14, 'bold'), bg='white', fg='#ff8c42').pack(pady=10)
        
        # Frame para las 3 tarjetas verticales
        frame_tarjetas = tk.Frame(contenedor_tarjetas, bg='white')
        frame_tarjetas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        frame_tarjetas.grid_rowconfigure(0, weight=1)
        frame_tarjetas.grid_rowconfigure(1, weight=1)
        frame_tarjetas.grid_rowconfigure(2, weight=1)
        frame_tarjetas.grid_columnconfigure(0, weight=1)
        
        # Crear tarjetas
        self.tarjetas = {}
        colores_tarjeta = {
            'RGB': {'bg': '#fff5f5', 'border': '#e74c3c'},
            'HSV': {'bg': '#f5f9ff', 'border': '#f1c40f'},
            'LAB': {'bg': '#f5fff5', 'border': '#2ecc71'}
        }
        
        for i, espacio in enumerate(["RGB", "HSV", "LAB"]):
            frame_tarjeta = tk.Frame(frame_tarjetas, bg='white', relief=tk.RAISED, bd=2)
            frame_tarjeta.grid(row=i, column=0, sticky='nsew', padx=5, pady=5)
            
            # Borde superior
            frame_borde = tk.Frame(frame_tarjeta, bg=colores_tarjeta[espacio]['border'], height=4)
            frame_borde.pack(fill=tk.X)
            
            # Contenido
            frame_contenido = tk.Frame(frame_tarjeta, bg='white')
            frame_contenido.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Primera fila: Nombre + Estado + Confianza
            frame_fila1 = tk.Frame(frame_contenido, bg='white')
            frame_fila1.pack(fill=tk.X, pady=3)
            
            tk.Label(frame_fila1, text=espacio, 
                    font=('Arial', 13, 'bold'), bg='white', 
                    fg=colores_tarjeta[espacio]['border']).pack(side=tk.LEFT, padx=10)
            
            lbl_estado = tk.Label(frame_fila1, text="Sin clasificar", 
                                font=('Arial', 12, 'bold'), bg='white', fg='#7f8c8d')
            lbl_estado.pack(side=tk.LEFT, padx=20)
            
            lbl_confianza = tk.Label(frame_fila1, text="Confianza: --%", 
                                   font=('Arial', 10), bg='white', fg='#95a5a6')
            lbl_confianza.pack(side=tk.RIGHT, padx=10)
            
            # Segunda fila: Barra de confianza
            frame_barra = tk.Frame(frame_contenido, bg='white')
            frame_barra.pack(fill=tk.X, pady=3)
            
            canvas_barra = tk.Canvas(frame_barra, height=15, bg='#ecf0f1', highlightthickness=0)
            canvas_barra.pack(fill=tk.X, padx=10)
            
            # Tercera fila: Evidencia en formato lista
            tk.Frame(frame_contenido, height=1, bg='#ecf0f1').pack(fill=tk.X, pady=3)
            
            tk.Label(frame_contenido, text="EVIDENCIA", 
                    font=('Arial', 9, 'bold'), bg='white', fg='#34495e').pack(pady=2)
            
            frame_evidencia = tk.Frame(frame_contenido, bg='#f8f9fa', relief=tk.FLAT, bd=1)
            frame_evidencia.pack(fill=tk.BOTH, expand=True, pady=3)
            
            # Labels para evidencia en formato lista vertical
            lbl_evidencia_lineas = []
            for j in range(5):
                lbl = tk.Label(frame_evidencia, text="", font=('Courier', 9), 
                             bg='#f8f9fa', fg='#2c3e50', anchor='w', justify='left')
                lbl.pack(fill=tk.X, padx=10, pady=1)
                lbl_evidencia_lineas.append(lbl)
            
            self.tarjetas[espacio] = {
                'frame': frame_tarjeta,
                'lbl_estado': lbl_estado,
                'lbl_confianza': lbl_confianza,
                'canvas_barra': canvas_barra,
                'lbl_evidencia': lbl_evidencia_lineas,
                'color': colores_tarjeta[espacio]['border']
            }
        
        # Placeholder
        self.placeholder(self.lbl_original)
    
    def placeholder(self, label):
        img = Image.new('RGB', (400, 400), '#d4e6f1')
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img)
        label.image = tk_img
    
    def mostrar(self, label, imagen):
        if imagen is None:
            self.placeholder(label)
            return
        
        if len(imagen.shape) == 3 and imagen.shape[2] == 3:
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        elif len(imagen.shape) == 2:
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = imagen
        
        h, w = img_rgb.shape[:2]
        label_width = label.winfo_width() if label.winfo_width() > 10 else 400
        label_height = label.winfo_height() if label.winfo_height() > 10 else 400
        
        ratio = min(label_width/w, label_height/h)
        nuevo_w, nuevo_h = int(w*ratio), int(h*ratio)
        img_redim = cv2.resize(img_rgb, (nuevo_w, nuevo_h))
        
        img_pil = Image.fromarray(img_redim)
        tk_img = ImageTk.PhotoImage(img_pil)
        label.config(image=tk_img)
        label.image = tk_img
    
    def cargar(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.webp")]
        )
        
        if ruta:
            self.imagen_original = cv2.imread(ruta)
            if self.imagen_original is None:
                self.info.config(text="Error al cargar la imagen")
                return
            
            preprocesado = preprocesar_imagen(ruta)
            if preprocesado is None:
                self.info.config(text="Error al preprocesar la imagen")
                return
            
            self.imagen_bgr = preprocesado["bgr"]
            self.imagen_hsv = preprocesado["hsv"]
            
            mascara_inicial = crear_mascara_fruta(self.imagen_hsv)
            mascara_limpia = limpiar_mascara(mascara_inicial)
            self.bbox = obtener_bounding_box(mascara_limpia)
            
            if self.bbox is None:
                self.info.config(text="No se detecto ninguna fruta")
                return
            
            x, y, w, h = self.bbox
            self.roi = self.imagen_bgr[y:y+h, x:x+w]
            self.roi_mask = mascara_limpia[y:y+h, x:x+w]
            
            self.mostrar(self.lbl_original, self.imagen_original)
            self.info_original.config(text=f"Original {self.imagen_original.shape[1]}x{self.imagen_original.shape[0]}")
            
            self.caracteristicas = None
            self.resultados_por_espacio = {}
            self.imagen_anotada = None
            self.resetear_tarjetas()
    
    def resetear_tarjetas(self):
        for espacio, datos in self.tarjetas.items():
            datos['lbl_estado'].config(text="Sin clasificar", fg='#7f8c8d')
            datos['lbl_confianza'].config(text="Confianza: --%", fg='#95a5a6')
            datos['canvas_barra'].delete("all")
            for lbl in datos['lbl_evidencia']:
                lbl.config(text="")
    
    def actualizar_tarjeta(self, espacio, resultado):
        datos = self.tarjetas[espacio]
        
        estado = resultado['etiqueta']
        color_hex = resultado['color_hex']
        datos['lbl_estado'].config(text=estado, fg=color_hex)
        
        confianza = resultado['confianza']
        datos['lbl_confianza'].config(text=f"Confianza: {confianza:.1f}%", fg='#ff8c42')
        
        canvas = datos['canvas_barra']
        canvas.delete("all")
        ancho = canvas.winfo_width() if canvas.winfo_width() > 10 else 200
        
        if confianza >= 80:
            color_barra = '#2ecc71'
        elif confianza >= 60:
            color_barra = '#f1c40f'
        else:
            color_barra = '#e74c3c'
        
        if ancho > 10:
            canvas.create_rectangle(0, 0, (confianza/100) * ancho, 15, fill=color_barra, outline='')
            canvas.create_rectangle(0, 0, ancho, 15, outline='#bdc3c7')
            canvas.create_text(ancho/2, 7, text=f"{confianza:.1f}%", 
                             fill='white' if confianza > 50 else '#2c3e50', 
                             font=('Arial', 7, 'bold'))
    
    def actualizar_evidencia(self, espacio):
        if self.caracteristicas is None:
            return
        
        evidencia = obtener_evidencia_espacio(self.caracteristicas, espacio)
        datos = self.tarjetas[espacio]
        
        for i, linea in enumerate(evidencia['lineas']):
            if i < len(datos['lbl_evidencia']):
                datos['lbl_evidencia'][i].config(text=linea)
    
    def comparar(self):
        if self.roi is None:
            self.info.config(text="Primero carga una imagen")
            return
        
        try:
            print("\n" + "="*50)
            print("MODULO 4: COMPARACION DE ESPACIOS DE COLOR")
            print("="*50)
            
            self.caracteristicas = extraer_caracteristicas(self.roi, self.roi_mask)
            
            print("Caracteristicas extraidas:")
            for key, value in self.caracteristicas.items():
                if key != "histogramas":
                    print(f"  {key}: {value}")
            
            self.resultados_por_espacio = {}
            for espacio in ["RGB", "HSV", "LAB"]:
                resultado = clasificar_madurez_por_espacio(self.caracteristicas, espacio)
                self.resultados_por_espacio[espacio] = resultado
                print(f"\n{espacio}: {resultado['etiqueta']} (Conf: {resultado['confianza']}%)")
                print(f"  Regla: {resultado['regla_activada']}")
                
                self.actualizar_tarjeta(espacio, resultado)
                self.actualizar_evidencia(espacio)
            
            self.imagen_anotada = generar_visualizacion_comparativa(
                self.imagen_bgr,
                self.bbox,
                self.resultados_por_espacio
            )
            
            self.mostrar(self.lbl_original, self.imagen_anotada)
            self.info_original.config(text="Imagen con comparacion")
            
            self.info.config(text="Comparacion completada")
            print("\nCOMPARACION COMPLETADA")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.info.config(text=f"Error: {str(e)[:30]}")
    
    def guardar_resultado(self):
        if self.imagen_anotada is None:
            self.info.config(text="Primero realiza la comparacion")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        
        if ruta:
            cv2.imwrite(ruta, self.imagen_anotada)
            self.info.config(text=f"Resultado guardado en: {os.path.basename(ruta)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardModulo4(root)
    root.mainloop()