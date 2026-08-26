import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from module1_preprocessing import preprocesar_imagen
    from module2_segmentation import segmentar_fruta
    from module3_feature_extraction import extraer_caracteristicas
    import config
    print("Módulos importados correctamente")
except Exception as e:
    print(f"Error al importar módulos: {e}")
    sys.exit(1)


class DashboardModulo3:
    def __init__(self, root):
        self.root = root
        self.root.title("Módulo 3 - Extracción de Características")
        self.root.geometry("1400x900")
        self.root.configure(bg='#e6f3ff')
        
        self.imagen_bgr = None
        self.imagen_hsv = None
        self.roi = None
        self.roi_mask = None
        self.caracteristicas = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Frame superior
        frame_top = tk.Frame(self.root, bg='#ff8c42', height=70)
        frame_top.pack(fill=tk.X)
        frame_top.pack_propagate(False)
        
        tk.Label(frame_top, text="MÓDULO 3: EXTRACCIÓN DE CARACTERÍSTICAS", 
                font=('Arial', 16, 'bold'), bg='#ff8c42', fg='white').pack(side=tk.LEFT, padx=20)
        
        btn_cargar = tk.Button(frame_top, text="Cargar Imagen", command=self.cargar,
                              font=('Arial', 11), bg='#ff8c42', fg='white', 
                              relief=tk.FLAT, padx=15, pady=5)
        btn_cargar.pack(side=tk.RIGHT, padx=5)
        
        btn_procesar = tk.Button(frame_top, text="Extraer Características", command=self.extraer,
                                font=('Arial', 11), bg='#ff8c42', fg='white',
                                relief=tk.FLAT, padx=15, pady=5)
        btn_procesar.pack(side=tk.RIGHT, padx=5)
        
        btn_guardar = tk.Button(frame_top, text="Guardar Métricas", command=self.guardar_metricas,
                               font=('Arial', 11), bg='#ff8c42', fg='white',
                               relief=tk.FLAT, padx=15, pady=5)
        btn_guardar.pack(side=tk.RIGHT, padx=5)
        
        # Información
        self.info = tk.Label(frame_top, text="Sin imagen cargada", 
                           bg='#ff8c42', fg='white', font=('Arial', 10))
        self.info.pack(side=tk.RIGHT, padx=20)
        
        # Frame principal
        frame_principal = tk.Frame(self.root, bg='#e6f3ff')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Imagen ROI y características
        frame_izquierdo = tk.Frame(frame_principal, bg='#e6f3ff')
        frame_izquierdo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # ROI Original (40% del espacio)
        contenedor_roi = tk.Frame(frame_izquierdo, bg='white', relief=tk.RAISED, bd=2)
        contenedor_roi.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.lbl_roi = tk.Label(contenedor_roi, bg='white')
        self.lbl_roi.pack(padx=20, pady=(10, 5), expand=True, fill=tk.BOTH)
        
        tk.Label(contenedor_roi, text="ROI", font=('Arial', 12, 'bold'), 
                bg='white').pack(pady=(0, 5))
        
        self.info_roi = tk.Label(contenedor_roi, text="", font=('Arial', 9), 
                               bg='white', fg='#7f8c8d')
        self.info_roi.pack(pady=(0, 10))
        
        # Frame para características (60% del espacio)
        frame_caracteristicas = tk.Frame(frame_izquierdo, bg='white', relief=tk.RAISED, bd=2)
        frame_caracteristicas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(frame_caracteristicas, text="CARACTERÍSTICAS EXTRAÍDAS", 
                font=('Arial', 12, 'bold'), bg='white', fg='#ff8c42').pack(pady=5)
        
        # Frame interno para características con grid
        frame_interno = tk.Frame(frame_caracteristicas, bg='white')
        frame_interno.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Promedios RGB
        self.frame_rgb = tk.LabelFrame(frame_interno, text="Promedios RGB", 
                                      font=('Arial', 10, 'bold'), bg='white', padx=10, pady=5)
        self.frame_rgb.grid(row=0, column=0, padx=5, pady=3, sticky='nsew')
        
        # Promedios HSV
        self.frame_hsv = tk.LabelFrame(frame_interno, text="Promedios HSV", 
                                      font=('Arial', 10, 'bold'), bg='white', padx=10, pady=5)
        self.frame_hsv.grid(row=0, column=1, padx=5, pady=3, sticky='nsew')
        
        # Promedios LAB
        self.frame_lab = tk.LabelFrame(frame_interno, text="Promedios LAB", 
                                      font=('Arial', 10, 'bold'), bg='white', padx=10, pady=5)
        self.frame_lab.grid(row=1, column=0, padx=5, pady=3, sticky='nsew')
        
        # Porcentajes
        self.frame_porcentajes = tk.LabelFrame(frame_interno, text="Porcentajes por Rango de Color", 
                                              font=('Arial', 10, 'bold'), bg='white', padx=10, pady=5)
        self.frame_porcentajes.grid(row=1, column=1, padx=5, pady=3, sticky='nsew')
        
        frame_interno.grid_rowconfigure(0, weight=1)
        frame_interno.grid_rowconfigure(1, weight=1)
        frame_interno.grid_columnconfigure(0, weight=1)
        frame_interno.grid_columnconfigure(1, weight=1)
        
        # Panel derecho - Histogramas (sin scroll)
        frame_derecho = tk.Frame(frame_principal, bg='white', relief=tk.RAISED, bd=2)
        frame_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Título
        tk.Label(frame_derecho, text="HISTOGRAMAS DE COLOR", 
                font=('Arial', 14, 'bold'), bg='white', fg='#ff8c42').pack(pady=5)
        
        # Frames para cada histograma (distribución en 3 filas iguales)
        self.hist_frames = {}
        for i, espacio in enumerate(["RGB", "HSV", "LAB"]):
            frame = tk.LabelFrame(frame_derecho, text=f"Histogramas {espacio}", 
                                 font=('Arial', 10, 'bold'), bg='white', padx=5, pady=5)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)
            self.hist_frames[espacio] = frame
            
            # Placeholder
            lbl_placeholder = tk.Label(frame, text=f"Histogramas {espacio} (sin datos)", 
                                      font=('Arial', 10), bg='white', fg='gray')
            lbl_placeholder.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Placeholder
        self.placeholder(self.lbl_roi)
        
        # Inicializar métricas vacías
        self.actualizar_metricas(None)
    
    def placeholder(self, label):
        img = Image.new('RGB', (300, 300), '#d4e6f1')
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
        ratio = min(300/w, 300/h)
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
            # Preprocesar imagen (Módulo 1)
            preprocesado = preprocesar_imagen(ruta)
            if preprocesado is None:
                self.info.config(text="Error al cargar la imagen")
                return
            
            # Segmentar (Módulo 2)
            resultado = segmentar_fruta(preprocesado["bgr"], preprocesado["hsv"])
            
            if resultado is None or resultado["roi"] is None:
                self.info.config(text="No se detectó ninguna fruta")
                return
            
            self.imagen_bgr = preprocesado["bgr"]
            self.imagen_hsv = preprocesado["hsv"]
            self.roi = resultado["roi"]
            self.roi_mask = resultado["roi_mask"]
            
            h, w = self.roi.shape[:2]
            self.info.config(text=f"ROI: {w}x{h}")
            
            # Mostrar ROI
            self.mostrar(self.lbl_roi, self.roi)
            self.info_roi.config(text=f"ROI recortada {w}x{h}")
            
            self.caracteristicas = None
            self.actualizar_metricas(None)
            
            # Limpiar histogramas
            self.limpiar_histogramas()
    
    def limpiar_histogramas(self):
        for espacio, frame in self.hist_frames.items():
            for widget in frame.winfo_children():
                widget.destroy()
            lbl_placeholder = tk.Label(frame, text=f"Histogramas {espacio} (sin datos)", 
                                      font=('Arial', 10), bg='white', fg='gray')
            lbl_placeholder.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
    
    def extraer(self):
        if self.roi is None:
            self.info.config(text="Primero carga una imagen")
            return
        
        try:
            print("\n" + "="*50)
            print("MÓDULO 3: EXTRACCIÓN DE CARACTERÍSTICAS")
            print("="*50)
            
            # Extraer características
            self.caracteristicas = extraer_caracteristicas(self.roi, self.roi_mask)
            
            print("Características extraídas:")
            for key, value in self.caracteristicas.items():
                if key != "histogramas":
                    print(f"  {key}: {value}")
            
            # Actualizar métricas
            self.actualizar_metricas(self.caracteristicas)
            
            # Mostrar histogramas de los tres espacios
            self.mostrar_histogramas(self.caracteristicas["histogramas"])
            
            self.info.config(text="Características extraídas correctamente")
            print("EXTRACCIÓN COMPLETADA")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.info.config(text=f"Error: {str(e)[:30]}")
    
    def mostrar_histogramas(self, histogramas):
        # Definir colores para cada espacio de color
        colores_espacio = {
            'RGB': {
                'R': '#e74c3c', # Rojo
                'G': '#2ecc71', # Verde
                'B': '#3498db' # Azul
            },
            'HSV': {
                'H': '#e67e22', # Naranja (representa matiz/tono)
                'S': '#8e44ad', # Púrpura (representa saturación)
                'V': '#2c3e50' # Gris oscuro (representa valor/brillo)
            },
            'LAB': {
                'L': '#95a5a6', # Gris (representa luminosidad)
                'a': '#e74c3c', # Rojo (representa canal rojo-verde)
                'b': '#3498db'# Azul (representa canal azul-amarillo)
            }
        }
        
        for espacio, hist_data in histogramas.items():
            frame = self.hist_frames[espacio]
            
            # Limpiar frame
            for widget in frame.winfo_children():
                widget.destroy()
            
            # Crear figura con tamaño ajustado al frame
            fig = Figure(figsize=(5.5, 2.2), dpi=100, facecolor='white')
            
            # Obtener canales y colores
            canales = list(hist_data.keys())
            colores = [colores_espacio[espacio][c] for c in canales]
            
            for i, (canal, hist) in enumerate(hist_data.items()):
                ax = fig.add_subplot(1, 3, i+1)
                bins = np.arange(len(hist))
                
                ax.bar(bins, hist, width=1, color=colores[i], alpha=0.7)
                ax.set_xlabel(canal, fontsize=8)
                ax.set_ylabel('Frecuencia', fontsize=7)
                
                # Títulos descriptivos según el espacio de color
                if espacio == 'HSV':
                    titulos = {
                        'H': 'H (Matiz)',
                        'S': 'S (Saturación)',
                        'V': 'V (Valor)'
                    }
                    ax.set_title(titulos.get(canal, f'Canal {canal}'), fontsize=8)
                elif espacio == 'LAB':
                    titulos = {
                        'L': 'L (Luminosidad)',
                        'a': 'a (Rojo-Verde)',
                        'b': 'b (Azul-Amarillo)'
                    }
                    ax.set_title(titulos.get(canal, f'Canal {canal}'), fontsize=8)
                else:
                    ax.set_title(f'Canal {canal}', fontsize=8)
                
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, max(hist) * 1.1 if max(hist) > 0 else 1)
                ax.tick_params(labelsize=7)
            
            fig.tight_layout()
            
            # Embed en tkinter
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def actualizar_metricas(self, caracteristicas):
        # Limpiar frames
        for frame in [self.frame_rgb, self.frame_hsv, self.frame_lab, self.frame_porcentajes]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        if caracteristicas is None:
            # Mostrar placeholders
            for frame in [self.frame_rgb, self.frame_hsv, self.frame_lab, self.frame_porcentajes]:
                tk.Label(frame, text="No hay datos", 
                        font=('Arial', 10), bg='white', fg='gray').pack(pady=5)
            return
        
        # Promedios RGB
        rgb_promedios = {
            'R': caracteristicas['R_promedio'],
            'G': caracteristicas['G_promedio'],
            'B': caracteristicas['B_promedio']
        }
        self.mostrar_promedios(self.frame_rgb, rgb_promedios)
        
        # Promedios HSV
        hsv_promedios = {
            'H': caracteristicas['H_promedio'],
            'S': caracteristicas['S_promedio'],
            'V': caracteristicas['V_promedio']
        }
        self.mostrar_promedios(self.frame_hsv, hsv_promedios)
        
        # Promedios LAB
        lab_promedios = {
            'L*': caracteristicas['L_promedio'],
            'a*': caracteristicas['a_promedio'],
            'b*': caracteristicas['b_promedio']
        }
        self.mostrar_promedios(self.frame_lab, lab_promedios)
        
        # Porcentajes
        porcentajes = {k: v for k, v in caracteristicas.items() if k.startswith('pct_')}
        
        fila = 0
        col = 0
        for nombre, valor in porcentajes.items():
            color = nombre.replace('pct_', '')
            frame_item = tk.Frame(self.frame_porcentajes, bg='white')
            frame_item.grid(row=fila, column=col, padx=5, pady=2, sticky='w')
            
            tk.Label(frame_item, text=f"{color.capitalize()}:", font=('Arial', 9), 
                    bg='white').pack(side=tk.LEFT)
            tk.Label(frame_item, text=f"{valor}%", font=('Arial', 9, 'bold'), 
                    bg='white', fg='#ff8c42').pack(side=tk.LEFT, padx=(5, 0))
            
            col += 1
            if col >= 2:
                col = 0
                fila += 1
    
    def mostrar_promedios(self, frame, promedios):
        fila = 0
        col = 0
        for nombre, valor in promedios.items():
            frame_item = tk.Frame(frame, bg='white')
            frame_item.grid(row=fila, column=col, padx=5, pady=2, sticky='w')
            
            tk.Label(frame_item, text=f"{nombre}:", font=('Arial', 9, 'bold'), 
                    bg='white').pack(side=tk.LEFT)
            tk.Label(frame_item, text=f"{valor:.2f}", font=('Arial', 9), 
                    bg='white', fg='#ff8c42').pack(side=tk.LEFT, padx=(5, 0))
            
            col += 1
            if col >= 3:
                col = 0
                fila += 1
    
    def guardar_metricas(self):
        if self.caracteristicas is None:
            self.info.config(text="Extrae las características primero")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt"), ("CSV", "*.csv")]
        )
        
        if ruta:
            try:
                with open(ruta, 'w') as f:
                    f.write("=== CARACTERÍSTICAS DE COLOR ===\n\n")
                    
                    f.write("PROMEDIOS RGB:\n")
                    f.write(f"  R: {self.caracteristicas['R_promedio']:.2f}\n")
                    f.write(f"  G: {self.caracteristicas['G_promedio']:.2f}\n")
                    f.write(f"  B: {self.caracteristicas['B_promedio']:.2f}\n\n")
                    
                    f.write("PROMEDIOS HSV:\n")
                    f.write(f"  H: {self.caracteristicas['H_promedio']:.2f}\n")
                    f.write(f"  S: {self.caracteristicas['S_promedio']:.2f}\n")
                    f.write(f"  V: {self.caracteristicas['V_promedio']:.2f}\n\n")
                    
                    f.write("PROMEDIOS LAB:\n")
                    f.write(f"  L*: {self.caracteristicas['L_promedio']:.2f}\n")
                    f.write(f"  a*: {self.caracteristicas['a_promedio']:.2f}\n")
                    f.write(f"  b*: {self.caracteristicas['b_promedio']:.2f}\n\n")
                    
                    f.write("PORCENTAJES POR RANGO:\n")
                    for key, value in self.caracteristicas.items():
                        if key.startswith('pct_'):
                            color = key.replace('pct_', '')
                            f.write(f"  {color.capitalize()}: {value}%\n")
                
                self.info.config(text="Métricas guardadas correctamente")
            except Exception as e:
                self.info.config(text=f"Error al guardar: {str(e)[:30]}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardModulo3(root)
    root.mainloop()