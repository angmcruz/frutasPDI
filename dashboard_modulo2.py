import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
try:
    from module2_segmentation import (
        crear_mascara_fruta,
        limpiar_mascara,
        obtener_bounding_box,
        segmentar_fruta
    )
    from module1_preprocessing import preprocesar_imagen
    import config
except Exception as e:
    print(f"Error al importar modulos: {e}")
    sys.exit(1)


class DashboardModulo2:
    def __init__(self, root):
        self.root = root
        self.root.title("Modulo 2 - Deteccion y Segmentacion")
        self.root.geometry("1200x750")
        self.root.configure(bg='#e6f3ff')
        
        self.imagen_bgr = None
        self.imagen_hsv = None
        self.imagen_original = None
        self.resultado_segmentacion = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # frame superior
        frame_top = tk.Frame(self.root, bg='#ff8c42', height=70)
        frame_top.pack(fill=tk.X)
        frame_top.pack_propagate(False)
        
        tk.Label(frame_top, text="MODULO 2: DETECCION Y SEGMENTACION", font=('Arial', 16, 'bold'), bg='#ff8c42', fg='white').pack(side=tk.LEFT, padx=20)
        
        btn_cargar = tk.Button(frame_top, text="Cargar Imagen", command=self.cargar, font=('Arial', 11), bg='#ff8c42', fg='white',  relief=tk.FLAT, padx=15, pady=5)
        btn_cargar.pack(side=tk.RIGHT, padx=5)
        
        btn_procesar = tk.Button(frame_top, text="Segmentar", command=self.segmentar, font=('Arial', 11), bg='#ff8c42', fg='white', relief=tk.FLAT, padx=15, pady=5)
        btn_procesar.pack(side=tk.RIGHT, padx=5)
        
        # Informacion
        self.info = tk.Label(frame_top, text="Sin imagen cargada", bg='#ff8c42', fg='white', font=('Arial', 10))
        self.info.pack(side=tk.RIGHT, padx=20)
        
        # frame imagenes (3 columnas x 2 filas)
        frame_imgs = tk.Frame(self.root, bg='#e6f3ff')
        frame_imgs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.labels = []
        self.info_labels = []
        titulos = [
            "1. Original (BGR)",
            "2. Máscara inicial (fondo invertido)",
            "3. Después de limpieza morfológica",
            "4. Contorno + Bounding Box",
            "5. ROI recortada",
            "6. ROI con máscara"
        ]
        
        for i in range(6):
            contenedor = tk.Frame(frame_imgs, bg='white', relief=tk.RAISED, bd=2)
            contenedor.grid(row=i//3, column=i%3, padx=5, pady=5, sticky='nsew')
            
            lbl = tk.Label(contenedor, bg='white')
            lbl.pack(padx=10, pady=(10, 0), expand=True, fill=tk.BOTH)
            self.labels.append(lbl)
            
            tk.Label(contenedor, text=titulos[i], font=('Arial', 10, 'bold'),  bg='white').pack(pady=(5, 0))
            
            info_lbl = tk.Label(contenedor, text="", font=('Arial', 8),  bg='white', fg='#7f8c8d')
            info_lbl.pack(pady=(0, 5))
            self.info_labels.append(info_lbl)
        
        frame_imgs.grid_rowconfigure((0, 1), weight=1)
        frame_imgs.grid_columnconfigure((0, 1, 2), weight=1)
        
        for lbl in self.labels:
            self.placeholder(lbl)
    
    def placeholder(self, label):
        img = Image.new('RGB', (250, 250), '#d4e6f1')
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img)
        label.image = tk_img
    
    def mostrar(self, label, imagen, es_bgr=True):
        """Muestra una imagen en el label"""
        if imagen is None:
            self.placeholder(label)
            return
        
        if len(imagen.shape) == 3 and imagen.shape[2] == 3:
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        elif len(imagen.shape) == 2:
            # Imagen en escala de grises -> BGR -> RGB
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = imagen
        
        h, w = img_rgb.shape[:2]
        ratio = min(250/w, 250/h)
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
            # usar preprocesamiento del modulo 1
            preprocesado = preprocesar_imagen(ruta)
            
            if preprocesado is None:
                self.info.config(text="Error al cargar la imagen")
                return
            
            self.imagen_bgr = preprocesado["bgr"]
            self.imagen_hsv = preprocesado["hsv"]
            self.imagen_original = cv2.imread(ruta)
            
            h, w = self.imagen_bgr.shape[:2]
            self.info.config(text=f"Listo: {w}x{h}")
            
            self.mostrar(self.labels[0], self.imagen_bgr)
            self.info_labels[0].config(text=f"{w}x{h}")
            
            for i in range(1, 6):
                self.placeholder(self.labels[i])
                self.info_labels[i].config(text="")
            
            self.resultado_segmentacion = None
    
    def segmentar(self):
        """Ejecuta el pipeline completo del modulo 2 mostrando cada paso"""
        if self.imagen_bgr is None:
            self.info.config(text="Primero carga una imagen")
            return
        
        try:
            print("\n" + "="*50)
            print("MODULO 2: SEGMENTACION")
            print("="*50)
            
            self.mostrar(self.labels[0], self.imagen_bgr)
            self.info_labels[0].config(text="Imagen preprocesada")
            
            print("1. crear_mascara_fruta() - Umbralizacion en HSV")
            mascara_inicial = crear_mascara_fruta(self.imagen_hsv)
            self.mostrar(self.labels[1], mascara_inicial)
            self.info_labels[1].config(text="Fondo detectado e invertido")
            
            print("2. limpiar_mascara() - Morfologia (cierre + apertura + erosion)")
            mascara_limpia = limpiar_mascara(mascara_inicial)
            self.mostrar(self.labels[2], mascara_limpia)
            self.info_labels[2].config(text="Huecos rellenos + ruido eliminado")
            
            print("3. obtener_bounding_box() - Contorno mas grande")
            bbox = obtener_bounding_box(mascara_limpia)
            
            if bbox is None:
                self.info.config(text="No se detecto ninguna fruta")
                return
            
            x, y, w, h = bbox
            print(f"   Bounding box: x={x}, y={y}, w={w}, h={h}")
            
            img_con_bbox = self.imagen_bgr.copy()
            cv2.rectangle(img_con_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            contornos, _ = cv2.findContours(
                mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contornos:
                contorno_mayor = max(contornos, key=cv2.contourArea)
                cv2.drawContours(img_con_bbox, [contorno_mayor], -1, (255, 0, 0), 2)
            
            self.mostrar(self.labels[3], img_con_bbox)
            self.info_labels[3].config(text=f"Contorno (azul) + BBox (verde) {w}x{h}")
            
            print("4. Recortando ROI")
            roi = self.imagen_bgr[y:y+h, x:x+w]
            roi_mask = mascara_limpia[y:y+h, x:x+w]
            
            self.mostrar(self.labels[4], roi)
            self.info_labels[4].config(text=f"ROI recortada {w}x{h}")
            
            roi_con_mascara = roi.copy()
            mascara_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            roi_con_mascara = cv2.addWeighted(roi, 0.7, mascara_color, 0.3, 0)
            
            self.mostrar(self.labels[5], roi_con_mascara)
            self.info_labels[5].config(text="ROI con máscara superpuesta")
            
            self.resultado_segmentacion = {
                "mascara": mascara_limpia,
                "bbox": bbox,
                "roi": roi,
                "roi_mask": roi_mask
            }

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.info.config(text=f"Error: {str(e)[:30]}")
    
    def guardar(self):
        if self.resultado_segmentacion is None:
            self.info.config(text="Segmenta la imagen primero")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        
        if ruta:
            roi = self.resultado_segmentacion["roi"]
            cv2.imwrite(ruta, roi)
            self.info.config(text=f"ROI guardada")


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardModulo2(root)
    root.mainloop()