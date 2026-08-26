import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from module1_preprocessing import (
        cargar_imagen,
        ecualizar_histograma,
        redimensionar,
        suavizar,
        convertir_espacios_color
    )
    import config
    print("Modulos importados correctamente")
except:
    print("Error al importar modulos")
    sys.exit(1)


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Modulo 1 - Preprocesamiento")
        self.root.geometry("1100x650")
        self.root.configure(bg='#e6f3ff')
        
        self.imagen = None
        self.procesada = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        frame_top = tk.Frame(self.root, bg='#ff8c42', height=70)
        frame_top.pack(fill=tk.X)
        frame_top.pack_propagate(False)
        
        tk.Label(frame_top, text="MODULO 1: PREPROCESAMIENTO", 
                font=('Arial', 16, 'bold'), bg='#ff8c42', fg='white').pack(side=tk.LEFT, padx=20)
        
        btn_cargar = tk.Button(frame_top, text="Cargar Imagen", command=self.cargar,
                              font=('Arial', 11), bg='#ff8c42', fg='white', 
                              relief=tk.FLAT, padx=15, pady=5)
        btn_cargar.pack(side=tk.RIGHT, padx=5)
        
        btn_procesar = tk.Button(frame_top, text="Procesar", command=self.procesar,
                                font=('Arial', 11), bg='#ff8c42', fg='white',
                                relief=tk.FLAT, padx=15, pady=5)
        btn_procesar.pack(side=tk.RIGHT, padx=5)
        
        self.info = tk.Label(frame_top, text="Sin imagen cargada",  
                           bg='#ff8c42', fg='white', font=('Arial', 10))
        self.info.pack(side=tk.RIGHT, padx=20)
        
        frame_imgs = tk.Frame(self.root, bg='#e6f3ff')
        frame_imgs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.labels = []
        self.info_labels = [] 
        titulos = ["Original", "Ecualizacion", "Redimensionado", "Suavizado", "HSV", "LAB"]
        
        for i in range(6):
            contenedor = tk.Frame(frame_imgs, bg='white', relief=tk.RAISED, bd=2)
            contenedor.grid(row=i//3, column=i%3, padx=5, pady=5, sticky='nsew')
            
            lbl = tk.Label(contenedor, bg='white')
            lbl.pack(padx=10, pady=(10, 0), expand=True, fill=tk.BOTH)
            self.labels.append(lbl)
            
            tk.Label(contenedor, text=titulos[i], font=('Arial', 10, 'bold'), bg='white').pack(pady=(5, 0))
            
            info_lbl = tk.Label(contenedor, text="", font=('Arial', 8), bg='white', fg='#7f8c8d')
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
    
    def mostrar(self, label, img_bgr):
        if img_bgr is None:
            self.placeholder(label)
            return
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
            self.imagen = cargar_imagen(ruta)
            if self.imagen is not None:
                h, w = self.imagen.shape[:2]
                self.info.config(text=f"Listo: {w}x{h}")
                self.mostrar(self.labels[0], self.imagen)
                self.info_labels[0].config(text=f"{w}x{h}")
                
                for i in range(1, 6):
                    self.placeholder(self.labels[i])
                    self.info_labels[i].config(text="")
                
                self.procesada = None
    
    def procesar(self):
        if self.imagen is None:
            self.info.config(text="Primero carga una imagen")
            return
        try:
            print("\n" + "="*50)
            print("MODULO 1: PREPROCESAMIENTO")
            print("="*50)
            
            # original
            self.mostrar(self.labels[0], self.imagen)
            self.info_labels[0].config(text="Imagen original BGR")
            
            print("1. ecualizar_histograma() - CLAHE")
            img1 = ecualizar_histograma(self.imagen)
            self.mostrar(self.labels[1], img1)
            self.info_labels[1].config(text="Mejora de contraste (CLAHE)")
            
            print("2. redimensionar()")
            img2 = redimensionar(img1)
            self.mostrar(self.labels[2], img2)
            self.info_labels[2].config(text=f"{config.STANDARD_SIZE[0]}x{config.STANDARD_SIZE[1]}")
            
            print("3. suavizar() - Gaussian Blur")
            img3 = suavizar(img2)
            self.mostrar(self.labels[3], img3)
            self.info_labels[3].config(text=f"Gaussian Blur (kernel {config.GAUSSIAN_KERNEL[0]}x{config.GAUSSIAN_KERNEL[1]})")
            self.procesada = img3
            
            print("4. convertir_espacios_color()")
            espacios = convertir_espacios_color(img3)
            
            hsv = espacios["hsv"]
            h_c, s_c, v_c = cv2.split(hsv)
            
            h_vis = cv2.cvtColor(cv2.normalize(h_c, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
            s_vis = cv2.cvtColor(cv2.normalize(s_c, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
            v_vis = cv2.cvtColor(cv2.normalize(v_c, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
            
            hsv_mostrar = np.hstack([h_vis, s_vis, v_vis])
            self.mostrar(self.labels[4], hsv_mostrar)
            self.info_labels[4].config(text="Canales H, S, V separados")
            
            lab = espacios["lab"]
            l_c, a_c, b_c = cv2.split(lab)
            
            l_vis = cv2.cvtColor(cv2.normalize(l_c, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
            a_vis = cv2.cvtColor(cv2.normalize(a_c, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
            b_vis = cv2.cvtColor(cv2.normalize(b_c, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
            
            lab_mostrar = np.hstack([l_vis, a_vis, b_vis])
            self.mostrar(self.labels[5], lab_mostrar)
            self.info_labels[5].config(text="Canales L, a*, b* separados")
            
            self.info.config(text="Procesamiento completado")
            print("PROCESAMIENTO COMPLETADO")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error: {e}")
            self.info.config(text=f"Error: {str(e)[:30]}")


if __name__ == "__main__":
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()