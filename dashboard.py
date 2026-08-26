import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import colorsys
import csv
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from module1_preprocessing import preprocesar_imagen
from module2_segmentation import crear_mascara_fruta, limpiar_mascara, obtener_bounding_box
from module3_feature_extraction import extraer_caracteristicas
from module4_classification import clasificar_madurez_por_espacio
import config

# FIGMA
BG        = "#efe9db"
PANEL     = "#f5f0e4"
HEADER    = "#e9e1cf"
BORDER    = "#ded4bd"
BROWN     = "#8a6d4b"
BROWN_DK  = "#6f573a"
TEXT      = "#3a3226"
MUTED     = "#8a7f6b"
GREEN     = "#6aa84f"
GREEN_BG  = "#e7f0dd"
GREEN_DK  = "#4e7d38"
GOLD      = "#d9a520"
ORANGE    = "#cf6b2f"
MARRON    = "#6b4423"

NIVEL = {"Inmaduro": 1, "Maduro": 2, "Sobremaduro": 3}
ESTADO_COLOR = {
    "Inmaduro":    (GREEN,  GREEN_BG, GREEN_DK),
    "Maduro":      (GOLD,   "#f5eecd", "#9c7712"),
    "Sobremaduro": (MARRON, "#ece0d3", "#5a3a1e"),
}
ZONA_COLOR = {"verde": GREEN, "amarillo": GOLD, "naranja": ORANGE, "marron": MARRON}
ZONA_LABEL = {"verde": "Verde", "amarillo": "Amarillo", "naranja": "Naranja", "marron": "Marrón"}


def boton(parent, texto, comando, primario=False):
    if primario:
        b = tk.Button(parent, text=texto, command=comando, font=("Segoe UI", 10, "bold"),
                      bg=BROWN, fg="white", activebackground=BROWN_DK, activeforeground="white",
                      relief=tk.FLAT, bd=0, padx=16, pady=8, cursor="hand2")
    else:
        b = tk.Button(parent, text=texto, command=comando, font=("Segoe UI", 10),
                      bg=PANEL, fg=BROWN_DK, activebackground=HEADER, activeforeground=BROWN_DK,
                      relief=tk.SOLID, bd=1, padx=16, pady=8, cursor="hand2",
                      highlightbackground=BORDER)
        b.configure(highlightthickness=1)
    return b


def tarjeta(parent, **kw):
    return tk.Frame(parent, bg=PANEL, relief=tk.SOLID, bd=1,
                    highlightbackground=BORDER, highlightthickness=1, **kw)


class DashboardGeneral:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Frutas por Madurez")
        self.root.geometry("960x680")
        self.root.configure(bg=BG)
        self.root.minsize(900, 640)

        self.ruta = None
        self.imagen_original = None
        self.imagen_bgr = None
        self.roi = None
        self.roi_mask = None
        self.bbox = None
        self.caracteristicas = None
        self.resultado = None
        self.metricas = None
        self.tab = "resultados"

        self._construir_barra_titulo()
        self.cuerpo = tk.Frame(self.root, bg=BG)
        self.cuerpo.pack(fill=tk.BOTH, expand=True)
        self.footer = tk.Label(self.root, bg=HEADER, fg=MUTED, font=("Segoe UI", 9),
                               anchor="center", pady=8)
        self.footer.pack(fill=tk.X, side=tk.BOTTOM)

        self.mostrar_menu()

    def _construir_barra_titulo(self):
        barra = tk.Frame(self.root, bg=HEADER, height=44)
        barra.pack(fill=tk.X)
        barra.pack_propagate(False)
        puntos = tk.Frame(barra, bg=HEADER)
        puntos.pack(side=tk.LEFT, padx=16)
        for c in ("#e06c6c", "#e6b84f", "#68b06a"):
            tk.Canvas(puntos, width=13, height=13, bg=HEADER, highlightthickness=0).pack(side=tk.LEFT, padx=3)
            puntos.winfo_children()[-1].create_oval(2, 2, 12, 12, fill=c, outline="")
        self.titulo_ventana = tk.Label(barra, text="Clasificador de Frutas por Madurez",
                                       bg=HEADER, fg=TEXT, font=("Segoe UI", 11, "bold"))
        self.titulo_ventana.pack(side=tk.LEFT, padx=8)

    def _limpiar_cuerpo(self):
        for w in self.cuerpo.winfo_children():
            w.destroy()

    def _toolbar(self, parent, botones):
        tb = tk.Frame(parent, bg=BG)
        tb.pack(fill=tk.X, padx=20, pady=(16, 8))
        for texto, cmd, primario in botones:
            boton(tb, texto, cmd, primario).pack(side=tk.LEFT, padx=(0, 8))
        return tb

    # MENU 
    def mostrar_menu(self):
        self.titulo_ventana.config(text="Clasificador de Frutas por Madurez")
        self._limpiar_cuerpo()
        self._toolbar(self.cuerpo, [
            ("Abrir imagen", self.abrir_imagen, False),
            ("Limpiar", self.limpiar, False),
            ("Procesar", self.procesar, True),
        ])

        zona = tarjeta(self.cuerpo)
        zona.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        interior = tk.Frame(zona, bg=PANEL)
        interior.place(relx=0.5, rely=0.5, anchor="center")
        icono = tk.Canvas(interior, width=64, height=52, bg=PANEL, highlightthickness=0)
        icono.pack()
        icono.create_rectangle(6, 6, 58, 46, outline=GOLD, width=2)
        icono.create_oval(16, 16, 26, 26, outline=GOLD, width=2)
        icono.create_polygon(14, 42, 30, 26, 42, 38, 50, 30, 56, 42, fill=GOLD, outline="")
        tk.Label(interior, text='Arrastra una imagen aquí o haz clic para abrir',
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(pady=(8, 2))
        tk.Label(interior, text="Formatos aceptados: JPG, PNG — fondo controlado recomendado",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack()
        for w in (zona, interior):
            w.bind("<Button-1>", lambda e: self.abrir_imagen())
        for w in interior.winfo_children():
            w.bind("<Button-1>", lambda e: self.abrir_imagen())

        fila = tk.Frame(self.cuerpo, bg=BG)
        fila.pack(fill=tk.X, padx=20, pady=(8, 16))
        fila.grid_columnconfigure(0, weight=1, uniform="x")
        fila.grid_columnconfigure(1, weight=1, uniform="x")

        c_fruta = tarjeta(fila)
        c_fruta.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(c_fruta, text="Tipo de fruta", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(12, 4))
        self.combo_fruta = ttk.Combobox(c_fruta, values=["Banano"], state="readonly",
                                        font=("Segoe UI", 12))
        self.combo_fruta.set("Banano")
        self.combo_fruta.pack(fill=tk.X, padx=14, pady=(0, 14))

        c_prev = tarjeta(fila)
        c_prev.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(c_prev, text="Vista previa", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(12, 4))
        self.lbl_preview = tk.Label(c_prev, bg=PANEL, fg=MUTED, text="sin imagen",
                                    font=("Segoe UI", 10))
        self.lbl_preview.pack(pady=(0, 14))
        if self.imagen_original is not None:
            self._preview(self.imagen_original)

        estado = "esperando imagen" if self.ruta is None else "imagen lista"
        self.footer.config(text=f"Estado: {estado} | Formatos: JPG / PNG | Fruta soportada: Banano")

    def _preview(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        r = min(150 / w, 110 / h)
        img = cv2.resize(rgb, (int(w * r), int(h * r)))
        tkimg = ImageTk.PhotoImage(Image.fromarray(img))
        self.lbl_preview.config(image=tkimg, text="")
        self.lbl_preview.image = tkimg

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp")])
        if not ruta:
            return
        img = cv2.imread(ruta)
        if img is None:
            self.footer.config(text="Estado: no se pudo leer la imagen")
            return
        self.ruta = ruta
        self.imagen_original = img
        self.resultado = None
        if hasattr(self, "lbl_preview") and self.lbl_preview.winfo_exists():
            self._preview(img)
        self.footer.config(text=f"Estado: imagen lista ({os.path.basename(ruta)}) | "
                                f"Formatos: JPG / PNG | Fruta soportada: Banano")

    def limpiar(self):
        self.ruta = None
        self.imagen_original = None
        self.roi = None
        self.caracteristicas = None
        self.resultado = None
        self.mostrar_menu()

    def procesar(self):
        if self.ruta is None:
            self.footer.config(text="Estado: primero abre una imagen")
            return
        try:
            pre = preprocesar_imagen(self.ruta)
            if pre is None:
                self.footer.config(text="Estado: error al preprocesar")
                return
            self.imagen_bgr = pre["bgr"]
            mascara = limpiar_mascara(crear_mascara_fruta(pre["hsv"]))
            self.bbox = obtener_bounding_box(mascara)
            if self.bbox is None:
                self.footer.config(text="Estado: no se detectó ninguna fruta")
                return
            x, y, w, h = self.bbox
            self.roi = self.imagen_bgr[y:y + h, x:x + w]
            self.roi_mask = mascara[y:y + h, x:x + w]
            self.caracteristicas = extraer_caracteristicas(self.roi, self.roi_mask)
            self.resultado = clasificar_madurez_por_espacio(self.caracteristicas, "HSV")
            c = self.caracteristicas
            self.metricas = {
                "H": c["H_promedio"] * 2,
                "S": c["S_promedio"] / 255,
                "V": c["V_promedio"] / 255,
                "L": c["L_promedio"] * 100 / 255,
            }
            self.tab = "resultados"
            self.mostrar_resultados()
        except Exception as e:
            self.footer.config(text=f"Estado: error — {str(e)[:40]}")

  
    def _header_resultados(self, sufijo):
        self.titulo_ventana.config(text=f"Clasificador de Frutas por Madurez — {sufijo}")
        self._limpiar_cuerpo()
        self._toolbar(self.cuerpo, [
            ("←  Nueva imagen", self.limpiar, False),
            ("Exportar imagen", self.exportar_imagen, False),
            ("Guardar CSV", self.guardar_csv, True),
        ])
        tabs = tk.Frame(self.cuerpo, bg=BG)
        tabs.pack(fill=tk.X, padx=20)
        for clave, texto in [("resultados", "Resultados"), ("analisis", "Análisis de características")]:
            activo = self.tab == clave
            t = tk.Label(tabs, text=texto, bg=BG,
                         fg=BROWN_DK if activo else MUTED,
                         font=("Segoe UI", 10, "bold" if activo else "normal"),
                         padx=6, pady=6, cursor="hand2")
            t.pack(side=tk.LEFT, padx=(0, 16))
            if activo:
                tk.Frame(tabs, bg=BROWN, height=2).place(in_=t, relx=0, rely=1.0, relwidth=1.0)
            t.bind("<Button-1>", lambda e, k=clave: self._cambiar_tab(k))
        self.footer.config(text="Estado: análisis completado | Formatos: JPG / PNG | Fruta soportada: Banano")

    def _cambiar_tab(self, clave):
        if clave == self.tab or self.resultado is None:
            return
        self.tab = clave
        self.mostrar_resultados() if clave == "resultados" else self.mostrar_analisis()

    #  RESULTADOS 
    def mostrar_resultados(self):
        self._header_resultados("Resultados")
        cont = tk.Frame(self.cuerpo, bg=BG)
        cont.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        estado = self.resultado["estado"]
        acc, bg_suave, dark = ESTADO_COLOR.get(estado, (BROWN, PANEL, BROWN_DK))
        izq = tarjeta(cont)
        izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        cont_img = tk.Frame(izq, bg=PANEL)
        cont_img.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        lbl_img = tk.Label(cont_img, bg=PANEL)
        lbl_img.pack(expand=True)
        self._mostrar_en(lbl_img, self.imagen_original, 300, 300)
        badge = tk.Label(cont_img, text=f"  {estado}  ", bg=acc, fg="white",
                         font=("Segoe UI", 9, "bold"))
        badge.place(relx=0.0, rely=0.0, x=6, y=6)

        der = tk.Frame(cont, bg=BG)
        der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        card = tk.Frame(der, bg=bg_suave, relief=tk.SOLID, bd=1,
                        highlightbackground=acc, highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(card, text="✓", bg=acc, fg="white", font=("Segoe UI", 14, "bold"),
                 width=2).pack(side=tk.LEFT, padx=12, pady=12)
        txt = tk.Frame(card, bg=bg_suave)
        txt.pack(side=tk.LEFT, pady=12)
        tk.Label(txt, text=estado, bg=bg_suave, fg=dark,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(txt, text=f"Banano · nivel {NIVEL.get(estado, '?')} de 3",
                 bg=bg_suave, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w")

        grid = tk.Frame(der, bg=BG)
        grid.pack(fill=tk.X)
        m = self.metricas
        celdas = [("H promedio", f"{m['H']:.1f}°"), ("S promedio", f"{m['S']:.2f}"),
                  ("V promedio", f"{m['V']:.2f}"), ("L* promedio", f"{m['L']:.1f}")]
        for i, (etq, val) in enumerate(celdas):
            grid.grid_columnconfigure(i % 2, weight=1, uniform="m")
            cel = tarjeta(grid)
            cel.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            tk.Label(cel, text=etq, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(10, 0))
            tk.Label(cel, text=val, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=12, pady=(0, 10))

        dist = tarjeta(self.cuerpo)
        dist.pack(fill=tk.X, padx=20, pady=(0, 14))
        tk.Label(dist, text="Distribución de color en la ROI", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        for zona in ("verde", "amarillo", "naranja", "marron"):
            pct = self.caracteristicas.get(f"pct_{zona}", 0.0)
            self._barra(dist, ZONA_LABEL[zona], pct, ZONA_COLOR[zona])
        tk.Frame(dist, bg=PANEL, height=8).pack()

    def _barra(self, parent, etiqueta, pct, color):
        fila = tk.Frame(parent, bg=PANEL)
        fila.pack(fill=tk.X, padx=14, pady=3)
        tk.Label(fila, text=etiqueta, bg=PANEL, fg=TEXT, width=9, anchor="w",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        canvas = tk.Canvas(fila, height=14, bg="#e5ddc9", highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))
        tk.Label(fila, text=f"{pct:.0f}%", bg=PANEL, fg=TEXT, width=4, anchor="e",
                 font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT)

        def dibujar(_=None):
            canvas.delete("all")
            w = canvas.winfo_width()
            if w > 1:
                canvas.create_rectangle(0, 0, max(2, w * min(pct, 100) / 100), 14,
                                        fill=color, outline="")
        canvas.bind("<Configure>", dibujar)

    def _mostrar_en(self, label, bgr, maxw, maxh):
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        r = min(maxw / w, maxh / h)
        img = cv2.resize(rgb, (int(w * r), int(h * r)))
        tkimg = ImageTk.PhotoImage(Image.fromarray(img))
        label.config(image=tkimg)
        label.image = tkimg

    # ANALASIS
    def mostrar_analisis(self):
        self._header_resultados("Análisis de características")

        c_hist = tarjeta(self.cuerpo)
        c_hist.pack(fill=tk.BOTH, expand=True, padx=20, pady=(12, 8))
        tk.Label(c_hist, text="Histogramas de color por canal (ROI normalizada)",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        self._dibujar_histogramas(c_hist)

        c_zonas = tarjeta(self.cuerpo)
        c_zonas.pack(fill=tk.X, padx=20, pady=(0, 14))
        tk.Label(c_zonas, text="Distribución de zonas cromáticas", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10, "bold", "italic")).pack(anchor="w", padx=14, pady=(12, 4))
        fila = tk.Frame(c_zonas, bg=PANEL)
        fila.pack(fill=tk.X, padx=14, pady=(0, 14))
        self._dibujar_donut(fila)
        self._leyenda_zonas(fila)

    def _dibujar_histogramas(self, parent):
        hist = self.caracteristicas["histogramas"]["HSV"]
        fig = Figure(figsize=(8.4, 2.4), dpi=100, facecolor=PANEL)
        titulos = {"H": "Canal H (Hue)", "S": "Canal S (Saturation)", "V": "Canal V (Value)"}
        for i, canal in enumerate(("H", "S", "V")):
            ax = fig.add_subplot(1, 3, i + 1)
            vals = hist[canal]
            n = len(vals)
            if canal == "H":
                colores = [self._hue_rgb(j / n) for j in range(n)]
            elif canal == "S":
                colores = [(0.15 + 0.5 * j / n, 0.35 + 0.4 * j / n, 0.7) for j in range(n)]
            else:
                colores = [(0.25 + 0.55 * j / n,) * 3 for j in range(n)]
            ax.bar(range(n), vals, width=1.0, color=colores)
            ax.set_title(titulos[canal], fontsize=9, color=TEXT)
            ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        fig.patch.set_facecolor(PANEL)
        fig.tight_layout(pad=0.6)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    @staticmethod
    def _hue_rgb(t):
        # t en [0,1] mapeado sobre el rango de tono relevante (verde->amarillo->naranja->marron)
        h = (85 - t * 80) / 180.0 if t < 0.5 else (25 - (t - 0.5) * 40) / 180.0
        r, g, b = colorsys.hsv_to_rgb(max(0.02, h), 0.8, 0.85)
        return (r, g, b)

    def _dibujar_donut(self, parent):
        etiquetas, valores, colores = [], [], []
        for zona in ("verde", "amarillo", "naranja", "marron"):
            v = self.caracteristicas.get(f"pct_{zona}", 0.0)
            if v > 0:
                etiquetas.append(ZONA_LABEL[zona])
                valores.append(v)
                colores.append(ZONA_COLOR[zona])
        if not valores:
            valores, colores = [1], ["#d8cfb8"]
        fig = Figure(figsize=(2.6, 2.6), dpi=100, facecolor=PANEL)
        ax = fig.add_subplot(111)
        ax.pie(valores, colors=colores, startangle=90,
               wedgeprops=dict(width=0.42, edgecolor=PANEL))
        ax.text(0, 0, "ROI", ha="center", va="center", fontsize=12, color=MUTED)
        ax.set(aspect="equal")
        fig.patch.set_facecolor(PANEL)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, padx=(0, 20))

    def _leyenda_zonas(self, parent):
        cont = tk.Frame(parent, bg=PANEL)
        cont.pack(side=tk.LEFT, anchor="center")
        zonas = list(ZONA_COLOR.keys())
        for i, zona in enumerate(zonas):
            pct = self.caracteristicas.get(f"pct_{zona}", 0.0)
            fila = tk.Frame(cont, bg=PANEL)
            fila.grid(row=i % 2, column=i // 2, sticky="w", padx=18, pady=8)
            cv = tk.Canvas(fila, width=12, height=12, bg=PANEL, highlightthickness=0)
            cv.pack(side=tk.LEFT)
            cv.create_oval(1, 1, 11, 11, fill=ZONA_COLOR[zona], outline="")
            tk.Label(fila, text=ZONA_LABEL[zona], bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 10), width=9, anchor="w").pack(side=tk.LEFT, padx=6)
            tk.Label(fila, text=f"{pct:.0f}%", bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

    #  EXPORTAR 
    def exportar_imagen(self):
        if self.imagen_original is None or self.resultado is None:
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not ruta:
            return
        img = self.imagen_bgr.copy() if self.imagen_bgr is not None else self.imagen_original.copy()
        estado = self.resultado["estado"]
        color = self.resultado.get("color_bgr", (0, 0, 0))
        if self.bbox is not None:
            x, y, w, h = self.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), color, -1)
        cv2.putText(img, f"{estado} - nivel {NIVEL.get(estado, '?')}/3",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(ruta, img)
        self.footer.config(text=f"Estado: imagen exportada — {os.path.basename(ruta)}")

    def guardar_csv(self):
        if self.resultado is None:
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile="results.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        m, c = self.metricas, self.caracteristicas
        fila = {
            "imagen": os.path.basename(self.ruta) if self.ruta else "",
            "fruta": "Banano",
            "estado": self.resultado["estado"],
            "nivel": NIVEL.get(self.resultado["estado"], ""),
            "confianza": self.resultado.get("confianza", ""),
            "H_grados": round(m["H"], 1), "S": round(m["S"], 3),
            "V": round(m["V"], 3), "L_estrella": round(m["L"], 1),
            "pct_verde": c.get("pct_verde", 0), "pct_amarillo": c.get("pct_amarillo", 0),
            "pct_naranja": c.get("pct_naranja", 0), "pct_marron": c.get("pct_marron", 0),
        }
        nuevo = not os.path.exists(ruta)
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fila.keys()))
            if nuevo:
                w.writeheader()
            w.writerow(fila)
        self.footer.config(text=f"Estado: CSV guardado — {os.path.basename(ruta)}")


if __name__ == "__main__":
    root = tk.Tk()
    DashboardGeneral(root)
    root.mainloop()
