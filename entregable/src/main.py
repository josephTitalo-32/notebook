import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from gestos import clasificar_gesto # Importamos la función de la Fase 1

# --- CONFIGURACIÓN DEL JUEGO ---
ANCHO_JUEGO, ALTO_JUEGO = 480, 640
ANCHO_CAM, ALTO_CAM = 320, 240
ANCHO_TOTAL = ANCHO_JUEGO + ANCHO_CAM
GRAVEDAD = 0.5
SALTO = -8
VELOCIDAD = 5
HUECO = 160
FRECUENCIA_TUBERIAS = 90

# --- CONFIGURACIÓN DE MEDIAPIPE ---
MODEL_DIR = "../modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de mediapipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo descargado correctamente")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones_mp = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

class AirFlapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AirFlap - Control por Gestos")
        self.root.resizable(False, False)
        
        # Canvas principal
        self.canvas = tk.Canvas(root, width=ANCHO_TOTAL, height=ALTO_JUEGO, bg="#222222", highlightthickness=0)
        self.canvas.pack()

        # --- Cargar Imágenes del Juego ---
        try:
            self.bird_img = tk.PhotoImage(file="assets/bird.png")
            self.pipe_img_bottom = tk.PhotoImage(file="assets/pipe_larga.png")      
            self.pipe_img_top = tk.PhotoImage(file="assets/pipe_top_larga.png")  
        except tk.TclError:
            print("ERROR: Faltan imágenes en la carpeta assets.")
            raise SystemExit

        self.bird_ancho = self.bird_img.width()
        self.bird_alto = self.bird_img.height()
        self.pipe_ancho = self.pipe_img_bottom.width()

        # Elementos visuales del juego
        self.texto_score = self.canvas.create_text(ANCHO_JUEGO//2, 50, text="0", font=("Arial", 30, "bold"), fill="white")
        self.fondo_mensaje = self.canvas.create_rectangle(ANCHO_JUEGO//2 - 150, ALTO_JUEGO//2 - 60, ANCHO_JUEGO//2 + 150, ALTO_JUEGO//2 + 60, fill="black", stipple="gray50", outline="")
        self.texto_mensaje = self.canvas.create_text(ANCHO_JUEGO//2, ALTO_JUEGO//2, text="Muestra la mano abierta", font=("Arial", 16, "bold"), fill="white")
        self.canvas.itemconfig(self.fondo_mensaje, state="hidden")
        
        # Texto del gesto en el panel de la cámara
        self.texto_gesto = self.canvas.create_text(ANCHO_JUEGO + ANCHO_CAM//2, ALTO_CAM + 20, text="Gesto: NINGUNO", font=("Arial", 14, "bold"), fill="cyan")

        # Controles
        self.root.bind("<r>", lambda e: self.reiniciar())
        self.root.bind("<Escape>", lambda e: self.cerrar())

        # --- Configurar Cámara y MediaPipe ---
        self.camara = cv2.VideoCapture(0)
        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.tiempo_inicio = time.monotonic()
        self.detector = HandLandmarker.create_from_options(opciones_mp)
        
        # Variables de estado del gesto (para el efecto de flanco)
        self.gesto_actual = "CERRADA"
        self.gesto_anterior = "CERRADA"
        
        # Contadores
        self.frames_totales = 0
        self.frames_desde_tuberia = 0
        
        self.reiniciar()
        self.loop()

    def reiniciar(self):
        self.vivo = True
        self.score = 0
        self.canvas.itemconfig(self.texto_score, text="0")
        self.canvas.itemconfig(self.texto_mensaje, text="")
        self.canvas.itemconfig(self.fondo_mensaje, state="hidden")
        
        self.bird_x = 100
        self.bird_y = ALTO_JUEGO // 2
        self.bird_vy = 0
        self.tuberias = []
        self.frames_desde_tuberia = 0
        
        self.canvas.delete("tuberia")
        self.canvas.delete("bird")
        self.bird = self.canvas.create_image(self.bird_x, self.bird_y, image=self.bird_img, anchor="nw", tags="bird")

    def saltar(self):
        if self.vivo:
            self.bird_vy = SALTO

    def crear_tuberia(self):
        centro_hueco = random.randint(200, ALTO_JUEGO - 200)
        top_pipe = self.canvas.create_image(ANCHO_JUEGO, centro_hueco - HUECO//2, image=self.pipe_img_top, anchor="sw", tags="tuberia")
        bot_pipe = self.canvas.create_image(ANCHO_JUEGO, centro_hueco + HUECO//2, image=self.pipe_img_bottom, anchor="nw", tags="tuberia")
        self.tuberias.append({"top": top_pipe, "bot": bot_pipe, "puntuada": False})

    def detectar_colision(self):
        if self.bird_y < 0 or self.bird_y + self.bird_alto > ALTO_JUEGO:
            return True
            
        bird_left = self.bird_x
        bird_right = self.bird_x + self.bird_ancho
        bird_top = self.bird_y
        bird_bottom = self.bird_y + self.bird_alto

        for t in self.tuberias:
            coords_top = self.canvas.coords(t["top"])
            coords_bot = self.canvas.coords(t["bot"])
            pipe_left = coords_top[0]
            pipe_right = pipe_left + self.pipe_ancho
            
            if bird_right > pipe_left and bird_left < pipe_right:
                if bird_top < coords_top[1] or bird_bottom > coords_bot[1]:
                    return True
        return False

    def procesar_camara(self):
        ret, frame = self.camara.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self.tiempo_inicio) * 1000)
        
        resultado = self.detector.detect_for_video(imagen_mp, ts)
        
        self.gesto_actual = "CERRADA" # Si no hay mano, se asume cerrada
        
        if resultado.hand_landmarks:
            landmarks = resultado.hand_landmarks[0]
            self.gesto_actual = clasificar_gesto(landmarks)
            
            # Dibujar los puntos de la mano en el frame
            alto, ancho, _ = frame.shape
            puntos = [(int(p.x * ancho), int(p.y * alto)) for p in landmarks]
            for inicio, fin in CONEXIONES:
                cv2.line(frame, puntos[inicio], puntos[fin], (0, 255, 0), 2)
            for x, y in puntos:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Lógica de FLANCO: Solo salta si pasa de CERRADA a ABIERTA
        if self.gesto_actual == "ABIERTA" and self.gesto_anterior == "CERRADA":
            self.saltar()
        self.gesto_anterior = self.gesto_actual

        # Actualizar texto del gesto
        color = "lime" if self.gesto_actual == "ABIERTA" else "red"
        self.canvas.itemconfig(self.texto_gesto, text=f"Gesto: {self.gesto_actual}", fill=color)

        # Convertir frame de OpenCV a imagen de Tkinter (Pillow)
        frame_pequeno = cv2.resize(frame, (ANCHO_CAM, ALTO_CAM))
        frame_rgb = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        self.img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Mostrar en el canvas
        self.canvas.create_image(ANCHO_JUEGO, 0, image=self.img_tk, anchor="nw", tags="camara")

    def loop(self):
        # 1. Procesar cámara cada 3 frames (para no saturar el CPU)
        if self.frames_totales % 3 == 0:
            self.procesar_camara()

        # 2. Lógica del juego
        if self.vivo:
            self.bird_vy += GRAVEDAD
            self.bird_y += self.bird_vy
            self.canvas.coords(self.bird, self.bird_x, self.bird_y)
            
            self.frames_desde_tuberia += 1
            if self.frames_desde_tuberia >= FRECUENCIA_TUBERIAS:
                self.crear_tuberia()
                self.frames_desde_tuberia = 0
                
            for t in self.tuberias:
                self.canvas.move(t["top"], -VELOCIDAD, 0)
                self.canvas.move(t["bot"], -VELOCIDAD, 0)
                
                coords_top = self.canvas.coords(t["top"])
                if not t["puntuada"] and (coords_top[0] + self.pipe_ancho) < self.bird_x:
                    t["puntuada"] = True
                    self.score += 1
                    self.canvas.itemconfig(self.texto_score, text=str(self.score))
            
            self.tuberias = [t for t in self.tuberias if self.canvas.coords(t["top"])[0] > -self.pipe_ancho]
            for t in self.tuberias:
                if self.canvas.coords(t["top"])[0] < -self.pipe_ancho:
                    self.canvas.delete(t["top"])
                    self.canvas.delete(t["bot"])

            if self.detectar_colision():
                self.morir()

        self.frames_totales += 1
        self.root.after(16, self.loop) # ~60 FPS

    def morir(self):
        self.vivo = False
        self.canvas.itemconfig(self.fondo_mensaje, state="normal")
        self.canvas.itemconfig(self.texto_mensaje, text=f"¡Chocaste!\nPuntos: {self.score}\nPresiona 'R' para reiniciar", justify="center")

    def cerrar(self):
        self.camara.release()
        self.detector.close()
        self.root.destroy()

if __name__ == "__main__":
    import random # Movido aquí por si acaso
    root = tk.Tk()
    app = AirFlapApp(root)
    # Asegurarnos de cerrar la cámara al cerrar la ventana
    root.protocol("WM_DELETE_WINDOW", app.cerrar)
    root.mainloop()