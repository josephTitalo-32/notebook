import tkinter as tk
import random
import os

ANCHO, ALTO = 480, 640
GRAVEDAD = 0.5
SALTO = -8
VELOCIDAD = 5
HUECO = 160 
FRECUENCIA_TUBERIAS = 90 

class FlappyBirdTk:
    def __init__(self, root):
        self.root = root
        self.root.title("AirFlap - Prototipo Teclado")
        self.root.resizable(False, False)
        
        self.canvas = tk.Canvas(root, width=ANCHO, height=ALTO, bg="#4ec0ca", highlightthickness=0)
        self.canvas.pack()

        # --- CARGAR LAS IMÁGENES ESTIRADAS ---
        try:
            self.bird_img = tk.PhotoImage(file="assets/bird.png")
            self.pipe_img_bottom = tk.PhotoImage(file="assets/pipe_larga.png")      
            self.pipe_img_top = tk.PhotoImage(file="assets/pipe_top_larga.png")  
        except tk.TclError:
            print("ERROR: Asegúrate de haber ejecutado preparar_tuberias.py primero.")
            raise SystemExit

        # Dimensiones para colisiones
        self.bird_ancho = self.bird_img.width()
        self.bird_alto = self.bird_img.height()
        self.pipe_ancho = self.pipe_img_bottom.width()
        
        # UI
        self.texto_score = self.canvas.create_text(ANCHO//2, 50, text="0", font=("Arial", 30, "bold"), fill="white")
        
        # Fondo semitransparente para el Game Over
        self.fondo_mensaje = self.canvas.create_rectangle(ANCHO//2 - 150, ALTO//2 - 60, ANCHO//2 + 150, ALTO//2 + 60, fill="black", stipple="gray50", outline="")
        self.texto_mensaje = self.canvas.create_text(ANCHO//2, ALTO//2, text="Presiona ESPACIO para jugar", font=("Arial", 16, "bold"), fill="white")
        self.canvas.itemconfig(self.fondo_mensaje, state="hidden")
        
        # Controles
        self.root.bind("<space>", self.saltar)
        self.root.bind("<r>", lambda e: self.reiniciar())

        self.reiniciar()
        self.loop()

    def reiniciar(self):
        self.vivo = True
        self.score = 0
        self.canvas.itemconfig(self.texto_score, text="0")
        self.canvas.itemconfig(self.texto_mensaje, text="")
        self.canvas.itemconfig(self.fondo_mensaje, state="hidden")
        
        self.bird_x = 100
        self.bird_y = ALTO // 2
        self.bird_vy = 0
        self.tuberias = []
        self.frames_desde_tuberia = 0
        
        self.canvas.delete("tuberia")
        self.canvas.delete("bird")
        
        self.bird = self.canvas.create_image(self.bird_x, self.bird_y, image=self.bird_img, anchor="nw", tags="bird")

    def saltar(self, event=None):
        if self.vivo:
            self.bird_vy = SALTO

    def crear_tuberia(self):
        # El centro del hueco varía entre 200 y ALTO-200 para que no quede muy pegado a los bordes
        centro_hueco = random.randint(200, ALTO - 200)
        
        # Tubería superior (anclada por su esquina inferior izquierda "sw")
        top_pipe = self.canvas.create_image(ANCHO, centro_hueco - HUECO//2, image=self.pipe_img_top, anchor="sw", tags="tuberia")
        
        # Tubería inferior (anclada por su esquina superior izquierda "nw")
        bot_pipe = self.canvas.create_image(ANCHO, centro_hueco + HUECO//2, image=self.pipe_img_bottom, anchor="nw", tags="tuberia")
        
        self.tuberias.append({"top": top_pipe, "bot": bot_pipe, "puntuada": False})

    def detectar_colision(self):
        if self.bird_y < 0 or self.bird_y + self.bird_alto > ALTO:
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

    def loop(self):
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

        self.root.after(16, self.loop)

    def morir(self):
        self.vivo = False
        self.canvas.itemconfig(self.fondo_mensaje, state="normal")
        self.canvas.itemconfig(self.texto_mensaje, text=f"¡Chocaste!\nPuntos: {self.score}\nPresiona 'R' para reiniciar", justify="center")

if __name__ == "__main__":
    root = tk.Tk()
    juego = FlappyBirdTk(root)
    root.mainloop()