"""
JUEGO: Piedra, Papel o Tijera
Visión Computacional con OpenCV + MediaPipe

Modos:
  - Humano vs Humano (2 jugadores, pantalla dividida)
  - Humano vs PC (PC elige aleatoriamente)

Rondas: mejor de 1, 3 o 5
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import random

# ============================================================
# 1) CONFIGURACIÓN DEL MODELO
# ============================================================
MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo descargado correctamente")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

GESTOS_VALIDOS = ["piedra", "papel", "tijera"]
NOMBRE_GESTO = {"piedra": "PIEDRA", "papel": "PAPEL", "tijera": "TIJERA"}


# ============================================================
# 2) DETECCIÓN DE GESTOS
# ============================================================
def fingers_extended(landmarks):
    """Devuelve [indice, medio, anular, meñique] extendidos (True/False).
    En coordenadas de imagen, y crece hacia abajo; un dedo extendido
    tiene la punta por encima (y menor) de la articulación PIP."""
    result = []
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        result.append(tip.y < pip.y)
    return result


def detect_gesture(landmarks):
    """Devuelve 'piedra', 'papel', 'tijera' o None."""
    fingers = fingers_extended(landmarks)
    index, middle, ring, pinky = fingers
    count = sum(fingers)

    # Tijera: índice y medio extendidos, anular y meñique plegados
    if index and middle and not ring and not pinky:
        return "tijera"
    # Papel: 3 o 4 dedos extendidos (mano abierta)
    if count >= 3:
        return "papel"
    # Piedra: puño cerrado
    if count == 0:
        return "piedra"
    return None


def evaluate_round(p1, p2):
    """Devuelve 'p1', 'p2', 'empate' o None si inválido."""
    if p1 is None or p2 is None:
        return None
    if p1 == p2:
        return "empate"
    vence_a = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
    return "p1" if vence_a[p1] == p2 else "p2"


# ============================================================
# 3) DIBUJO
# ============================================================
def dibujar_mano(frame, landmarks, color_lineas=(0, 255, 0),
                 color_puntos=(0, 0, 255)):
    alto, ancho, _ = frame.shape
    puntos = [(int(p.x * ancho), int(p.y * alto)) for p in landmarks]
    for ini, fin in CONEXIONES:
        cv2.line(frame, puntos[ini], puntos[fin], color_lineas, 2)
    for (x, y) in puntos:
        cv2.circle(frame, (x, y), 5, color_puntos, -1)


def get_hand_center(landmarks, w, h):
    return (int(np.mean([p.x for p in landmarks]) * w),
            int(np.mean([p.y for p in landmarks]) * h))


def draw_text_center(frame, text, y, scale, color, thickness=2):
    ancho = frame.shape[1]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (ancho - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


def draw_menu(frame, substate, mode_sel):
    alto, ancho, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (ancho, alto), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    if substate == "mode":
        draw_text_center(frame, "PIEDRA, PAPEL O TIJERA", 130, 1.4, (0, 255, 255), 3)
        draw_text_center(frame, "Selecciona el modo de juego", 220, 0.9, (255, 255, 255), 2)
        draw_text_center(frame, "[ H ]   Humano vs Humano", 320, 1.0, (0, 255, 0), 2)
        draw_text_center(frame, "[ P ]   Humano vs PC", 390, 1.0, (255, 200, 0), 2)
        draw_text_center(frame, "[ Q ]   Salir", 500, 0.7, (180, 180, 180), 2)
    else:
        mode_str = "Humano vs Humano" if mode_sel == "hh" else "Humano vs PC"
        draw_text_center(frame, f"Modo: {mode_str}", 130, 1.0, (255, 255, 255), 2)
        draw_text_center(frame, "Selecciona el numero de rondas", 220, 0.9, (255, 255, 255), 2)
        draw_text_center(frame, "[ 1 ]   Al mejor de 1", 320, 1.0, (0, 255, 0), 2)
        draw_text_center(frame, "[ 3 ]   Al mejor de 3", 390, 1.0, (0, 255, 0), 2)
        draw_text_center(frame, "[ 5 ]   Al mejor de 5", 460, 1.0, (0, 255, 0), 2)
        draw_text_center(frame, "ESC para volver", 550, 0.7, (180, 180, 180), 2)


def draw_hands_and_gestures(frame, hands, show_labels=True):
    alto, ancho, _ = frame.shape
    for side, (landmarks, gesture) in hands.items():
        color_lineas = (0, 255, 0) if side == "left" else (255, 150, 0)
        dibujar_mano(frame, landmarks, color_lineas=color_lineas)
        if show_labels and gesture:
            cx, cy = get_hand_center(landmarks, ancho, alto)
            txt = NOMBRE_GESTO[gesture]
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(frame, txt, (cx - tw // 2, cy - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)


def get_sides(game, hands):
    """Devuelve (lado_jugador1, lado_jugador2).
    - HH: P1 siempre izquierda, P2 siempre derecha.
    - PC: el lado donde esté la mano es P1; el otro es PC."""
    if game.mode == "hh":
        return "left", "right"
    # Modo PC
    if "right" in hands and "left" not in hands:
        return "right", "left"
    if "left" in hands and "right" in hands:
        # ambos: preferir el que tenga gesto válido
        if hands["left"][1] is not None and hands["right"][1] is None:
            return "left", "right"
        if hands["right"][1] is not None and hands["left"][1] is None:
            return "right", "left"
        return "left", "right"
    return "left", "right"


def draw_game(frame, game, hands):
    alto, ancho, _ = frame.shape
    mid = ancho // 2

    player_side, pc_side = get_sides(game, hands)

    def side_x(side):
        return 20 if side == "left" else mid + 20

    # Barra superior
    cv2.rectangle(frame, (0, 0), (ancho, 70), (30, 30, 30), -1)
    # Divisor central
    cv2.line(frame, (mid, 70), (mid, alto - 50), (100, 100, 100), 2)

    # Nombres
    if game.mode == "hpc":
        p1_name, p2_name = "TU", "PC"
    else:
        p1_name, p2_name = "JUGADOR 1", "JUGADOR 2"
    cv2.putText(frame, p1_name, (side_x(player_side), 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, p2_name, (side_x(pc_side), 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2)

    # Puntajes
    p1_sx = (mid // 2 - 20) if player_side == "left" else (mid + mid // 2 - 20)
    p2_sx = (mid // 2 - 20) if pc_side == "left" else (mid + mid // 2 - 20)
    cv2.putText(frame, str(game.scores[0]), (p1_sx, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, str(game.scores[1]), (p2_sx, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 180, 0), 3)

    # Info inferior
    info = f"Ronda {game.round_num}  |  Al mejor de {game.best_of}"
    draw_text_center(frame, info, alto - 15, 0.7, (200, 200, 200), 2)

    # Manos (solo etiquetas en waiting/countdown)
    show_labels = game.state in ("waiting", "countdown")
    draw_hands_and_gestures(frame, hands, show_labels=show_labels)

    # Overlay según estado
    if game.state == "waiting":
        if game.mode == "hpc":
            if not hands:
                msg = "Muestra tu mano en la zona izquierda"
            else:
                msg = "Listo! Presiona ESPACIO"
        else:
            if "left" not in hands and "right" not in hands:
                msg = "Muestren sus manos"
            elif "left" not in hands:
                msg = "Falta JUGADOR 1 (izquierda)"
            elif "right" not in hands:
                msg = "Falta JUGADOR 2 (derecha)"
            else:
                msg = "Listo! Presiona ESPACIO"
        draw_text_center(frame, msg, alto // 2 + 40, 0.9, (255, 255, 255), 2)
        draw_text_center(frame, "ESC para volver al menu",
                         alto // 2 + 90, 0.6, (150, 150, 150), 1)

    elif game.state == "countdown":
        elapsed = time.time() - game.state_start
        if elapsed < 0.75:
            num, col = "3", (0, 100, 255)
        elif elapsed < 1.50:
            num, col = "2", (0, 100, 255)
        elif elapsed < 2.25:
            num, col = "1", (0, 100, 255)
        else:
            num, col = "YA!", (0, 255, 0)
        draw_text_center(frame, num, alto // 2 + 70, 4.0, col, 8)
        draw_text_center(frame, "Haz tu jugada!",
                         alto // 2 + 140, 0.8, (255, 255, 255), 2)

    elif game.state == "result":
        if game.result is None:
            draw_text_center(frame, "JUGADA NO VALIDA",
                             alto // 2 + 40, 1.3, (0, 0, 255), 3)
            draw_text_center(frame, "Repitiendo ronda...",
                             alto // 2 + 90, 0.7, (200, 200, 200), 2)
        else:
            g1 = NOMBRE_GESTO[game.p1_gesture] if game.p1_gesture else "???"
            g2 = NOMBRE_GESTO[game.p2_gesture] if game.p2_gesture else "???"

            x1 = (mid // 2 - 80) if player_side == "left" else (mid + mid // 2 - 80)
            x2 = (mid // 2 - 80) if pc_side == "left" else (mid + mid // 2 - 80)
            cv2.putText(frame, g1, (x1, alto // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, g2, (x2, alto // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 180, 0), 3)

            if game.result == "p1":
                msg = "GANA JUGADOR 1" if game.mode == "hh" else "GANASTE!"
                col = (0, 255, 0)
            elif game.result == "p2":
                msg = "GANA JUGADOR 2" if game.mode == "hh" else "GANA LA PC"
                col = (255, 180, 0)
            else:
                msg, col = "EMPATE", (255, 255, 0)
            draw_text_center(frame, msg, alto // 2 + 90, 1.4, col, 3)

    elif game.state == "gameover":
        if game.scores[0] > game.scores[1]:
            winner = "JUGADOR 1" if game.mode == "hh" else "TU"
            col = (0, 255, 0)
        else:
            winner = "JUGADOR 2" if game.mode == "hh" else "PC"
            col = (255, 180, 0)
        draw_text_center(frame, f"¡GANA {winner}!", alto // 2 - 40, 1.6, col, 4)
        draw_text_center(frame, f"Marcador: {game.scores[0]} - {game.scores[1]}",
                         alto // 2 + 40, 1.0, (255, 255, 255), 2)
        draw_text_center(frame, "ESPACIO para volver al menu",
                         alto // 2 + 120, 0.7, (200, 200, 0), 2)


# ============================================================
# 4) LÓGICA DEL JUEGO
# ============================================================
class Game:
    def __init__(self, mode, best_of):
        self.mode = mode          # "hh" | "hpc"
        self.best_of = best_of
        self.needed = best_of // 2 + 1
        self.scores = [0, 0]
        self.round_num = 1
        self.state = "waiting"    # waiting | countdown | result | gameover
        self.state_start = time.time()
        self.p1_gesture = None
        self.p2_gesture = None
        self.last_p1 = None
        self.last_p2 = None
        self.result = None

    def start_countdown(self):
        self.state = "countdown"
        self.state_start = time.time()
        self.p1_gesture = None
        self.p2_gesture = None
        self.last_p1 = None
        self.last_p2 = None
        self.result = None

    def update(self, hands):
        now = time.time()
        elapsed = now - self.state_start

        if self.state == "countdown":
            # Guardamos el último gesto válido visto
            if self.mode == "hpc":
                for side in ("left", "right"):
                    if side in hands and hands[side][1] is not None:
                        self.last_p1 = hands[side][1]
                        break
            else:
                if "left" in hands and hands["left"][1] is not None:
                    self.last_p1 = hands["left"][1]
                if "right" in hands and hands["right"][1] is not None:
                    self.last_p2 = hands["right"][1]

            if elapsed >= 3.0:
                self.p1_gesture = self.last_p1
                if self.mode == "hpc":
                    self.p2_gesture = random.choice(GESTOS_VALIDOS)
                else:
                    self.p2_gesture = self.last_p2

                self.result = evaluate_round(self.p1_gesture, self.p2_gesture)
                if self.result == "p1":
                    self.scores[0] += 1
                elif self.result == "p2":
                    self.scores[1] += 1

                self.state = "result"
                self.state_start = now

        elif self.state == "result":
            if elapsed >= 3.0:
                if self.result is None:
                    # Ronda inválida: se repite
                    self.state = "waiting"
                elif self.is_over():
                    self.state = "gameover"
                else:
                    self.round_num += 1
                    self.state = "waiting"
                self.state_start = now

    def is_over(self):
        return self.scores[0] >= self.needed or self.scores[1] >= self.needed


# ============================================================
# 5) PROCESAMIENTO DE MANOS
# ============================================================
def process_hands(resultado, frame_width):
    hands = {}
    if resultado.hand_landmarks:
        for landmarks in resultado.hand_landmarks:
            cx = np.mean([p.x for p in landmarks]) * frame_width
            side = "left" if cx < frame_width / 2 else "right"
            gesture = detect_gesture(landmarks)
            if side not in hands:
                hands[side] = (landmarks, gesture)
            elif hands[side][1] is None and gesture is not None:
                hands[side] = (landmarks, gesture)
    return hands


# ============================================================
# 6) LOOP PRINCIPAL
# ============================================================
def main():
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("ERROR: no se pudo abrir la camara")
        return

    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    app_state = "menu"        # menu | game
    menu_sub = "mode"         # mode | rounds
    selected_mode = None
    game = None

    with HandLandmarker.create_from_options(opciones) as detector:
        t0 = time.monotonic()

        while True:
            ret, frame = camara.read()
            if not ret:
                print("Error al capturar imagen.")
                break

            # (Sin flip para que las dos zonas coincidan con los jugadores)
            alto, ancho, _ = frame.shape

            # --- Detección ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.monotonic() - t0) * 1000)
            resultado = detector.detect_for_video(imagen_mp, ts)
            hands = process_hands(resultado, ancho)

            # --- Render ---
            if app_state == "menu":
                draw_menu(frame, menu_sub, selected_mode)
            else:
                game.update(hands)
                draw_game(frame, game, hands)

            cv2.imshow("Piedra, Papel o Tijera - MediaPipe", frame)

            # --- Teclado ---
            key = cv2.waitKey(1) & 0xFF

            if app_state == "menu":
                if menu_sub == "mode":
                    if key == ord('h'):
                        selected_mode = "hh"
                        menu_sub = "rounds"
                    elif key == ord('p'):
                        selected_mode = "hpc"
                        menu_sub = "rounds"
                    elif key in (ord('q'), 27):
                        break
                elif menu_sub == "rounds":
                    if key in (ord('1'), ord('3'), ord('5')):
                        game = Game(selected_mode, int(chr(key)))
                        app_state = "game"
                    elif key == 27:
                        menu_sub = "mode"
                        selected_mode = None
                    elif key == ord('q'):
                        break
            else:
                if key == ord('q'):
                    break
                elif key == 27:
                    app_state = "menu"
                    menu_sub = "mode"
                    selected_mode = None
                    game = None
                elif key == ord(' '):
                    if game.state == "waiting":
                        if game.mode == "hpc" and hands:
                            game.start_countdown()
                        elif game.mode == "hh" and "left" in hands and "right" in hands:
                            game.start_countdown()
                    elif game.state == "gameover":
                        app_state = "menu"
                        menu_sub = "mode"
                        selected_mode = None
                        game = None

    camara.release()
    cv2.destroyAllWindows()
    print("Programa finalizado")


if __name__ == "__main__":
    main()