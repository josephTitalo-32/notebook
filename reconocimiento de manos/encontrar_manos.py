import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request

# Crear la carpeta "modelo" si no existe
MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de mediapipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo descargado correctamente")

# Configurar MEDIAPIPE
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuramos el detector
opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

CONEXIONES = [
    # pulgar
    (0, 1), (1, 2), (2, 3), (3, 4),
    # indice
    (0, 5), (5, 6), (6, 7), (7, 8),
    # medio
    (5, 9), (9, 10), (10, 11), (11, 12),
    # anular
    (9, 13), (13, 14), (14, 15), (15, 16),
    # meñique
    (13, 17), (17, 18), (18, 19), (19, 20),
    # conexion de la palma
    (0, 17)
]

# Función para dibujar los landmarks
def dibujar_mano(frame, landmarks):
    alto, ancho, _ = frame.shape
    puntos = []
    # Coordenadas normalizadas a píxeles
    for punto in landmarks:
        x = int(punto.x * ancho)
        y = int(punto.y * alto)
        puntos.append((x, y))
    
    # Dibujar conexiones entre puntos
    for inicio, fin in CONEXIONES:
        cv2.line(
            frame,
            puntos[inicio],
            puntos[fin],
            (0, 255, 0),
            2
        )
        
    # Dibujar los 21 puntos
    for numero, (x, y) in enumerate(puntos):
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )
        # Mostrar el número en cada punto
        cv2.putText(
            frame,
            str(numero),
            (x + 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1
        )

# Activar la cámara web
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: no se pudo abrir la cámara")
    raise SystemExit

# Configurar Resolución
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with HandLandmarker.create_from_options(opciones) as detector:
    tiempo_inicio = time.monotonic()
    print("Reconocimiento de manos iniciado.")
    print("Presionar Q para salir.")

    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al capturar imagen.")
            break

        # Voltear imagen horizontalmente
        #frame = cv2.flip(frame, 6)
        alto, ancho, _ = frame.shape

        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Asegurar memoria contigua para numpy
        imagen_rgb = np.ascontiguousarray(imagen_rgb)
        imagen_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imagen_rgb
        )

        # Obtener tiempo en milisegundos
        timestamp_ms = int((time.monotonic() - tiempo_inicio) * 1000)

        resultado = detector.detect_for_video(
            imagen_mp,
            timestamp_ms
        )

        # Dibujar barra de título superior
        cv2.rectangle(frame, (0, 0), (ancho, 65), (40, 40, 40), -1)
        cv2.putText(frame, "Reconocimiento de manos", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Procesar manos detectadas
        if resultado.hand_landmarks:
            for indice, landmarks in enumerate(resultado.hand_landmarks):
                dibujar_mano(frame, landmarks)
                centro_x = int(np.mean([p.x for p in landmarks]) * ancho)
                centro_y = int(np.mean([p.y for p in landmarks]) * alto)

                cv2.putText(
                    frame, 
                    f"Mano {indice+1}", 
                    (centro_x, centro_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 0, 255), 
                    2
                )

        # Mostrar ventana y gestionar salida
        cv2.imshow("Vision artificial - MediaPipe", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break

camara.release()
cv2.destroyAllWindows()
print("Programa finalizado")