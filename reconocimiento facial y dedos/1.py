qimport cv2
import os
import urllib.request

# Nombre del archivo clasificador
xml_filename = 'haarcascade_frontalface_default.xml'
# URL oficial del modelo en el repositorio de OpenCV
xml_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

# 1. Descargar el archivo XML si no existe en la carpeta actual
if not os.path.exists(xml_filename):
    print(f"Descargando {xml_filename} automáticamente desde el repositorio de OpenCV...")
    try:
        urllib.request.urlretrieve(xml_url, xml_filename)
        print("¡Descarga completada con éxito!")
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")
        exit()

# 2. Cargar el clasificador desde la carpeta local
faceClassif = cv2.CascadeClassifier(xml_filename)

if faceClassif.empty():
    print("Error: El archivo XML está dañado o no se pudo cargar correctamente.")
    exit()

# 3. Iniciar la captura de video
cap = cv2.VideoCapture(0)

print("Iniciando cámara... Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede acceder a la webcam.")
        break

    # Convertir a escala de grises para optimizar la detección
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = faceClassif.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Dibujar rectángulos en cada rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Rostro', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    cv2.imshow('Deteccion de Rostros - Python', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

# Función callback vacía para los trackbars
def nothing(x):
    pass

# Crear ventana de controles
cv2.namedWindow('Controles')
cv2.createTrackbar('H Min', 'Controles', 0, 180, nothing)
cv2.createTrackbar('H Max', 'Controles', 25, 180, nothing)
cv2.createTrackbar('S Min', 'Controles', 15, 255, nothing)
cv2.createTrackbar('S Max', 'Controles', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Controles', 40, 255, nothing)
cv2.createTrackbar('V Max', 'Controles', 255, 255, nothing)
cv2.createTrackbar('Profundidad d', 'Controles', 10, 100, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Región de Interés (ROI)
    cv2.rectangle(frame, (350, 100), (600, 350), (0, 255, 0), 2)
    roi = frame[100:350, 350:600]

    # Leer valores actuales de los trackbars
    h_min = cv2.getTrackbarPos('H Min', 'Controles')
    h_max = cv2.getTrackbarPos('H Max', 'Controles')
    s_min = cv2.getTrackbarPos('S Min', 'Controles')
    s_max = cv2.getTrackbarPos('S Max', 'Controles')
    v_min = cv2.getTrackbarPos('V Min', 'Controles')
    v_max = cv2.getTrackbarPos('V Max', 'Controles')
    min_d = cv2.getTrackbarPos('Profundidad d', 'Controles') * 100

    # Convertir ROI a HSV y aplicar máscara
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_skin = np.array([h_max, s_max, v_max], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Limpieza de imagen
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    total_dedos = 0

    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # Dibujar mano si el área es lo suficientemente grande
        if cv2.contourArea(cnt) > 2000:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

            hull_indices = cv2.convexHull(cnt, returnPoints=False)

            if hull_indices is not None and len(hull_indices) > 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull_indices)
                    if defects is not None:
                        count_defects = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            
                            # Cálculo del ángulo entre los dedos
                            a = np.linalg.norm(np.array(end) - np.array(start))
                            b = np.linalg.norm(np.array(far) - np.array(start))
                            c = np.linalg.norm(np.array(end) - np.array(far))

                            # Evitar división por cero
                            if b * c != 0:
                                angle = np.degrees(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))

                                # Ángulo agudo (< 90°) y filtro de profundidad regulable
                                if angle <= 90 and d > min_d:
                                    count_defects += 1
                                    cv2.circle(roi, far, 6, (255, 0, 0), -1)

                        if count_defects > 0:
                            total_dedos = count_defects + 1
                        else:
                            total_dedos = 1

                except Exception:
                    pass

    # Mostrar resultado en pantalla
    cv2.putText(frame, f'Dedos: {total_dedos}', (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Conteo de Dedos', frame)
    cv2.imshow('Mascara HSV (La mano debe verse blanca)', mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()