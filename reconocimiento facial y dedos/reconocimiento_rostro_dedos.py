import cv2
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


