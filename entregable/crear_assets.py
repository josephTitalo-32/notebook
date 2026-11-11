import cv2
import numpy as np

def crear_tuberias_largas():
    # Leer la imagen original
    img = cv2.imread("assets/pipe.png", cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print("Error: No se encontró assets/pipe.png. Asegúrate de haberlo descargado.")
        return

    alto, ancho = img.shape[:2]
    
    # Asumimos que la boquilla (el borde ancho) son los primeros 35 píxeles
    # Si tu imagen es distinta, ajusta este número.
    alto_boquilla = 35 
    
    boquilla = img[:alto_boquilla, :]
    cuerpo = img[alto_boquilla:, :]
    
    # Estiramos el cuerpo del tubo para que, junto con la boquilla, llegue a 800px
    altura_necesaria = 800 - alto_boquilla
    cuerpo_estirado = cv2.resize(cuerpo, (ancho, altura_necesaria), interpolation=cv2.INTER_NEAREST)
    
    # Juntamos la boquilla con el cuerpo estirado
    tuberia_larga = np.vstack((boquilla, cuerpo_estirado))
    
    # Creamos la versión volteada para el techo
    tuberia_top = cv2.flip(tuberia_larga, 0)
    
    # Guardamos ambas imágenes
    cv2.imwrite("assets/pipe_larga.png", tuberia_larga)
    cv2.imwrite("assets/pipe_top_larga.png", tuberia_top)
    print("¡Éxito! Se crearon 'pipe_larga.png' y 'pipe_top_larga.png' en la carpeta assets.")

if __name__ == "__main__":
    crear_tuberias_largas()