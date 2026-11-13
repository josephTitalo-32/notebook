def verificar_dependencias():
    print("Verificando librerías para AirFlap...\n")
    
    # 1. OpenCV
    try:
        import cv2
        print(f"✅ OpenCV instalado (versión {cv2.__version__})")
    except ImportError:
        print("❌ Falta OpenCV -> pip install opencv-python")

    # 2. MediaPipe
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe instalado (versión {mp.__version__})")
    except ImportError:
        print("❌ Falta MediaPipe -> pip install mediapipe")

    # 3. Numpy
    try:
        import numpy as np
        print(f"✅ Numpy instalado (versión {np.__version__})")
    except ImportError:
        print("❌ Falta Numpy -> pip install numpy")

    # 4. Pillow (Para la cámara en Tkinter)
    try:
        from PIL import Image, ImageTk
        import PIL
        print(f"✅ Pillow instalado (versión {PIL.__version__})")
    except ImportError:
        print("❌ Falta Pillow -> pip install pillow")

    # 5. Tkinter (Viene incluido en Python)
    try:
        import tkinter as tk
        print("✅ Tkinter disponible (incluido en Python)")
    except ImportError:
        print("❌ Falta Tkinter. Si estás en Linux, instálalo con: sudo apt-get install python3-tk")

    print("\n¡Verificación terminada!")

if __name__ == "__main__":
    verificar_dependencias()