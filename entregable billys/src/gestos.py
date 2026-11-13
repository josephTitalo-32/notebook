def clasificar_gesto(landmarks):
    """
    Recibe los 21 landmarks de la mano.
    Devuelve "ABIERTA" si al menos 3 dedos están extendidos hacia arriba,
    "CERRADA" en caso contrario.
    """
    # Pares de (punta del dedo, articulación media) para índice, medio, anular y meñique
    pares = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extendidos = 0
    
    for tip, pip in pares:
        # Si la punta del dedo tiene una 'y' menor (más arriba) que su articulación, está extendido
        if landmarks[tip].y < landmarks[pip].y:
            extendidos += 1
            
    return "ABIERTA" if extendidos >= 3 else "CERRADA"