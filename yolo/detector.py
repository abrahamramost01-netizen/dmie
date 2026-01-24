from ultralytics import YOLO

model = YOLO("yolo/best.pt")

def detectar_fichas(image_path: str) -> int:
    """
    Detecta fichas de dominó y calcula puntos totales.
    """
    results = model(image_path, conf=0.5, verbose=False)

    total = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])  # clase detectada
        # EJEMPLO: si entrenaste clases como "0-0", "0-1", etc.
        # Debes mapearlas a puntos
        total += CLASE_A_PUNTOS[cls]

    return total


# Ajusta esto EXACTAMENTE a tus clases
CLASE_A_PUNTOS = {
    0: 0,   # 0-0
    1: 1,   # 0-1
    2: 2,   # 0-2
    3: 3,
    # ...
    27: 12  # 6-6
}
