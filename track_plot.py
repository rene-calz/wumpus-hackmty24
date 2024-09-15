from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Para graficar la trayectoria
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")

# Ruta del video
video_path = "C:/Users/areba/OneDrive/Documentos/hackmty2024/yolo_env/videos/prueba2.mp4"
cap = cv2.VideoCapture(video_path)

# Diccionario para almacenar el historial completo de las posiciones (x, y) por cada ID
track_history = defaultdict(lambda: [])

# Loop a través de los frames del video
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Ejecuta el seguimiento con YOLOv8 solo para personas (clase 0)
        results = model.track(frame, persist=True, classes=[0])

        # Obtén las cajas delimitadoras y los IDs de seguimiento
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualiza los resultados en el frame
        annotated_frame = results[0].plot()

        # Guarda las posiciones de cada ID
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            # Guardar la posición del centro del cuadro (x, y)
            track_history[track_id].append((float(x), float(y)))

        # Mostrar el video con las anotaciones
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Finaliza cuando el video se haya procesado completamente
        break

# Libera el video y cierra la ventana
cap.release()
cv2.destroyAllWindows()

# Guardar el historial completo de posiciones en un archivo CSV
track_data = []

# Convertir el diccionario de track_history a una lista de filas
for track_id, positions in track_history.items():
    for frame_num, (x, y) in enumerate(positions):
        track_data.append([track_id, frame_num, x, y])

# Crear un DataFrame para organizar los datos
df = pd.DataFrame(track_data, columns=['ID', 'Frame', 'X', 'Y'])

# Guardar en un archivo CSV
output_path = "track_history.csv"
df.to_csv(output_path, index=False)

# --- Graficar la trayectoria para el primer ID detectado ---

# Obtener el primer ID detectado
first_id = df['ID'].iloc[0]

# Filtrar los datos para el primer ID
first_id_data = df[df['ID'] == first_id]

# Graficar la trayectoria del primer ID
plt.figure(figsize=(8, 6))
plt.plot(first_id_data['X'], first_id_data['Y'], marker='o', linestyle='-', color='b', label=f'Trayectoria del ID {first_id}')
plt.title(f'Trayectoria de la persona con ID {first_id}')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.legend()
plt.grid(True)
plt.show()
