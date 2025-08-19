import cv2
from ultralytics import YOLO
import json
import numpy as np

# --- CONFIGURACIÓN ---
video_source = 'http://192.168.1.5:8080/video'  # O tu video, ej. 'video_parqueadero.mp4'

# Carga el modelo de detección (para carros y motos)
model = YOLO('yolov8n.pt')

# --- VARIABLES ---
try:
    with open('parqueadero_rois.json', 'r') as f:
        park_areas_coords = json.load(f)
except FileNotFoundError:
    print("Error: El archivo de ROIs 'parqueadero_rois.json' no existe.")
    print("Por favor, ejecuta 'roi_selector.py' primero.")
    exit()

# La lógica para la cámara
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: No se pudo abrir la fuente de video '{video_source}'.")
    exit()

print("Análisis de parqueadero iniciado. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al leer el frame.")
        break

    # Reiniciar el estado de los espacios de parqueo para el fotograma actual
    park_status = [False] * len(park_areas_coords)

    # Realizar la detección
    results = model(frame, conf=0.5, classes=[2, 3, 5, 7]) # Clases: carro, moto, bus, camión

    # Lógica de detección en las ROIs
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            for i, roi_points in enumerate(park_areas_coords):
                roi_np = np.array(roi_points, np.int32)
                roi_np = roi_np.reshape((-1, 1, 2))
                
                # Comprobar si el centro del vehículo está en la ROI
                if cv2.pointPolygonTest(roi_np, (center_x, center_y), False) >= 0:
                    park_status[i] = True
                    break

    # Dibujar los resultados
    annotated_frame = results[0].plot()

    # Contar cupos
    occupied_count = sum(park_status)
    total_spaces = len(park_areas_coords)
    available_spaces = total_spaces - occupied_count
    
    # Dibujar las ROIs y su estado
    for i, roi_points in enumerate(park_areas_coords):
        roi_np = np.array(roi_points, np.int32)
        roi_np = roi_np.reshape((-1, 1, 2))
        color = (0, 0, 255) if park_status[i] else (0, 255, 0)
        cv2.polylines(annotated_frame, [roi_np], True, color, 2)
        
        avg_x = sum(p[0] for p in roi_points) // len(roi_points)
        avg_y = sum(p[1] for p in roi_points) // len(roi_points)
        cv2.putText(annotated_frame, f'E{i+1}', (avg_x - 15, avg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el conteo
    text = f'Cupos Disponibles: {available_spaces} / {total_spaces}'
    cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Analisis de Parqueadero', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()