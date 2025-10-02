# import cv2
# import sys
# import time

# Configuración de conexión
# Cambia esta IP por la de tu dispositivo móvil
# DEVICE_IP = '192.168.1.6'  # Cambia esta IP
# PORT = '8080'

# Diferentes URLs que puedes intentar (comenta/descomenta según tu app)
# urls_to_try = [
#     f'http://{DEVICE_IP}:{PORT}',      # HTTP sin SSL
#     f'https://{DEVICE_IP}:{PORT}',     # HTTPS con SSL
#     f'http://{DEVICE_IP}:{PORT}/video',      # HTTP sin SSL
#     f'https://{DEVICE_IP}:{PORT}/video',     # HTTPS con SSL
#     f'http://{DEVICE_IP}:{PORT}/mjpeg',      # Formato MJPEG
#     f'http://{DEVICE_IP}:{PORT}/cam/1/mjpeg', # Algunos apps usan este formato
# ]

# print(f"Intentando conectar a dispositivo en IP: {DEVICE_IP}:{PORT}")
# print("URLs a probar:")
# for i, url in enumerate(urls_to_try, 1):
#     print(f"  {i}. {url}")
# print()

# # Intenta conectar con diferentes URLs
# cap = None
# working_url = None

# for url in urls_to_try:
#     print(f"Probando: {url}")
#     cap = cv2.VideoCapture(url)
    
#     # Configura timeouts más cortos
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
#     # Intenta leer un frame para verificar la conexión
#     if cap.isOpened():
#         ret, frame = cap.read()
#         if ret and frame is not None:
#             working_url = url
#             print(f"✓ Conexión exitosa con: {url}")
#             break
#         else:
#             print(f"✗ No se pudo leer frame de: {url}")
#             cap.release()
#     else:
#         print(f"✗ No se pudo abrir: {url}")
#         cap.release()
    
#     time.sleep(1)  # Pausa entre intentos

# if not working_url:
#     print("\n❌ Error: No se pudo conectar a ninguna URL.")
#     print("\n🔍 Pasos para solucionar:")
#     print(f"1. Verifica que tu dispositivo móvil esté en la red WiFi")
#     print(f"2. Confirma la IP del dispositivo (puede haber cambiado)")
#     print(f"3. Asegúrate de que la app de cámara IP esté ejecutándose")
#     print(f"4. Verifica el puerto (puede ser diferente a 8080)")
#     print(f"5. Intenta acceder desde un navegador web: http://{DEVICE_IP}:{PORT}")
#     sys.exit(1)

# print("Conectado a la cámara. Presiona 'q' para salir.")

# while True:
#     # Lee un fotograma (frame) del video
#     ret, frame = cap.read()

#     # Si el fotograma se leyó correctamente, 'ret' será True
#     if not ret:
#         print("Error: No se pudo recibir un fotograma.")
#         break

#     # Muestra el fotograma en una ventana
#     cv2.imshow('Video del Celular', frame)

#     # Si se presiona la tecla 'q', sal del bucle
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Libera el objeto de captura y cierra todas las ventanas
# cap.release()
# cv2.destroyAllWindows()



# nueva data abajo de este comentario



import cv2
from ultralytics import YOLO
import numpy as np
import math

# --- CONFIGURACIÓN ---
video_source = 'http://10.173.114.226:8080/video'
# video_source = './videoPrueba.mp4'  # Ruta a un archivo de video local
model = YOLO('yolov8n.pt')

# --- MEMORIA DE OBJETOS ---
object_memory = {}
next_unique_id = 0
object_counts = {}

# Umbral de similitud para considerar dos objetos como el mismo
SIMILARITY_THRESHOLD = 0.85 # He aumentado ligeramente el umbral
MIN_DETECTION_SIZE = 50 
VECTOR_SIZE = (100, 100) # El nuevo tamaño estandarizado

def cosine_similarity(v1, v2):
    """Calcula la similitud del coseno entre dos vectores."""
    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    return dot_product / (norm_a * norm_b)

def get_image_vector(image):
    """
    Convierte una imagen en un vector de características,
    redimensionándola primero para asegurar un tamaño consistente.
    """
    # --- CAMBIO CRÍTICO AQUÍ: Redimensionamos la imagen ---
    resized_image = cv2.resize(image, VECTOR_SIZE, interpolation=cv2.INTER_AREA)
    return resized_image.flatten() / 255.0

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: No se pudo abrir la fuente de video.")
    exit()

print("Sistema de conteo inteligente iniciado. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, conf=0.5, persist=True)
    
    object_names = results[0].names
    annotated_frame = frame.copy()
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            class_name = object_names[class_id]

            if (x2 - x1) < MIN_DETECTION_SIZE or (y2 - y1) < MIN_DETECTION_SIZE:
                continue

            object_roi = frame[y1:y2, x1:x2]
            if object_roi.size == 0:
                continue
                
            current_vector = get_image_vector(object_roi)

            match_found = False
            for unique_id, stored_data in object_memory.items():
                if stored_data['class'] == class_name:
                    similarity = cosine_similarity(current_vector, stored_data['vector'])
                    if similarity > SIMILARITY_THRESHOLD:
                        match_found = True
                        label = f'{class_name} ID: {unique_id} (Visto)'
                        color = (255, 0, 0)
                        break
            
            if not match_found:
                new_unique_id = next_unique_id
                next_unique_id += 1
                
                object_memory[new_unique_id] = {
                    'vector': current_vector,
                    'class': class_name
                }
                
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                label = f'{class_name} ID: {new_unique_id} (Nuevo)'
                color = (0, 255, 0)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    y_offset = 30
    header_text = 'Inventario Acumulativo:'
    cv2.putText(annotated_frame, header_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y_offset += 25
    
    for class_name, count in object_counts.items():
        text = f'{class_name}: {count}'
        cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    cv2.imshow('Reconocimiento de Objetos', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()