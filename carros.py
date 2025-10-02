import cv2
from ultralytics import YOLO
import numpy as np
import json

# --- CONFIGURACIÓN ---
# Ruta de la IMAGEN DE REFERENCIA (donde marcaste las ROIs)
reference_image = './fotoPrueba.png'  # La imagen donde marcaste las ROIs
# Ruta del video a procesar
video_source = './videoPrueba.mp4'  # Cambia esto según tu video
# video_source = 'http://192.168.1.6:8080/video'  # O usa stream de cámara IP
rois_file = 'parqueadero_rois.json'  # Archivo JSON con las ROIs

# El ID del objeto que quieres detectar
# 0: persona, 2: carro, 3: moto, 5: bus
TARGET_CLASS_ID = 3  # 3 es el ID para "moto" en el modelo YOLO

# --- AJUSTES FINOS ---
# Ajuste manual para corregir desalineación (en píxeles)
OFFSET_X = 0  # Ajusta si las ROIs están muy a la izquierda (-) o derecha (+)
OFFSET_Y = 0  # Ajusta si las ROIs están muy arriba (-) o abajo (+)

# Configuración de persistencia (para evitar parpadeo en detección)
PERSISTENCE_FRAMES = 15  # Cuántos frames debe estar "ocupada" una ROI después de detectar
CONFIDENCE_THRESHOLD = 0.3  # Umbral de confianza (más bajo = más sensible)

# Carga el modelo YOLOv8
model = YOLO('yolov8n.pt')

# --- CARGAR ROIs DESDE JSON ---
def load_rois(filename):
    """Carga las ROIs desde un archivo JSON"""
    try:
        with open(filename, 'r') as f:
            rois = json.load(f)
        print(f"Se cargaron {len(rois)} ROIs desde {filename}")
        return rois
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
        return []
    except json.JSONDecodeError:
        print(f"Error: El archivo {filename} no tiene un formato JSON válido")
        return []

# --- FUNCIÓN PARA VERIFICAR SI UN PUNTO ESTÁ DENTRO DE UN POLÍGONO ---
def point_in_roi(point, roi_points):
    """Verifica si un punto está dentro de una ROI (polígono)"""
    roi_array = np.array(roi_points, dtype=np.int32)
    result = cv2.pointPolygonTest(roi_array, point, False)
    return result >= 0

# --- FUNCIÓN PARA ESCALAR ROIs ---
def scale_rois(rois, original_size, target_size, offset_x=0, offset_y=0):
    """
    Escala las ROIs de la resolución original a la resolución objetivo.
    
    Args:
        rois: Lista de ROIs (cada ROI es una lista de puntos)
        original_size: Tupla (ancho, alto) de la imagen original
        target_size: Tupla (ancho, alto) del video/imagen objetivo
        offset_x: Desplazamiento horizontal manual en píxeles
        offset_y: Desplazamiento vertical manual en píxeles
    
    Returns:
        Lista de ROIs escaladas
    """
    orig_width, orig_height = original_size
    target_width, target_height = target_size
    
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    scaled_rois = []
    for roi in rois:
        scaled_roi = []
        for point in roi:
            new_x = int(point[0] * scale_x) + offset_x
            new_y = int(point[1] * scale_y) + offset_y
            scaled_roi.append((new_x, new_y))
        scaled_rois.append(scaled_roi)
    
    print(f"ROIs escaladas de {original_size} a {target_size}")
    print(f"Factor de escala: X={scale_x:.2f}, Y={scale_y:.2f}")
    if offset_x != 0 or offset_y != 0:
        print(f"Offset aplicado: X={offset_x}, Y={offset_y}")
    
    return scaled_rois

# --- FUNCIÓN PARA DIBUJAR ROIs ---
def draw_rois(frame, rois, roi_status):
    """Dibuja las ROIs en el frame con colores según el estado"""
    for idx, roi in enumerate(rois):
        roi_array = np.array(roi, dtype=np.int32)
        # Verde si hay moto, rojo si no hay
        color = (0, 255, 0) if roi_status[idx] else (0, 0, 255)
        cv2.polylines(frame, [roi_array], isClosed=True, color=color, thickness=3)
        
        # Calcula el centro de la ROI para poner el texto
        M = cv2.moments(roi_array)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            status_text = "MOTO" if roi_status[idx] else "LIBRE"
            cv2.putText(frame, f"ROI {idx+1}: {status_text}", (cx - 50, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Carga las ROIs
rois = load_rois(rois_file)

if len(rois) == 0:
    print("No se encontraron ROIs. Asegúrate de haber creado el archivo JSON con roi_selector.py")
    exit()

# Lee la imagen de referencia para obtener sus dimensiones
ref_img = cv2.imread(reference_image)
if ref_img is None:
    print(f"ADVERTENCIA: No se pudo cargar la imagen de referencia '{reference_image}'")
    print("Las ROIs podrían no estar en la posición correcta si el video tiene diferente resolución.")
    original_size = None
else:
    original_size = (ref_img.shape[1], ref_img.shape[0])  # (ancho, alto)
    print(f"Imagen de referencia: {original_size[0]}x{original_size[1]}")

# Abre el video
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video '{video_source}'")
    exit()

# Obtiene las dimensiones del video
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_size = (video_width, video_height)
print(f"Video: {video_width}x{video_height}")

# Escala las ROIs si es necesario
if original_size and original_size != video_size:
    print(f"\n⚠️  RESOLUCIONES DIFERENTES DETECTADAS")
    print(f"Imagen de referencia: {original_size[0]}x{original_size[1]}")
    print(f"Video: {video_width}x{video_height}")
    rois = scale_rois(rois, original_size, video_size, OFFSET_X, OFFSET_Y)
    print("✅ ROIs escaladas correctamente\n")
else:
    print("✅ Resoluciones coinciden, no se requiere escalar\n")
    # Aplicar offset incluso si las resoluciones coinciden
    if OFFSET_X != 0 or OFFSET_Y != 0:
        adjusted_rois = []
        for roi in rois:
            adjusted_roi = [(p[0] + OFFSET_X, p[1] + OFFSET_Y) for p in roi]
            adjusted_rois.append(adjusted_roi)
        rois = adjusted_rois
        print(f"Offset aplicado: X={OFFSET_X}, Y={OFFSET_Y}\n")

print(f"Procesando video '{video_source}'. Detectando motos en {len(rois)} ROIs...")
print(f"Confianza mínima: {CONFIDENCE_THRESHOLD*100}%")
print(f"Persistencia: {PERSISTENCE_FRAMES} frames")
print("Presiona 'q' para salir.\n")

# Inicializar sistema de persistencia para cada ROI
roi_persistence = [0] * len(rois)  # Contador de frames desde última detección

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Fin del video o error al leer el frame.")
        break
    
    frame_count += 1
    
    # Realiza la detección de objetos (solo motos) con umbral ajustable
    results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=[TARGET_CLASS_ID])
    
    # Lista para almacenar detecciones
    detected_boxes = []
    
    # Procesa los resultados de la detección
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            detected_boxes.append({
                'box': (x1, y1, x2, y2),
                'centroid': centroid
            })
            
            # Dibuja la caja y el centro del objeto detectado
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, centroid, 5, (255, 0, 255), -1)
            cv2.putText(frame, 'MOTO', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Verifica qué ROIs tienen motos (con sistema de persistencia)
    roi_status = []
    for idx, roi in enumerate(rois):
        has_motorcycle = False
        for detection in detected_boxes:
            # Verifica si el centroide de la detección está dentro de la ROI
            if point_in_roi(detection['centroid'], roi):
                has_motorcycle = True
                roi_persistence[idx] = PERSISTENCE_FRAMES  # Resetea el contador
                break
        
        # Si no se detectó ahora, verifica la persistencia
        if not has_motorcycle and roi_persistence[idx] > 0:
            has_motorcycle = True  # Mantener como ocupada
            roi_persistence[idx] -= 1  # Decrementar contador
        
        roi_status.append(has_motorcycle)
    
    # Dibuja las ROIs con su estado
    draw_rois(frame, rois, roi_status)
    
    # Muestra el total de motos detectadas
    total_motos = len(detected_boxes)
    cv2.putText(frame, f'Total Motos: {total_motos}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Información del frame
    cv2.putText(frame, f'Frame: {frame_count}', (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Muestra el video con las detecciones
    cv2.imshow('Deteccion de Motos en ROIs - Presiona q para salir', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nProcesamiento finalizado.")