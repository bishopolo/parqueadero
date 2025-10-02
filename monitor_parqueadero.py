import cv2
from ultralytics import YOLO
import numpy as np
import json
from datetime import datetime

# --- CONFIGURACIÓN ---
# Ruta de la IMAGEN DE REFERENCIA (donde marcaste las ROIs)
reference_image = './fotoPrueba.png'  # La imagen donde marcaste las ROIs
# Ruta del video a procesar
video_source = './videoPrueba.mp4'  # Cambia esto según tu video
# video_source = 'http://192.168.1.6:8080/video'  # O usa stream de cámara IP
# video_source = 0  # O usa webcam
rois_file = 'parqueadero_rois.json'  # Archivo JSON con las ROIs

# Detectar MOTOS Y CARROS
# 0: persona, 2: carro, 3: moto, 5: bus
TARGET_CLASSES = [2, 3]  # 2=carro, 3=moto

# Nombres de los espacios (personaliza según tus necesidades)
SPACE_NAMES = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']

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
        print(f"✅ Se cargaron {len(rois)} ROIs desde {filename}")
        return rois
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {filename}")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: El archivo {filename} no tiene un formato JSON válido")
        return []

# --- FUNCIÓN PARA VERIFICAR SI UN PUNTO ESTÁ DENTRO DE UN POLÍGONO ---
def point_in_roi(point, roi_points):
    """Verifica si un punto está dentro de una ROI (polígono)"""
    roi_array = np.array(roi_points, dtype=np.int32)
    result = cv2.pointPolygonTest(roi_array, point, False)
    return result >= 0

# --- FUNCIÓN PARA ESCALAR ROIs ---
def scale_rois(rois, original_size, target_size, offset_x=0, offset_y=0):
    """Escala las ROIs de la resolución original a la resolución objetivo."""
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
    
    print(f"🔄 ROIs escaladas de {original_size} a {target_size}")
    print(f"   Factor de escala: X={scale_x:.2f}, Y={scale_y:.2f}")
    if offset_x != 0 or offset_y != 0:
        print(f"   Offset aplicado: X={offset_x}, Y={offset_y}")
    
    return scaled_rois

# --- FUNCIÓN PARA DIBUJAR DASHBOARD ---
def draw_dashboard(frame, total_spaces, occupied_spaces, free_spaces, roi_status, space_names):
    """Dibuja un panel de información en la parte superior del frame"""
    # Fondo del dashboard
    dashboard_height = 160
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], dashboard_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Título
    cv2.putText(frame, 'SISTEMA DE MONITOREO - PARQUEADERO UNIVERSIDAD', (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    
    # Hora actual
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, current_time, (frame.shape[1] - 150, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Estadísticas principales
    y_pos = 75
    cv2.putText(frame, f'ESPACIOS TOTALES: {total_spaces}', (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Espacios libres (verde)
    cv2.circle(frame, (370, y_pos - 8), 8, (0, 255, 0), -1)
    cv2.putText(frame, f'LIBRES: {free_spaces}', (390, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Espacios ocupados (rojo)
    cv2.circle(frame, (630, y_pos - 8), 8, (0, 0, 255), -1)
    cv2.putText(frame, f'OCUPADOS: {occupied_spaces}', (650, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Estado de disponibilidad
    availability_text = "✓ HAY ESPACIOS DISPONIBLES" if free_spaces > 0 else "✗ PARQUEADERO LLENO"
    availability_color = (0, 255, 0) if free_spaces > 0 else (0, 0, 255)
    cv2.putText(frame, availability_text, (20, y_pos + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, availability_color, 3)
    
    # Lista de espacios individuales
    x_start = 20
    y_start = y_pos + 85
    for idx, status in enumerate(roi_status):
        if idx >= len(space_names):
            break
        space_name = space_names[idx]
        status_text = "OCUPADO" if status['occupied'] else "LIBRE"
        color = (0, 0, 255) if status['occupied'] else (0, 255, 0)
        
        # Círculo indicador
        circle_x = x_start + (idx * 130)
        cv2.circle(frame, (circle_x, y_start - 5), 6, color, -1)
        
        # Texto del espacio
        text = f"{space_name}: {status_text}"
        cv2.putText(frame, text, (circle_x + 15, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# --- FUNCIÓN PARA DIBUJAR ROIs ---
def draw_rois(frame, rois, roi_status, space_names):
    """Dibuja las ROIs en el frame con colores según el estado"""
    for idx, roi in enumerate(rois):
        roi_array = np.array(roi, dtype=np.int32)
        # Rojo si está ocupado, verde si está libre
        color = (0, 0, 255) if roi_status[idx]['occupied'] else (0, 255, 0)
        cv2.polylines(frame, [roi_array], isClosed=True, color=color, thickness=3)
        
        # Calcula el centro de la ROI para poner el texto
        M = cv2.moments(roi_array)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Nombre del espacio
            space_name = space_names[idx] if idx < len(space_names) else f"E{idx+1}"
            status_text = "OCUPADO" if roi_status[idx]['occupied'] else "LIBRE"
            vehicle_type = roi_status[idx]['vehicle_type']
            
            # Dibuja fondo para el texto
            text = f"{space_name}: {status_text}"
            if vehicle_type:
                text += f" ({vehicle_type})"
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (cx - text_width//2 - 5, cy - text_height - 5), 
                         (cx + text_width//2 + 5, cy + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (cx - text_width//2, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# --- FUNCIÓN PRINCIPAL ---
def main():
    # Carga las ROIs
    rois = load_rois(rois_file)
    
    if len(rois) == 0:
        print("❌ No se encontraron ROIs. Ejecuta primero roi_selector.py")
        return
    
    # Lee la imagen de referencia para obtener sus dimensiones
    ref_img = cv2.imread(reference_image)
    if ref_img is None:
        print(f"⚠️  ADVERTENCIA: No se pudo cargar '{reference_image}'")
        print("   Las ROIs podrían no estar en la posición correcta.")
        original_size = None
    else:
        original_size = (ref_img.shape[1], ref_img.shape[0])
        print(f"📷 Imagen de referencia: {original_size[0]}x{original_size[1]}")
    
    # Abre el video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video '{video_source}'")
        return
    
    # Obtiene las dimensiones del video
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (video_width, video_height)
    print(f"🎥 Video: {video_width}x{video_height}")
    
    # Escala las ROIs si es necesario
    if original_size and original_size != video_size:
        print(f"\n⚠️  RESOLUCIONES DIFERENTES DETECTADAS")
        rois = scale_rois(rois, original_size, video_size, OFFSET_X, OFFSET_Y)
    else:
        print("✅ Resoluciones coinciden")
        if OFFSET_X != 0 or OFFSET_Y != 0:
            rois = [[(p[0] + OFFSET_X, p[1] + OFFSET_Y) for p in roi] for roi in rois]
            print(f"   Offset aplicado: X={OFFSET_X}, Y={OFFSET_Y}")
    
    print(f"\n🚀 Iniciando monitoreo...")
    print(f"   📍 {len(rois)} espacios monitoreados")
    print(f"   🎯 Detectando: Motos y Carros")
    print(f"   📊 Confianza mínima: {CONFIDENCE_THRESHOLD*100}%")
    print(f"   ⏱️  Persistencia: {PERSISTENCE_FRAMES} frames")
    print("   ⌨️  Presiona 'q' para salir\n")
    
    # Inicializar sistema de persistencia para cada ROI
    roi_persistence = [{'frames': 0, 'vehicle_type': None} for _ in range(len(rois))]
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("\n✅ Fin del video.")
            break
        
        frame_count += 1
        
        # Realiza la detección de objetos (motos y carros)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASSES)
        
        # Lista para almacenar detecciones
        detected_boxes = []
        
        # Procesa los resultados de la detección
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                class_id = int(box.cls[0])
                
                # Determina el tipo de vehículo
                vehicle_type = "CARRO" if class_id == 2 else "MOTO"
                
                detected_boxes.append({
                    'box': (x1, y1, x2, y2),
                    'centroid': centroid,
                    'class_id': class_id,
                    'vehicle_type': vehicle_type
                })
                
                # Dibuja la caja y el centro del objeto detectado
                color = (0, 255, 255) if class_id == 2 else (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, centroid, 5, color, -1)
                cv2.putText(frame, vehicle_type, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Verifica qué ROIs tienen vehículos (con sistema de persistencia)
        roi_status = []
        occupied_count = 0
        
        for idx, roi in enumerate(rois):
            has_vehicle = False
            current_vehicle_type = None
            
            for detection in detected_boxes:
                # Verifica si el centroide está dentro de la ROI
                if point_in_roi(detection['centroid'], roi):
                    has_vehicle = True
                    current_vehicle_type = detection['vehicle_type']
                    roi_persistence[idx]['frames'] = PERSISTENCE_FRAMES
                    roi_persistence[idx]['vehicle_type'] = current_vehicle_type
                    break
            
            # Si no se detectó ahora, verifica la persistencia
            if not has_vehicle and roi_persistence[idx]['frames'] > 0:
                has_vehicle = True
                current_vehicle_type = roi_persistence[idx]['vehicle_type']
                roi_persistence[idx]['frames'] -= 1
            elif not has_vehicle:
                roi_persistence[idx]['vehicle_type'] = None
            
            roi_status.append({
                'occupied': has_vehicle,
                'vehicle_type': current_vehicle_type
            })
            
            if has_vehicle:
                occupied_count += 1
        
        # Calcula estadísticas
        total_spaces = len(rois)
        free_spaces = total_spaces - occupied_count
        
        # Dibuja el dashboard
        draw_dashboard(frame, total_spaces, occupied_count, free_spaces, roi_status, SPACE_NAMES)
        
        # Dibuja las ROIs con su estado
        draw_rois(frame, rois, roi_status, SPACE_NAMES)
        
        # Muestra información adicional
        cv2.putText(frame, f'Frame: {frame_count} | Detecciones: {len(detected_boxes)}', 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Muestra el video con las detecciones
        cv2.imshow('Sistema de Monitoreo de Parqueadero - Presiona q para salir', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Deteniendo monitoreo...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Sistema finalizado.\n")

if __name__ == '__main__':
    main()
