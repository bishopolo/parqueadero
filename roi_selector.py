# import cv2
# import json
# import numpy as np

# rois = []
# current_roi = []
# drawing = False
# image = None
# window_name = 'Seleccionar ROIs (Clic en vertices, Enter para guardar ROI, "s" para guardar todo y salir)'

# def draw_roi(event, x, y, flags, param):
#     """
#     Función de callback para el manejo de eventos del mouse.
#     Permite dibujar polígonos para definir las ROIs.
#     """
#     global current_roi, drawing, image

#     if event == cv2.EVENT_LBUTTONDOWN:
#         current_roi.append((x, y))
#         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
#         cv2.imshow(window_name, image)

# def save_roi():
#     """Guarda la ROI actual y limpia la lista para la siguiente."""
#     global rois, current_roi
#     if len(current_roi) >= 3:
#         rois.append(current_roi.copy())
#         print(f"ROI guardada: {current_roi}")
#     current_roi = []
#     redraw_rois()

# def redraw_rois():
#     """Redibuja todas las ROIs guardadas en la imagen original."""
#     global image, rois
#     temp_image = image.copy()
#     for roi in rois:
#         pts = np.array(roi, np.int32)
#         pts = pts.reshape((-1, 1, 2))
#         cv2.polylines(temp_image, [pts], True, (255, 0, 0), 2)
#         for point in roi:
#             cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
#     cv2.imshow(window_name, temp_image)

# if __name__ == '__main__':
#     # Ruta a la imagen del parqueadero. ¡Cámbiala por la tuya!
#     image_path = 'uno.jpg'
#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"Error: No se pudo cargar la imagen '{image_path}'")
#         exit()

#     cv2.namedWindow(window_name)
#     cv2.setMouseCallback(window_name, draw_roi)

#     print("Instrucciones:")
#     print("1. Haz clic izquierdo en los vértices de cada espacio de parqueo.")
#     print("2. Presiona la tecla 'Enter' para guardar la ROI actual.")
#     print("3. Repite los pasos para todos los espacios.")
#     print("4. Presiona la tecla 's' para guardar todas las ROIs en un archivo JSON y salir.")
#     print("5. Presiona la tecla 'q' para salir sin guardar.")

#     cv2.imshow(window_name, image)

#     while True:
#         key = cv2.waitKey(1) & 0xFF

#         if key == 13:  # Tecla Enter
#             save_roi()
#         elif key == ord('s'):
#             if rois:
#                 with open('rois.json', 'w') as f:
#                     json.dump(rois, f)
#                 print("ROIs guardadas en 'rois.json'")
#             break
#         elif key == ord('q'):
#             break

#     cv2.destroyAllWindows()
#     print("Programa finalizado.")



# version 2
import cv2
import json
import numpy as np

rois = []
image = None
window_name = 'Seleccionar ROIs (Clic, Enter, s, q)'

def on_mouse(event, x, y, flags, param):
    """Callback para manejar eventos del mouse."""
    global rois, image

    if event == cv2.EVENT_LBUTTONDOWN:
        rois[-1].append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(window_name, image)

def main_roi_selector():
    """Función principal para seleccionar las ROIs."""
    global rois, image
    
    # 1. Carga la imagen de tu parqueadero (asegúrate de que esté en la misma carpeta)
    image_path = 'fotoPrueba.png'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: No se pudo cargar la imagen '{image_path}'.")
        return

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Instrucciones:")
    print("1. Clic izquierdo en los vértices de cada espacio de parqueo.")
    print("2. Presiona ENTER para guardar la ROI actual y empezar una nueva.")
    print("3. Presiona 's' para guardar todas las ROIs y salir.")
    print("4. Presiona 'q' para salir sin guardar.")

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Tecla ENTER
            if len(rois) > 0 and len(rois[-1]) >= 3:
                cv2.polylines(image, [np.array(rois[-1], np.int32)], True, (255, 0, 0), 2)
                rois.append([])
            elif len(rois) == 0:
                rois.append([])

        elif key == ord('s'):
            if len(rois[-1]) < 3:
                rois.pop()
            with open('parqueadero_rois.json', 'w') as f:
                json.dump(rois, f)
            print("ROIs guardadas en 'parqueadero_rois.json'.")
            break

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == '__main__':
    main_roi_selector()