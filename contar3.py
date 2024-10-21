import cv2
import mediapipe as mp

# Inicializar el detector de rostros de Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Cargar la imagen
image = cv2.imread('foto3.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Iniciar el modelo
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
    result = face_detection.process(image_rgb)

    # Dibujar los rectángulos alrededor de los rostros detectados
    if result.detections:
        for detection in result.detections:
            mp_drawing.draw_detection(image, detection)

# Mostrar la imagen con detecciones
cv2.imshow("Rostros detectados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
