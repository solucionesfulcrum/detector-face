from mtcnn import MTCNN
import cv2

# Inicializar el detector MTCNN
detector = MTCNN()

# Cargar la imagen
image_path = 'foto9.jpeg'  # Reemplazar con la ruta de tu imagen
image = cv2.imread(image_path)

# Verificar si la imagen fue cargada correctamente
if image is None:
    print(f"Error: No se pudo cargar la imagen desde {image_path}")
else:
    # Redimensionar la imagen si es muy grande para mejorar la eficiencia (opcional)
    max_dimension = 800  # Ajustar este valor según tus necesidades
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    # Convertir la imagen a RGB, ya que MTCNN lo requiere
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar rostros en la imagen
    results = detector.detect_faces(image_rgb)

    # Dibujar rectángulos alrededor de los rostros detectados
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el número de rostros detectados
    print(f"Número de rostros detectados: {len(results)}")

    # Mostrar la imagen con los rostros detectados
    cv2.imshow("Rostros detectados", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
