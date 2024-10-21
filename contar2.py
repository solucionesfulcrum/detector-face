from mtcnn import MTCNN
import cv2

# Inicializar el detector MTCNN
detector = MTCNN()

# Cargar la imagen
image_path = 'foto9.jpeg'  # Reemplazar con la ruta de tu imagen
image = cv2.imread(image_path)

# Convertir la imagen a RGB, ya que MTCNN lo requiere
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar rostros en la imagen
results = detector.detect_faces(image_rgb)

# Dibujar rectángulos alrededor de los rostros detectados
for result in results:
    x, y, w, h = result['box']
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Obtener el número total de rostros detectados
total_personas = len(results)

# Colocar el total en la parte superior derecha
# Obtener el tamaño de la imagen para calcular la posición del texto
height, width, _ = image.shape

# Posición del texto en la parte superior derecha
text_position = (10, 50)  # Cambiado para estar cerca de la esquina izquierda

# Colocar el texto con el total de personas en rojo
cv2.putText(image, f'Total personas: {total_personas}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Mostrar el número de rostros detectados en la consola
print(f"Número de personas detectadas: {total_personas}")

# Mostrar la imagen con el total de personas detectadas
cv2.imshow("Rostros detectados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
