import cv2

# Cargar el clasificador de rostros preentrenado (cascada de Haar)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen
image_path = 'foto8.jpg'  # Reemplazar con la ruta de tu imagen
image = cv2.imread(image_path)

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostros en la imagen
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar rectángulos alrededor de los rostros detectados
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Mostrar el número de rostros detectados
print(f"Número de rostros detectados: {len(faces)}")

# Mostrar la imagen con los rostros detectados
cv2.imshow("Rostros detectados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
