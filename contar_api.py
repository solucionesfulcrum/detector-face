from flask import Flask, request, jsonify, send_file
from mtcnn import MTCNN
import cv2
import numpy as np
import io
from PIL import Image

# Inicializar Flask y el detector MTCNN
app = Flask(__name__)
detector = MTCNN()

# Ruta para procesar la imagen
@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    # Verificar que el archivo fue enviado
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    # Convertir la imagen en un array de numpy
    np_img = np.frombuffer(file.read(), np.uint8)

    # Cargar la imagen con OpenCV
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convertir la imagen a RGB para el detector MTCNN
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar rostros en la imagen
    results = detector.detect_faces(image_rgb)

    # Dibujar rectángulos alrededor de los rostros detectados
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Colocar la cantidad de personas detectadas en la parte superior izquierda
    total_faces = len(results)
    cv2.putText(image, f'Total: {total_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Convertir la imagen resultante (con rectángulos y texto) a JPEG en memoria
    _, img_encoded = cv2.imencode('.jpg', image)

    # Convertir el buffer de la imagen a un archivo de bytes para enviarlo
    img_bytes = io.BytesIO(img_encoded)

    # Enviar la imagen y el número total de personas detectadas
    return send_file(img_bytes, mimetype='image/jpeg', as_attachment=False, download_name="resultado.jpg")

# Iniciar el servidor
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
