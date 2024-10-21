import cv2
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Cargar el modelo pre-entrenado Faster R-CNN desde Detectron2
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Umbral de confianza
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU si está disponible
    return DefaultPredictor(cfg)

# Función para detectar rostros en la imagen
def detect_faces(image_path, predictor):
    # Leer la imagen
    img = cv2.imread(image_path)

    # Realizar la predicción
    outputs = predictor(img)

    # Extraer las predicciones de clases (rostros y otros objetos)
    pred_classes = outputs["instances"].pred_classes
    pred_boxes = outputs["instances"].pred_boxes

    # Filtrar solo las detecciones de rostros (código de clase 0 en COCO es para "personas")
    face_boxes = [box for i, box in enumerate(pred_boxes) if pred_classes[i] == 0]

    # Dibujar rectángulos alrededor de los rostros detectados
    for box in face_boxes:
        x1, y1, x2, y2 = box.cpu().numpy()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Mostrar imagen con rostros detectados
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Retornar número de rostros detectados
    return len(face_boxes)

# Main: Cargar el modelo y detectar rostros
if __name__ == "__main__":
    image_path = "foto.jpg"  # Coloca el camino de tu imagen
    predictor = load_model()
    num_faces = detect_faces(image_path, predictor)
    print(f"Se detectaron {num_faces} rostros en la imagen.")
