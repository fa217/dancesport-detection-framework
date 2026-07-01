from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np


def run_image_detection(
    model_path,
    image,
    image_name,
    selected_class="All",
    confidence_threshold=0.4,
    imgsz=416
):
    model = YOLO(str(model_path))

    image_np = np.array(image)

    results = model(
        image_np,
        imgsz=imgsz,
        conf=confidence_threshold
    )

    result = results[0]

    detections = []
    keep_indices = []

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if selected_class == "All" or class_name == selected_class:
            keep_indices.append(i)
            detections.append({
                "Image": image_name,
                "Class": class_name,
                "Confidence": round(conf, 2)
            })

    if selected_class != "All":
        result.boxes = result.boxes[keep_indices]

    result_image = result.plot()

    return result_image, detections, model.names
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2


def run_image_detection(
    model_path,
    image,
    image_name,
    selected_class="All",
    confidence_threshold=0.4,
    imgsz=416
):
    model = YOLO(str(model_path))

    image_np = np.array(image)

    results = model(
        image_np,
        imgsz=imgsz,
        conf=confidence_threshold
    )

    result = results[0]

    detections = []
    keep_indices = []

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if selected_class == "All" or class_name == selected_class:
            keep_indices.append(i)

            detections.append({
                "Image": image_name,
                "Class": class_name,
                "Confidence": round(conf, 2)
            })

    if selected_class != "All":
        result.boxes = result.boxes[keep_indices]

    result_image = result.plot()

    return result_image, detections, model.names


# --------------------------------------------------
# Standalone test-set prediction
# --------------------------------------------------
if __name__ == "__main__":

    MODELS = {
        "dancer_category": {
            "model": "runs/detect/dancer_category_model/weights/best.pt",
            "source": "datasets/dancer_category/images/test",
            "imgsz": 416,
            "conf": 0.4,
            "output": "dancer_category"
        },
        "dance_style": {
            "model": "runs/detect/dance_style_model/weights/best.pt",
            "source": "datasets/dance_style/images/test",
            "imgsz": 416,
            "conf": 0.4,
            "output": "dance_style"
        },
        "formation_couple": {
            "model": "runs/detect/formation_couple_model/weights/best.pt",
            "source": "datasets/formation_couple/images/test",
            "imgsz": 640,
            "conf": 0.4,
            "output": "formation_couple"
        }
    }

    SELECTED_MODEL = "dancer_category"
    # SELECTED_MODEL = "dance_style"
    # SELECTED_MODEL = "formation_couple"

    config = MODELS[SELECTED_MODEL]

    model_path = Path(config["model"])
    source_path = Path(config["source"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Test image folder not found: {source_path}")

    model = YOLO(str(model_path))

    model.predict(
        source=str(source_path),
        imgsz=config["imgsz"],
        conf=config["conf"],
        save=True,
        project="runs/test",
        name=config["output"],
        exist_ok=True
    )

    print("\nTest prediction completed.")
    print(f"Model: {SELECTED_MODEL}")
    print(f"Results saved in: runs/test/{config['output']}")