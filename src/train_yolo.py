import torch
from ultralytics import YOLO
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "yolo26n.pt"

DATASETS = {
    "dancer_category": {
        "data": "datasets/dancer_category/data.yaml",
        "epochs": 50,
        "imgsz": 416,
        "batch": 4,
        "name": "dancer_category_model"
    },
    "dance_style": {
        "data": "datasets/dance_style/data.yaml",
        "epochs": 50,
        "imgsz": 416,
        "batch": 4,
        "name": "dance_style_model"
    },
    "formation_couple": {
        "data": "datasets/formation_couple/data.yaml",
        "epochs": 80,
        "imgsz": 640,
        "batch": 4,
        "name": "formation_couple_model"
    }
}


# --------------------------------------------------
# Select Dataset
# --------------------------------------------------

#SELECTED_DATASET = "dancer_category"
SELECTED_DATASET = "dance_style"
#SELECTED_DATASET = "formation_couple"

config = DATASETS[SELECTED_DATASET]

data_path = Path(config["data"])

if not data_path.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {data_path}")


# --------------------------------------------------
# Select Device
# --------------------------------------------------
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA GPU not available. Training will use the CPU.")


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = YOLO(MODEL_NAME)


# --------------------------------------------------
# Train Model
# --------------------------------------------------

model.train(
    data=str(ROOT / config["data"]),
    epochs=config["epochs"],
    imgsz=config["imgsz"],
    batch=config["batch"],
    workers=0,
    device=DEVICE,
    project=str(ROOT / "runs/detect"),
    name=config["name"],
    exist_ok=True
)


print("\nTraining completed.")
print(f"Dataset: {SELECTED_DATASET}")
print(f"Device: {DEVICE}")
print(f"Model saved in: runs/detect/{config['name']}")