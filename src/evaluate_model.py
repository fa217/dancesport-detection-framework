from ultralytics import YOLO
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


MODELS = {
    "dancer_category": {
        "model": "runs/detect/dancer_category_model/weights/best.pt",
        "data": "datasets/dancer_category/data.yaml",
        "imgsz": 416
    },

    "dance_style": {
        "model": "runs/detect/dance_style_model/weights/best.pt",
        "data": "datasets/dance_style/data.yaml",
        "imgsz": 416
    },

    "formation_couple": {
        "model": "runs/detect/formation_couple_model/weights/best.pt",
        "data": "datasets/formation_couple/data.yaml",
        "imgsz": 640
    }
}


# --------------------------------------------------
# Select Model
# --------------------------------------------------

#SELECTED_MODEL = "dancer_category"

SELECTED_MODEL = "dance_style"
#SELECTED_MODEL = "formation_couple"

config = MODELS[SELECTED_MODEL]


# --------------------------------------------------
# Check Files
# --------------------------------------------------

model_path = Path(config["model"])
data_path = Path(config["data"])

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {data_path}"
    )


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = YOLO(str(model_path))


# --------------------------------------------------
# Evaluate Model on Test Set
# --------------------------------------------------

metrics = model.val(
    data=str(data_path),
    split="test",
    imgsz=config["imgsz"],
    project=str(ROOT / "runs/evaluation"),
    name=SELECTED_MODEL,
    exist_ok=True,
    workers=0
)


# --------------------------------------------------
# Store Results
# --------------------------------------------------

results_df = pd.DataFrame([
    {
        "model": SELECTED_MODEL,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4)
    }
])

output_csv = (
    Path("runs/evaluation")
    / SELECTED_MODEL
    / "evaluation_metrics.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\nEvaluation Results")
print("------------------")

print(f"Model      : {SELECTED_MODEL}")
print(f"Precision  : {metrics.box.mp:.4f}")
print(f"Recall     : {metrics.box.mr:.4f}")
print(f"mAP50      : {metrics.box.map50:.4f}")
print(f"mAP50-95   : {metrics.box.map:.4f}")

print(
    f"\nResults saved in: runs/evaluation/{SELECTED_MODEL}"
)

print(
    f"Metrics CSV saved in: {output_csv}"
)