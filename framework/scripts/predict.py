from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

model_path = ROOT / "framework/models/dancer_category.pt"
input_path = ROOT / "framework/input"
output_path = ROOT / "framework/output"

model = YOLO(model_path)

model.predict(
    source=input_path,
    imgsz=416,
    conf=0.5,
    save=True,
    project=str(output_path),
    name="predictions",
    exist_ok=True
)

print("Results saved in:", output_path / "predictions")