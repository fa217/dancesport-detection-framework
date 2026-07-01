from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import math

model = YOLO("runs/detect/formation_couple_model/weights/best.pt")

source = "all_formation_images"

results = model(
    source,
    imgsz=640,
    conf=0.4,
    save=True
)

all_rows = []

for result in results:
    image_name = Path(result.path).name

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        all_rows.append({
            "image": image_name,
            "couple_id": i + 1,
            "confidence": round(conf, 3),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "center_x": round(center_x, 2),
            "center_y": round(center_y, 2)
        })

df = pd.DataFrame(all_rows)

output_path = "formation_couple_coordinates.csv"
df.to_csv(output_path, index=False)

print(f"Saved coordinates to {output_path}")
print(df.head())