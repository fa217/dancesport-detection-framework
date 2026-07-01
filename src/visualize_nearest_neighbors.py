import cv2
import pandas as pd
import math
from pathlib import Path

IMAGE_NAMES = [
    "LAT_FORM_FB_AD_TP_111.jpg",
]

CSV_PATH = Path("formation_couple_coordinates.csv")

IMAGE_FOLDERS = [
    Path("all_formation_images"),
    Path("datasets/formation_couple/images/train"),
    Path("datasets/formation_couple/images/val"),
    Path("datasets/formation_couple/images/test"),
]



#OUTPUT_DIR = Path("formation_visualizations")
OUTPUT_DIR = Path("formation_error_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

for image_name in IMAGE_NAMES:
    image_df = df[df["image"] == image_name].copy()

    if image_df.empty:
        print(f"Skipping {image_name}: no coordinates found.")
        continue

    image_path = None

    for folder in IMAGE_FOLDERS:
        candidate = folder / image_name

        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"Skipping {image_name}: image not found.")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipping {image_name}: could not read image.")
        continue

    centers = []

    for _, row in image_df.iterrows():
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        cx = int(row["center_x"])
        cy = int(row["center_y"])
        couple_id = int(row["couple_id"])

        centers.append((couple_id, cx, cy))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            image,
            (cx, cy),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            image,
            f"C{couple_id}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    drawn_lines = set()

    for i, (id1, x1, y1) in enumerate(centers):
        best_distance = float("inf")
        nearest = None
        nearest_id = None

        for j, (id2, x2, y2) in enumerate(centers):
            if i == j:
                continue

            distance = math.sqrt(
                (x1 - x2) ** 2 +
                (y1 - y2) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                nearest = (x2, y2)
                nearest_id = id2

        if nearest is not None:
            line_key = tuple(sorted((id1, nearest_id)))

            if line_key not in drawn_lines:
                cv2.line(
                    image,
                    (x1, y1),
                    nearest,
                    (255, 255, 0),
                    2
                )

                mid_x = int((x1 + nearest[0]) / 2)
                mid_y = int((y1 + nearest[1]) / 2)

                cv2.putText(
                    image,
                    f"{best_distance:.0f}",
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                drawn_lines.add(line_key)

    output_file = OUTPUT_DIR / f"nearest_neighbor_distance_{image_name}"
    cv2.imwrite(str(output_file), image)

    print(f"Saved: {output_file}")