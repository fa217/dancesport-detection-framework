import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("datasets/formation_couple")
PREDICTION_CSV = Path("formation_couple_coordinates.csv")

results = []

for split in ["train", "val", "test"]:
    label_dir = DATASET_ROOT / "labels" / split

    for label_file in label_dir.glob("*.txt"):
        image_name = label_file.name.replace(".txt", ".jpg")
        parts = image_name.split("_")

        dance_style = parts[0]
        viewpoint = parts[4]

        if not (
            (dance_style == "LAT" and viewpoint == "AV")
            or
            (dance_style == "STD" and viewpoint == "AV")
        ):
            continue

        with open(label_file, "r") as f:
            gt_count = len([line for line in f.readlines() if line.strip()])

        results.append({
            "image": image_name,
            "dance_style": dance_style,
            "viewpoint": viewpoint,
            "split": split,
            "ground_truth_couples": gt_count
        })

gt_df = pd.DataFrame(results)

pred_df = pd.read_csv(PREDICTION_CSV)

pred_counts = (
    pred_df.groupby("image")
    .size()
    .reset_index(name="predicted_couples")
)

comparison = gt_df.merge(
    pred_counts,
    on="image",
    how="left"
)

comparison["predicted_couples"] = (
    comparison["predicted_couples"]
    .fillna(0)
    .astype(int)
)

comparison["difference"] = (
    comparison["predicted_couples"] -
    comparison["ground_truth_couples"]
)

comparison["absolute_difference"] = (
    comparison["difference"].abs()
)

comparison["exact_match"] = (
    comparison["difference"] == 0
)

comparison["within_one"] = (
    comparison["absolute_difference"] <= 1
)

comparison["detection_ratio"] = (
    comparison["predicted_couples"] /
    comparison["ground_truth_couples"]
).round(3)

comparison.to_csv(
    "lat_std_detection_quality_per_image.csv",
    index=False
)

summary = (
    comparison.groupby("dance_style")
    .agg(
        images=("image", "count"),
        gt_couples=("ground_truth_couples", "sum"),
        predicted_couples=("predicted_couples", "sum"),
        avg_gt_per_image=("ground_truth_couples", "mean"),
        avg_pred_per_image=("predicted_couples", "mean"),
        mean_absolute_error=("absolute_difference", "mean"),
        exact_match_rate=("exact_match", "mean"),
        within_one_rate=("within_one", "mean"),
        avg_detection_ratio=("detection_ratio", "mean")
    )
    .reset_index()
)

summary["exact_match_rate"] = (
    summary["exact_match_rate"] * 100
).round(2)

summary["within_one_rate"] = (
    summary["within_one_rate"] * 100
).round(2)

summary = summary.round(3)

summary.to_csv(
    "lat_std_detection_quality_summary.csv",
    index=False
)

print(summary)

print("\nSaved:")
print("lat_std_detection_quality_per_image.csv")
print("lat_std_detection_quality_summary.csv")