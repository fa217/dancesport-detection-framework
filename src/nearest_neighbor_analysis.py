import pandas as pd
import math
from pathlib import Path

INPUT_CSV = Path("formation_couple_coordinates.csv")

df = pd.read_csv(INPUT_CSV)

results = []

for image_name, group in df.groupby("image"):
    points = group[["center_x", "center_y"]].values

    if len(points) < 2:
        continue

    nearest_distances = []

    for i, p1 in enumerate(points):
        distances = []

        for j, p2 in enumerate(points):
            if i == j:
                continue

            dist = math.sqrt(
                (p1[0] - p2[0]) ** 2 +
                (p1[1] - p2[1]) ** 2
            )
            distances.append(dist)

        nearest_distances.append(min(distances))

    avg_nn = pd.Series(nearest_distances).mean()
    std_nn = pd.Series(nearest_distances).std()
    cv_nn = std_nn / avg_nn if avg_nn > 0 else None

    parts = image_name.split("_")

    dance_style = parts[0]
    viewpoint = parts[4]

    results.append({
        "image": image_name,
        "dance_style": dance_style,
        "viewpoint": viewpoint,
        "num_couples": len(points),
        "avg_nearest_neighbor": round(avg_nn, 2),
        "std_nearest_neighbor": round(std_nn, 2),
        "cv_nearest_neighbor": round(cv_nn, 3),
        "min_nearest_neighbor": round(min(nearest_distances), 2),
        "max_nearest_neighbor": round(max(nearest_distances), 2)
    })

result_df = pd.DataFrame(results)

# 1. Alle Ergebnisse speichern
result_df.to_csv("nearest_neighbor_analysis_all.csv", index=False)

# 2. Nur wissenschaftlich relevante Gruppen auswählen:
# LAT AV, LAT TP, STD AV
relevant = result_df[
    (
        ((result_df["dance_style"] == "LAT") & (result_df["viewpoint"].isin(["AV", "TP"])))
        |
        ((result_df["dance_style"] == "STD") & (result_df["viewpoint"] == "AV"))
    )
].copy()

# 3. Plausible Formationen behalten:
# mindestens 5 Paare, maximal 8 Paare
filtered = relevant[
    (relevant["num_couples"] >= 5)
    &
    (relevant["num_couples"] <= 8)
].copy()

# 4. Ausgeschlossene Bilder speichern
excluded = relevant[
    ~relevant["image"].isin(filtered["image"])
].copy()

excluded.to_csv("nearest_neighbor_excluded_images.csv", index=False)

# 5. Finale gefilterte Analyse speichern
filtered.to_csv("nearest_neighbor_analysis_filtered.csv", index=False)

# 6. Gruppenzusammenfassung berechnen
summary = (
    filtered.groupby(["dance_style", "viewpoint"])
    .agg(
        images=("image", "count"),
        avg_couples=("num_couples", "mean"),
        avg_nearest_neighbor=("avg_nearest_neighbor", "mean"),
        avg_std_nn=("std_nearest_neighbor", "mean"),
        avg_cv_nn=("cv_nearest_neighbor", "mean")
    )
    .reset_index()
    .round(3)
)

summary.to_csv("nearest_neighbor_group_summary_filtered.csv", index=False)

print(summary)

print("\nSaved:")
print("nearest_neighbor_analysis_all.csv")
print("nearest_neighbor_excluded_images.csv")
print("nearest_neighbor_analysis_filtered.csv")
print("nearest_neighbor_group_summary_filtered.csv")