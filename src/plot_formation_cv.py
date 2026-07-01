import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "nearest_neighbor_group_summary_filtered.csv"
)

labels = df["dance_style"] + "_" + df["viewpoint"]

values = df["avg_cv_nn"]

plt.figure(figsize=(8, 5))

plt.bar(labels, values)

plt.title(
    "Coefficient of Variation of Nearest Neighbor Distances"
)

plt.xlabel("Formation Group")
plt.ylabel("Average CV")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.005,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "formation_comparison_cv.png",
    dpi=300
)

plt.show()

print("Saved: formation_comparison_cv.png")