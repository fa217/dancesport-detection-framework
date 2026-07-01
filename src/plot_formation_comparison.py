import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "nearest_neighbor_group_summary_filtered.csv"
)

labels = (
    df["dance_style"] + "_" + df["viewpoint"]
)

values = df["avg_nearest_neighbor"]

plt.figure(figsize=(8, 5))

plt.bar(labels, values)

plt.title(
    "Average Nearest-Neighbor Distance by Formation Type"
)

plt.xlabel("Formation Group")
plt.ylabel("Average Nearest-Neighbor Distance (pixels)")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 2,
        f"{value:.1f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "formation_comparison_avg_nn.png",
    dpi=300
)

plt.show()

print("Saved: formation_comparison_avg_nn.png")