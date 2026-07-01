import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "lat_std_detection_quality_summary.csv"
)

labels = df["dance_style"]
values = df["exact_match_rate"]

plt.figure(figsize=(7,5))

plt.bar(labels, values)

plt.title(
    "Exact Detection Rate: LAT AV vs STD AV"
)

plt.xlabel("Dance Style")
plt.ylabel("Exact Match Rate (%)")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.ylim(0,100)

plt.tight_layout()

plt.savefig(
    "formation_exact_match_comparison.png",
    dpi=300
)

plt.show()

print(
    "Saved: formation_exact_match_comparison.png"
)