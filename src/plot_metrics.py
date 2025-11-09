import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "../models/traffic_sign_model/results.csv"

if not os.path.exists(csv_path):
    print("ERROR: results.csv not found at:", csv_path)
    exit()

df = pd.read_csv(csv_path)

print("✅ CSV Loaded Successfully!")
print("Available columns:", list(df.columns))


def plot_metric(columns, title, save_name):
    plt.figure(figsize=(10, 6))

    for col, label in columns:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=label)
        else:
            print(f"⚠️ WARNING: Column '{col}' not found. Skipping.")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    save_path = f"../results/{save_name}.png"
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.show()


# --------------------------
# 1. TRAINING LOSSES
# --------------------------
plot_metric(
    [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Class Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
    ],
    "Training Losses Over Epochs",
    "training_losses"
)

# --------------------------
# 2. VALIDATION LOSSES
# --------------------------
plot_metric(
    [
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Class Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
    ],
    "Validation Losses Over Epochs",
    "validation_losses"
)

# --------------------------
# 3. PRECISION & RECALL
# --------------------------
plot_metric(
    [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
    ],
    "Precision & Recall Over Epochs",
    "precision_recall"
)

# --------------------------
# 4. mAP50 & mAP50-95
# --------------------------
plot_metric(
    [
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ],
    "mAP Over Epochs",
    "map_scores"
)
