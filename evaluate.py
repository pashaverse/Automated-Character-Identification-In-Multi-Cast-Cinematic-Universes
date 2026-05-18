from pathlib import Path
from typing import List

import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import scipy.sparse
import matplotlib.pyplot as plt

from src.dataset import CharacterDataset
from src.model import CharacterModel


BATCH_SIZE = 32
NUM_WORKERS = 0
CHECKPOINT = "best_model.pth"
THETA = 0.7
OUTPUT_DIR = Path("outputs")
CLASS_NAMES = ["dustin", "eleven", "hopper", "lucas", "steve"]
NUM_CLASSES = len(CLASS_NAMES)

# Type aliases for clarity after numpy conversion
NDArray = np.ndarray


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(model, path: str, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)


def build_loader():
    test_dataset = CharacterDataset(split="test")
    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return test_loader


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CharacterModel(num_classes=NUM_CLASSES, dropout_p=0.4).to(device)
    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    load_checkpoint(model, CHECKPOINT, device)
    model.eval()

    loader = build_loader()

    y_true_confident_list: List[int] = []
    y_pred_confident_list: List[int] = []
    y_score_confident_list: List[List[float]] = []
    uncertain_count = 0
    uncertain_entries: List[tuple] = []

    with torch.no_grad():
        for batch in loader:
            images, labels, _paths = batch
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for i in range(probs.shape[0]):
                p = probs[i]
                maxp = float(p.max())
                pred = int(p.argmax())
                true = int(labels_np[i])
                path = _paths[i]

                if maxp < THETA:
                    uncertain_count += 1
                    uncertain_entries.append((path, float(maxp), int(pred), int(true)))
                    continue

                y_true_confident_list.append(true)
                y_pred_confident_list.append(pred)
                y_score_confident_list.append(p.tolist())

    y_true_confident = np.array(y_true_confident_list)
    y_pred_confident = np.array(y_pred_confident_list)
    y_score_confident = np.array(y_score_confident_list)

    if len(y_true_confident) == 0:
        print("No confident predictions above threshold. Nothing to evaluate.")
        print(f"Uncertain count: {uncertain_count}")
        return

    # Metrics
    top1 = accuracy_score(y_true_confident, y_pred_confident)
    macro_f1 = f1_score(y_true_confident, y_pred_confident, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        y_true_confident, y_pred_confident, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    print(f"Top-1 Accuracy (confident only): {top1:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("Per-class F1-score:")
    per_class_f1_arr = np.atleast_1d(per_class_f1)
    for idx in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[idx]:<7}: {per_class_f1_arr[idx]:.4f}")
    print(f"Uncertain predictions (below {THETA}): {uncertain_count}")

    ensure_output_dir()

    # Save uncertain examples for manual review
    if uncertain_entries:
        csv_path = OUTPUT_DIR / "uncertain_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filepath", "max_prob", "predicted_label", "true_label"])
            for row in uncertain_entries:
                writer.writerow(row)
        print(f"Saved uncertain predictions to {csv_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true_confident, y_pred_confident, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    fig.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    # ROC One-vs-Rest
    y_true_binarized = label_binarize(y_true_confident, classes=list(range(NUM_CLASSES)))
    # Ensure dense array
    y_true_binarized = np.asarray(y_true_binarized)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(NUM_CLASSES):
        # skip if class not present
        if y_true_binarized[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_score_confident[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # compute macro & micro AUC if possible
    try:
        macro_auc = roc_auc_score(y_true_binarized, y_score_confident, average="macro", multi_class="ovr")
    except Exception:
        macro_auc = None
    try:
        micro_auc = roc_auc_score(y_true_binarized, y_score_confident, average="micro", multi_class="ovr")
    except Exception:
        micro_auc = None

    title = "ROC Curve (One-vs-Rest)"
    if macro_auc is not None and micro_auc is not None:
        title += f" — macro AUC: {macro_auc:.3f}, micro AUC: {micro_auc:.3f}"
    ax.set_title(title)
    ax.legend(loc="lower right")
    roc_path = OUTPUT_DIR / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)

    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved ROC curve to {roc_path}")


if __name__ == "__main__":
    evaluate()
