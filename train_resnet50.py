import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CharacterDataset
from src.model import CharacterModel


BATCH_SIZE = 32
NUM_WORKERS = 0
PATIENCE = 5
CHECKPOINT_PATH = "best_model.pth"


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_accuracy: float


def build_dataloaders(batch_size: int = BATCH_SIZE):
    train_dataset = CharacterDataset(split="train")
    val_dataset = CharacterDataset(split="val")
    test_dataset = CharacterDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def move_batch_to_device(batch, device):
    images, labels, _paths = batch
    return images.to(device), labels.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        images, labels = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct = 0

    for batch in loader:
        images, labels = move_batch_to_device(batch, device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)
    return avg_loss, accuracy


def save_checkpoint(model, phase, epoch, metrics):
    torch.save(
        {
            "phase": phase,
            "epoch": epoch,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "metrics": {
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "val_accuracy": metrics.val_accuracy,
            },
        },
        CHECKPOINT_PATH,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, _test_dataset, train_loader, val_loader, _test_loader = build_dataloaders()

    model = CharacterModel(num_classes=5, dropout_p=0.4).to(device)
    best_accuracy = float("-inf")

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for phase, max_epochs, use_scheduler in ((1, 5, False), (2, 20, True)):
        criterion = nn.CrossEntropyLoss()

        if phase == 1:
            model.freeze_backbone()
        else:
            model.unfreeze_from_layer3()

        optimizer = model.get_optimizer(phase)
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            if scheduler is not None:
                scheduler.step()

            print(
                f"Phase {phase} | Epoch {epoch}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    phase,
                    epoch,
                    EpochMetrics(train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy),
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"Phase {phase}: early stopping triggered after {epoch} epochs.")
                break

    print(f"Best checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()