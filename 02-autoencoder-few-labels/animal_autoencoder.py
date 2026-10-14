"""Semi-supervised two-animal classification with autoencoder pretraining.

The folder names supplied with --class-a/--class-b are the original labels.  The
script hides most labels after a stratified train/test split:

1. Pretrain a convolutional autoencoder on every development image without labels.
2. Train a classifier with only a small labeled subset.
3. Compare it with an identical classifier trained directly by backpropagation.

No torchvision dependency is required; image loading uses Pillow.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


@dataclass
class Config:
    data_dir: str
    output_dir: str
    class_a: str | None
    class_b: str | None
    labeled_fraction: float = 0.10
    test_fraction: float = 0.20
    validation_fraction: float = 0.15
    max_per_class: int = 400
    image_size: int = 64
    batch_size: int = 32
    ae_epochs: int = 12
    classifier_epochs: int = 20
    bottleneck_dim: int = 128
    seed: int = 42
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_files_under(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_class_folders(data_dir: Path) -> dict[str, Path]:
    """Find leaf-like folders that contain image files.

    If duplicate folder names occur under train/test trees, only the largest folder
    is reported. This dataset is expected to have one directory per animal class.
    """
    candidates: dict[str, tuple[Path, int]] = {}
    for root, _, files in os.walk(data_dir):
        count = sum(Path(name).suffix.lower() in IMAGE_EXTENSIONS for name in files)
        if not count:
            continue
        path = Path(root)
        key = path.name
        if key not in candidates or count > candidates[key][1]:
            candidates[key] = (path, count)
    return {name: value[0] for name, value in sorted(candidates.items())}


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def resolve_class(requested: str, folders: dict[str, Path]) -> str:
    if requested in folders:
        return requested
    target = normalize_name(requested)
    exact = [name for name in folders if normalize_name(name) == target]
    if len(exact) == 1:
        return exact[0]
    partial = [name for name in folders if target in normalize_name(name)]
    if len(partial) == 1:
        return partial[0]
    raise ValueError(f"Could not uniquely match class {requested!r}. Available: {sorted(folders)}")


def choose_classes(folders: dict[str, Path], a: str | None, b: str | None) -> tuple[str, str]:
    names = list(folders)
    if len(names) < 2:
        raise ValueError(f"Found only {len(names)} class folders: {names}")
    if a and b:
        return resolve_class(a, folders), resolve_class(b, folders)

    # Prefer the intuitive dog-versus-cat assignment if both exist.
    dog = [n for n in names if normalize_name(n) in {"dog", "dogs"}]
    cat = [n for n in names if normalize_name(n) in {"cat", "cats"}]
    if dog and cat:
        return cat[0], dog[0]

    ranked = sorted(names, key=lambda n: len(image_files_under(folders[n])), reverse=True)
    return ranked[0], ranked[1]


def resize_and_crop(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def image_to_tensor(path: Path, size: int, augment: bool) -> torch.Tensor:
    with Image.open(path) as raw:
        image = resize_and_crop(raw, size)
    if augment:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class AnimalDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], image_size: int, augment: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        return image_to_tensor(path, self.image_size, self.augment), torch.tensor(label)


class Encoder(nn.Module):
    def __init__(self, image_size: int, bottleneck_dim: int):
        super().__init__()
        if image_size % 8:
            raise ValueError("image_size must be divisible by 8")
        self.feature_size = image_size // 8
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.feature_size * self.feature_size, bottleneck_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.conv(x))


class Autoencoder(nn.Module):
    def __init__(self, image_size: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = Encoder(image_size, bottleneck_dim)
        feature_size = image_size // 8
        self.feature_size = feature_size
        self.expand = nn.Sequential(
            nn.Linear(bottleneck_dim, 64 * feature_size * feature_size),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        hidden = self.expand(z).view(-1, 64, self.feature_size, self.feature_size)
        return self.decoder(hidden)


class Classifier(nn.Module):
    def __init__(self, encoder: Encoder, bottleneck_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(bottleneck_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def cap_and_split(
    files: list[Path], class_id: int, cfg: Config, rng: random.Random
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    files = files.copy()
    rng.shuffle(files)
    # Skip corrupt/non-image files gracefully while preserving the seeded order.
    valid_files = []
    target = cfg.max_per_class if cfg.max_per_class > 0 else len(files)
    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()
            valid_files.append(path)
            if len(valid_files) >= target:
                break
        except (OSError, ValueError):
            print(f"Skipping unreadable image: {path}")
    files = valid_files
    if len(files) < 20:
        raise ValueError(f"Class {class_id} has only {len(files)} usable images; at least 20 are needed")

    test_n = max(1, round(len(files) * cfg.test_fraction))
    test = [(p, class_id) for p in files[:test_n]]
    development = files[test_n:]
    val_n = max(1, round(len(development) * cfg.validation_fraction))
    validation = [(p, class_id) for p in development[:val_n]]
    train_pool = development[val_n:]
    labeled_n = max(2, round(len(train_pool) * cfg.labeled_fraction))
    labeled = [(p, class_id) for p in train_pool[:labeled_n]]
    # Pretraining is restricted to the training pool. Validation and test images
    # remain unseen, even without labels, for a strict inductive evaluation.
    unlabeled = [(p, class_id) for p in train_pool]
    return unlabeled, labeled, validation, test


def make_loader(samples: list[tuple[Path, int]], cfg: Config, shuffle: bool, augment: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        AnimalDataset(samples, cfg.image_size, augment),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=generator,
    )


def train_autoencoder(model: Autoencoder, loader: DataLoader, epochs: int, device: torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(images)
            loss = loss_fn(recon, images)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            seen += images.size(0)
        value = total / seen
        history.append(value)
        print(f"AE epoch {epoch + 1:02d}/{epochs}: reconstruction MSE={value:.5f}")
    return history


@torch.no_grad()
def evaluate(model: Classifier, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[float] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        probs = torch.softmax(logits, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(probs[:, 1].cpu().tolist())
    return total_loss / len(y_true), y_true, y_pred, probabilities


def train_classifier(
    model: Classifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss = running / seen
        val_loss, truth, pred, _ = evaluate(model, val_loader, device)
        val_acc = float(np.mean(np.asarray(truth) == np.asarray(pred)))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Classifier epoch {epoch + 1:02d}/{epochs}: "
            f"train loss={train_loss:.4f}, val loss={val_loss:.4f}, val accuracy={val_acc:.3f}"
        )
    model.load_state_dict(best_state)
    return history


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    cm = np.zeros((2, 2), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        cm[truth, pred] += 1
    per_class = []
    for cls in range(2):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": int(cm[cls].sum())})
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_precision": float(np.mean([m["precision"] for m in per_class])),
        "macro_recall": float(np.mean([m["recall"] for m in per_class])),
        "macro_f1": float(np.mean([m["f1"] for m in per_class])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def plot_split(split_counts: dict[str, int], output: Path) -> None:
    labels = list(split_counts)
    values = [split_counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylabel("Number of images")
    ax.set_title("Image roles (labeled train is a subset of the pretraining pool)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_histories(ae_history, direct_history, pretrained_history, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    axes[0].plot(range(1, len(ae_history) + 1), ae_history, color="#4C78A8")
    axes[0].set(title="Autoencoder pretraining", xlabel="Epoch", ylabel="Reconstruction MSE")
    for name, history, color in [
        ("Direct BP", direct_history, "#E45756"),
        ("AE pretrained", pretrained_history, "#54A24B"),
    ]:
        axes[1].plot(history["train_loss"], label=f"{name}: train", color=color, linestyle="-")
        axes[1].plot(history["val_loss"], label=f"{name}: validation", color=color, linestyle="--")
        axes[2].plot(history["val_accuracy"], label=name, color=color)
    axes[1].set(title="Classifier loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[2].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


@torch.no_grad()
def plot_reconstructions(model: Autoencoder, loader: DataLoader, device: torch.device, output: Path) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:6].to(device)
    recon = model(images).cpu().clamp(0, 1)
    images = images.cpu()
    fig, axes = plt.subplots(2, len(images), figsize=(12, 4.2))
    for i in range(len(images)):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[1, i].imshow(recon[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    fig.suptitle("Autoencoder reconstruction examples")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confusions(metrics: dict[str, dict], class_names: list[str], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.6))
    for ax, (title, result) in zip(axes, metrics.items()):
        cm = np.asarray(result["confusion_matrix"])
        image = ax.imshow(cm, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], class_names, rotation=20)
        ax.set_yticks([0, 1], class_names)
        ax.set(xlabel="Predicted", ylabel="Actual", title=title)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(cfg: Config) -> None:
    seed_everything(cfg.seed)
    started = time.time()
    data_dir = Path(cfg.data_dir).resolve()
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    folders = discover_class_folders(data_dir)
    class_a, class_b = choose_classes(folders, cfg.class_a, cfg.class_b)
    if class_a == class_b:
        raise ValueError("Choose two different classes")
    class_names = [class_a, class_b]
    print(f"Discovered {len(folders)} classes: {', '.join(folders)}")
    print(f"Selected classes: {class_a!r} and {class_b!r}")

    rng = random.Random(cfg.seed)
    split_parts = [cap_and_split(image_files_under(folders[name]), label, cfg, rng) for label, name in enumerate(class_names)]
    unlabeled = [sample for part in split_parts for sample in part[0]]
    labeled = [sample for part in split_parts for sample in part[1]]
    validation = [sample for part in split_parts for sample in part[2]]
    test = [sample for part in split_parts for sample in part[3]]
    for items in (unlabeled, labeled, validation, test):
        rng.shuffle(items)

    split_counts = {
        "Unlabeled pretrain": len(unlabeled),
        "Labeled train": len(labeled),
        "Validation": len(validation),
        "Held-out test": len(test),
    }
    print("Split sizes:", split_counts)
    print("Labeled counts by class:", Counter(label for _, label in labeled))

    train_unlabeled = make_loader(unlabeled, cfg, shuffle=True, augment=True)
    train_labeled = make_loader(labeled, cfg, shuffle=True, augment=True)
    validation_loader = make_loader(validation, cfg, shuffle=False, augment=False)
    test_loader = make_loader(test, cfg, shuffle=False, augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    autoencoder = Autoencoder(cfg.image_size, cfg.bottleneck_dim).to(device)
    ae_history = train_autoencoder(autoencoder, train_unlabeled, cfg.ae_epochs, device)
    plot_reconstructions(autoencoder, test_loader, device, figures / "reconstructions.png")

    # Same architecture and labeled images; only initialization differs.
    seed_everything(cfg.seed + 1)
    direct = Classifier(Encoder(cfg.image_size, cfg.bottleneck_dim), cfg.bottleneck_dim)
    direct_history = train_classifier(
        direct, train_labeled, validation_loader, cfg.classifier_epochs, device, lr=1e-3
    )
    seed_everything(cfg.seed + 1)
    pretrained_encoder = copy.deepcopy(autoencoder.encoder)
    pretrained = Classifier(pretrained_encoder, cfg.bottleneck_dim)
    pretrained_history = train_classifier(
        pretrained, train_labeled, validation_loader, cfg.classifier_epochs, device, lr=3e-4
    )

    test_results = {}
    prediction_rows = []
    for name, model in [("Direct BP", direct), ("AE pretrained", pretrained)]:
        _, truth, pred, probabilities = evaluate(model, test_loader, device)
        test_results[name] = classification_metrics(truth, pred)
        if name == "AE pretrained":
            for (path, _), actual, predicted, p1 in zip(test, truth, pred, probabilities):
                prediction_rows.append(
                    {
                        "file": str(path),
                        "actual": class_names[actual],
                        "predicted": class_names[predicted],
                        f"probability_{class_names[1]}": p1,
                    }
                )

    plot_split(split_counts, figures / "data_split.png")
    plot_histories(ae_history, direct_history, pretrained_history, figures / "training_curves.png")
    plot_confusions(test_results, class_names, figures / "confusion_matrices.png")
    torch.save(autoencoder.state_dict(), output / "autoencoder.pt")
    torch.save(pretrained.state_dict(), output / "pretrained_classifier.pt")
    torch.save(direct.state_dict(), output / "direct_bp_classifier.pt")

    result = {
        "experiment": "Two-class semi-supervised classification using autoencoder pretraining",
        "class_names": class_names,
        "available_classes": sorted(folders),
        "config": asdict(cfg),
        "device": str(device),
        "split_counts": split_counts,
        "class_source_counts": {name: len(image_files_under(folders[name])) for name in class_names},
        "test_metrics": test_results,
        "histories": {
            "autoencoder": ae_history,
            "direct_bp": direct_history,
            "ae_pretrained": pretrained_history,
        },
        "elapsed_seconds": time.time() - started,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "test_predictions.json").write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
    print(json.dumps(test_results, indent=2))
    print(f"Saved results to {output}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Extracted Kaggle dataset directory")
    parser.add_argument("--output-dir", default="results", help="Artifact directory")
    parser.add_argument("--class-a", help="First animal folder (default: Cat if present)")
    parser.add_argument("--class-b", help="Second animal folder (default: Dog if present)")
    parser.add_argument("--labeled-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=400, help="0 means no cap")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ae-epochs", type=int, default=12)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.labeled_fraction <= 1:
        parser.error("--labeled-fraction must be in (0, 1]")
    if not 0 < args.test_fraction < 1 or not 0 < args.validation_fraction < 1:
        parser.error("test and validation fractions must be in (0, 1)")
    return Config(**vars(args))


if __name__ == "__main__":
    main(parse_args())
