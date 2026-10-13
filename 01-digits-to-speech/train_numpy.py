"""Manual NumPy backpropagation for the handwritten-digit-to-speech MLP."""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
ARCHITECTURE = [784, 512, 512, 1024, 2048, 2983]
LEARNING_RATE = 0.1
BATCH_SIZE = 100
ITERATIONS = 300
SEED = 2023521460


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def forward(weights: list[np.ndarray], x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the linear output and all activations (samples are columns)."""
    activations = [x]
    a = x
    for weight in weights[:-1]:
        a = sigmoid(weight @ a)
        activations.append(a)
    output = weights[-1] @ a
    activations.append(output)
    return output, activations


def initialize_weights(rng: np.random.Generator) -> list[np.ndarray]:
    weights = []
    for fan_in, fan_out in zip(ARCHITECTURE[:-1], ARCHITECTURE[1:]):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weight = rng.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float32)
        weights.append(weight)
    return weights


def read_audio_targets() -> tuple[np.ndarray, int]:
    signals = []
    sample_rate = None
    for digit in range(10):
        path = ROOT / "audio" / f"{digit}.wav"
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError(f"Expected mono 16-bit PCM: {path}")
            rate = wav.getframerate()
            signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
        if signal.size != 2983:
            raise ValueError(f"{path} has {signal.size} samples, expected 2983")
        sample_rate = rate if sample_rate is None else sample_rate
        if rate != sample_rate:
            raise ValueError("Audio sample rates are inconsistent")
        # Match the teacher's normalization: [-1, 1] to [0, 1].
        signals.append((signal / 32768.0 + 1.0) / 2.0)
    return np.stack(signals).astype(np.float32), int(sample_rate)


def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def make_figures(
    test_images: np.ndarray,
    test_labels: np.ndarray,
    audio_targets: np.ndarray,
    sample_rate: int,
    losses: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    predicted_label: int,
    true_label: int,
) -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    fig, axes = plt.subplots(2, 5, figsize=(8.0, 3.2))
    for digit, axis in enumerate(axes.flat):
        index = int(np.flatnonzero(test_labels == digit)[0])
        axis.imshow(test_images[index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"Digit {digit}")
        axis.axis("off")
    fig.suptitle("Real MNIST samples (held-out subset)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "fig1_mnist_samples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    t = np.arange(2983) / sample_rate
    fig, axis = plt.subplots(figsize=(8.0, 2.8))
    axis.plot(t, audio_targets[9] * 2.0 - 1.0, color="#1769aa", linewidth=0.75)
    axis.set(xlabel="Time (s)", ylabel="Amplitude", title="Real AudioMNIST recording: digit 9")
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_audio_waveform.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 3.0))
    axis.plot(np.arange(1, losses.size + 1), losses, color="#a51c30", linewidth=1.4)
    axis.set(xlabel="Mini-batch iteration", ylabel="MSE", title="Measured training loss (300 updates)")
    axis.grid(alpha=0.25)
    axis.annotate(
        f"Final MSE = {losses[-1]:.6f}",
        xy=(losses.size, losses[-1]),
        xytext=(-118, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 3.2))
    axis.plot(t, target, label=f"Target digit {true_label}", color="#1769aa", linewidth=0.9)
    axis.plot(t, prediction, label=f"Predicted waveform (nearest template: {predicted_label})", color="#e06b27", linewidth=0.8, alpha=0.86)
    axis.set(xlabel="Time (s)", ylabel="Amplitude", title="Measured test prediction versus target")
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_prediction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = np.load(ROOT / "mnist_small.npz")
    train_images = data["train_images"]
    train_labels = data["train_labels"].astype(np.int64)
    test_images = data["test_images"]
    test_labels = data["test_labels"].astype(np.int64)
    audio_targets, sample_rate = read_audio_targets()

    train_x = (train_images.reshape(train_images.shape[0], -1).astype(np.float32) / 255.0)
    test_x = (test_images.reshape(test_images.shape[0], -1).astype(np.float32) / 255.0)
    rng = np.random.default_rng(SEED)
    weights = initialize_weights(rng)
    losses = np.empty(ITERATIONS, dtype=np.float64)
    start = time.perf_counter()

    for iteration in range(ITERATIONS):
        indices = rng.choice(train_x.shape[0], size=BATCH_SIZE, replace=False)
        x = np.ascontiguousarray(train_x[indices].T)
        y = np.ascontiguousarray(audio_targets[train_labels[indices]].T)

        prediction, activations = forward(weights, x)
        error = prediction - y
        losses[iteration] = float(np.mean(error * error))

        # MSE derivative. The 1/output_size factor is part of the mean and
        # the 1/batch_size factor is applied when each weight gradient is formed.
        delta = (2.0 / ARCHITECTURE[-1]) * error
        gradients: list[np.ndarray] = [np.empty(0)] * len(weights)
        for layer in range(len(weights) - 1, -1, -1):
            gradients[layer] = (delta @ activations[layer].T) / BATCH_SIZE
            if layer > 0:
                hidden = activations[layer]
                delta = (weights[layer].T @ delta) * hidden * (1.0 - hidden)
        for layer in range(len(weights)):
            weights[layer] -= LEARNING_RATE * gradients[layer]

        if (iteration + 1) % 10 == 0 or iteration == 0:
            elapsed = time.perf_counter() - start
            print(
                f"Iteration {iteration + 1:3d}/{ITERATIONS}: "
                f"MSE={losses[iteration]:.8f}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    duration = time.perf_counter() - start
    np.save(RESULTS / "J.npy", losses)
    np.savez(RESULTS / "model_weights.npz", **{f"w{i + 1}": w for i, w in enumerate(weights)})

    # Deterministic held-out example: first test occurrence of digit 6.
    example_digit = 6
    test_index = int(np.flatnonzero(test_labels == example_digit)[0])
    test_prediction, _ = forward(weights, np.ascontiguousarray(test_x[test_index : test_index + 1].T))
    prediction_01 = test_prediction[:, 0]
    prediction_audio = np.clip(prediction_01 * 2.0 - 1.0, -1.0, 1.0)
    target_audio = audio_targets[example_digit] * 2.0 - 1.0
    template_mse = np.mean((audio_targets - prediction_01[None, :]) ** 2, axis=1)
    predicted_label = int(np.argmin(template_mse))
    test_mse = float(np.mean((prediction_01 - audio_targets[example_digit]) ** 2))
    write_wav(RESULTS / "predicted_digit_6.wav", prediction_audio, sample_rate)
    np.savez(
        RESULTS / "training_results.npz",
        J=losses,
        prediction=prediction_audio,
        target=target_audio,
        true_label=np.int64(example_digit),
        predicted_label=np.int64(predicted_label),
        test_mse=np.float64(test_mse),
        duration_seconds=np.float64(duration),
    )
    make_figures(
        test_images,
        test_labels,
        audio_targets,
        sample_rate,
        losses,
        target_audio,
        prediction_audio,
        predicted_label,
        example_digit,
    )
    summary = {
        "architecture": ARCHITECTURE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "iterations": ITERATIONS,
        "training_samples_available": int(train_x.shape[0]),
        "test_samples": int(test_x.shape[0]),
        "initial_mse": float(losses[0]),
        "final_mse": float(losses[-1]),
        "best_mse": float(losses.min()),
        "held_out_true_label": example_digit,
        "held_out_nearest_audio_template": predicted_label,
        "held_out_waveform_mse": test_mse,
        "training_seconds": duration,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
