"""Download and prepare the real MNIST and AudioMNIST data used by the assignment.

The script creates:
  mnist_small.npz
  mnist_small_matlab.mat
  audio/0.wav ... audio/9.wav (mono, 4 kHz, exactly 2,983 samples)
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_raw"
AUDIO_DIR = ROOT / "audio"

MNIST_URLS = {
    "images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}
AUDIO_BASE = "https://raw.githubusercontent.com/soerenab/AudioMNIST/master/data/01"


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic number: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic number: {magic}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError("Label count does not match IDX header")
    return labels


def read_pcm_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width == 1:
        signal = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        signal = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        signal = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    return sample_rate, signal


def write_pcm16(path: Path, signal: np.ndarray, sample_rate: int = 4000) -> None:
    pcm = np.clip(signal, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def resample_exact(signal: np.ndarray, length: int = 2983) -> np.ndarray:
    if signal.size < 2:
        raise ValueError("Audio recording is too short")
    source_x = np.linspace(0.0, 1.0, signal.size, dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, length, dtype=np.float64)
    result = np.interp(target_x, source_x, signal).astype(np.float32)
    result -= result.mean()
    peak = float(np.max(np.abs(result)))
    if peak > 0:
        result *= 0.95 / peak
    return result


def main() -> None:
    RAW.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)

    image_gz = RAW / "train-images-idx3-ubyte.gz"
    label_gz = RAW / "train-labels-idx1-ubyte.gz"
    download(MNIST_URLS["images"], image_gz)
    download(MNIST_URLS["labels"], label_gz)

    images = read_idx_images(image_gz)[:10000]
    labels = read_idx_labels(label_gz)[:10000]
    train_images, test_images = images[:9000], images[9000:]
    train_labels, test_labels = labels[:9000], labels[9000:]

    np.savez_compressed(
        ROOT / "mnist_small.npz",
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    # MATLAB-compatible layout: 28 x 28 x N images and 10 x N one-hot labels.
    train_onehot = np.eye(10, dtype=np.float32)[train_labels].T
    test_onehot = np.eye(10, dtype=np.float32)[test_labels].T
    savemat(
        ROOT / "mnist_small_matlab.mat",
        {
            "trainData": np.transpose(train_images, (1, 2, 0)),
            "trainLabels": train_onehot,
            "testData": np.transpose(test_images, (1, 2, 0)),
            "testLabels": test_onehot,
        },
        do_compression=True,
    )

    for digit in range(10):
        raw_path = RAW / f"{digit}_01_0.wav"
        download(f"{AUDIO_BASE}/{digit}_01_0.wav", raw_path)
        original_rate, signal = read_pcm_wav(raw_path)
        processed = resample_exact(signal, 2983)
        output_path = AUDIO_DIR / f"{digit}.wav"
        write_pcm16(output_path, processed, 4000)
        print(
            f"Prepared {output_path.name}: source={original_rate} Hz, "
            f"output=4000 Hz, samples={processed.size}",
            flush=True,
        )

    print("Data preparation complete: 9,000 train + 1,000 test MNIST images.", flush=True)


if __name__ == "__main__":
    main()
