"""
model.py — LSTM Autoencoder for behavioral stress anomaly detection.

Architecture: bidirectional LSTM encoder → latent dense → LSTM decoder
Training: pre-train on CMU/Balabit public datasets, fine-tune on personal baseline
Export: TensorFlow Lite with INT8 quantization for Raspberry Pi inference
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    KERAS_MODEL_PATH,
    LATENT_DIM,
    LEARNING_RATE,
    LSTM_UNITS,
    N_FEATURES,
    PRE_TRAIN_EPOCHS,
    RECONSTRUCTION_THRESHOLD_PERCENTILE,
    SEQUENCE_LENGTH,
    THRESHOLD_PATH,
    TFLITE_MODEL_PATH,
)

logger = logging.getLogger(__name__)


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_autoencoder(
    seq_len:   int = SEQUENCE_LENGTH,
    n_feat:    int = N_FEATURES,
    units:     List[int] = LSTM_UNITS,
    latent:    int = LATENT_DIM,
    dropout:   float = DROPOUT_RATE,
) -> keras.Model:
    """
    Build LSTM Autoencoder.

    Encoder: BiLSTM stack → Dense latent
    Decoder: RepeatVector → LSTM stack → TimeDistributed Dense

    Input shape:  (batch, seq_len, n_feat)
    Output shape: (batch, seq_len, n_feat)  ← reconstruction
    """
    inp = keras.Input(shape=(seq_len, n_feat), name="input")

    # ── Encoder ──────────────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(units[0], return_sequences=True, name="enc_lstm_1"),
        name="bi_enc_1",
    )(inp)
    x = layers.Dropout(dropout, name="enc_drop_1")(x)
    x = layers.Bidirectional(
        layers.LSTM(units[1], return_sequences=False, name="enc_lstm_2"),
        name="bi_enc_2",
    )(x)
    x = layers.Dropout(dropout, name="enc_drop_2")(x)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    latent_vec = layers.Dense(latent, activation="relu", name="latent")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    y = layers.RepeatVector(seq_len, name="repeat")(latent_vec)
    y = layers.LSTM(units[1], return_sequences=True, name="dec_lstm_1")(y)
    y = layers.Dropout(dropout, name="dec_drop_1")(y)
    y = layers.LSTM(units[0], return_sequences=True, name="dec_lstm_2")(y)
    y = layers.Dropout(dropout, name="dec_drop_2")(y)
    out = layers.TimeDistributed(
        layers.Dense(n_feat, activation="linear"), name="reconstruction"
    )(y)

    model = keras.Model(inputs=inp, outputs=out, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    logger.info("Autoencoder built: %d params", model.count_params())
    return model


# ─── Synthetic Pre-training Data ──────────────────────────────────────────────

def generate_synthetic_baseline(
    n_samples: int = 5000,
    seq_len: int = SEQUENCE_LENGTH,
    n_feat: int = N_FEATURES,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic behavioral sequences representing normal work patterns.
    Used when public datasets (CMU/Balabit) are unavailable.

    Encodes:
    - Circadian rhythm in typing speed
    - Correlations between related features
    - Realistic value ranges
    """
    rng = np.random.default_rng(seed)

    sequences = []
    for _ in range(n_samples):
        # Base typing speed: 30-80 WPM with slow drift
        speed_base = rng.uniform(30, 80)
        time_vec   = np.linspace(0, 1, seq_len)

        # Feature-specific baselines with realistic correlations
        typing_speed     = speed_base + rng.normal(0, 5, seq_len)
        dwell_mean       = rng.uniform(0.08, 0.14) + rng.normal(0, 0.01, seq_len)
        dwell_std        = dwell_mean * rng.uniform(0.1, 0.3) + rng.normal(0, 0.005, seq_len)
        flight_mean      = rng.uniform(0.15, 0.35) + rng.normal(0, 0.02, seq_len)
        flight_std       = flight_mean * rng.uniform(0.2, 0.5) + rng.normal(0, 0.01, seq_len)
        error_rate       = rng.uniform(0.02, 0.08) + rng.normal(0, 0.01, seq_len)
        backspace_ratio  = error_rate * rng.uniform(0.9, 1.1)
        burst_count      = rng.uniform(3, 15) + rng.normal(0, 1, seq_len)
        pause_count      = rng.uniform(0, 5)  + rng.normal(0, 0.5, seq_len)
        rhythm_cv        = rng.uniform(0.2, 0.6) + rng.normal(0, 0.05, seq_len)
        mouse_speed_m    = rng.uniform(100, 600) + rng.normal(0, 50, seq_len)
        mouse_speed_s    = mouse_speed_m * rng.uniform(0.3, 0.7)
        mouse_acc_m      = rng.uniform(500, 3000) + rng.normal(0, 200, seq_len)
        click_rate       = rng.uniform(2, 10) + rng.normal(0, 1, seq_len)
        dclick_rate      = click_rate * rng.uniform(0.05, 0.15)
        scroll_vel       = rng.uniform(1, 8) + rng.normal(0, 0.5, seq_len)
        idle_ratio       = rng.uniform(0.1, 0.4) + rng.normal(0, 0.05, seq_len)
        path_eff         = rng.uniform(0.5, 0.9) + rng.normal(0, 0.05, seq_len)

        seq = np.stack([
            typing_speed, dwell_mean, dwell_std, flight_mean, flight_std,
            error_rate, backspace_ratio, burst_count, pause_count, rhythm_cv,
            mouse_speed_m, mouse_speed_s, mouse_acc_m, click_rate, dclick_rate,
            scroll_vel, idle_ratio, path_eff,
        ], axis=1).astype(np.float32)

        # Clip to plausible ranges
        seq = np.clip(seq, -5, 5)
        sequences.append(seq)

    return np.stack(sequences)   # (n_samples, seq_len, n_feat)


def load_cmu_dataset(data_dir: Path) -> Optional[np.ndarray]:
    """
    Load the CMU Keystroke Dynamics dataset if present.
    Expected format: preprocessed .npy file at data_dir/cmu_sequences.npy

    Returns (N, seq_len, n_feat) or None if unavailable.
    """
    fpath = data_dir / "cmu_sequences.npy"
    if not fpath.exists():
        logger.info("CMU dataset not found at %s — using synthetic data", fpath)
        return None
    try:
        data = np.load(fpath).astype(np.float32)
        logger.info("Loaded CMU dataset: shape=%s", data.shape)
        return data
    except Exception as exc:
        logger.warning("CMU dataset load failed: %s", exc)
        return None


# ─── Training ─────────────────────────────────────────────────────────────────

def pretrain(
    model: keras.Model,
    data_dir: Path,
    epochs: int = PRE_TRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> keras.callbacks.History:
    """
    Pre-train the autoencoder on population-level behavioral data.
    Falls back to synthetic data if public datasets are unavailable.
    """
    data = load_cmu_dataset(data_dir)
    if data is None:
        logger.info("Generating synthetic pre-training data")
        data = generate_synthetic_baseline(n_samples=8000)

    # Train/val split
    n_val = max(1, int(len(data) * 0.15))
    idx   = np.random.permutation(len(data))
    x_train = data[idx[n_val:]]
    x_val   = data[idx[:n_val]]

    logger.info("Pre-training: %d train / %d val sequences", len(x_train), len(x_val))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    history = model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Validation reconstruction accuracy
    pred = model.predict(x_val, verbose=0)
    mse  = np.mean((x_val - pred) ** 2, axis=(1, 2))
    reconstruction_acc = float(np.mean(mse < np.percentile(mse, 85)))
    logger.info("Pre-train val reconstruction accuracy: %.1f%%", reconstruction_acc * 100)
    return history


def fine_tune(
    model: keras.Model,
    personal_sequences: np.ndarray,
    epochs: int = FINE_TUNE_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> keras.callbacks.History:
    """
    Personalize the model on individual baseline sequences.
    Freezes encoder layers; only fine-tunes decoder + latent.
    """
    # Freeze encoder
    for layer in model.layers:
        if "enc" in layer.name or "bi_enc" in layer.name:
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="mse",
        metrics=["mae"],
    )

    n_val = max(1, int(len(personal_sequences) * 0.2))
    idx   = np.random.permutation(len(personal_sequences))
    x_train = personal_sequences[idx[n_val:]]
    x_val   = personal_sequences[idx[:n_val]]

    logger.info("Fine-tuning: %d personal sequences", len(x_train))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    history = model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Re-enable all layers after fine-tuning
    for layer in model.layers:
        layer.trainable = True

    return history


def compute_threshold(model: keras.Model, baseline_sequences: np.ndarray) -> float:
    """
    Compute anomaly threshold as the 95th percentile of reconstruction error
    on baseline (normal) sequences.
    """
    pred = model.predict(baseline_sequences, verbose=0, batch_size=BATCH_SIZE)
    mse  = np.mean((baseline_sequences - pred) ** 2, axis=(1, 2))
    threshold = float(np.percentile(mse, RECONSTRUCTION_THRESHOLD_PERCENTILE))
    logger.info(
        "Anomaly threshold: %.6f (p%d of %d baseline sequences)",
        threshold, RECONSTRUCTION_THRESHOLD_PERCENTILE, len(baseline_sequences)
    )
    return threshold


def save_threshold(threshold: float, path: Path = THRESHOLD_PATH) -> None:
    with open(path, "w") as fh:
        json.dump({"threshold": threshold, "computed_ts": time.time()}, fh)


def load_threshold(path: Path = THRESHOLD_PATH) -> Optional[float]:
    if not path.exists():
        return None
    with open(path) as fh:
        data = json.load(fh)
    return float(data["threshold"])


# ─── TFLite Conversion ────────────────────────────────────────────────────────

def export_tflite(
    model: keras.Model,
    output_path: Path = TFLITE_MODEL_PATH,
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
) -> None:
    """
    Convert Keras model to TFLite with optional INT8 quantization.
    INT8 quantization reduces model size ~4× and speeds up inference on ARM.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize and representative_data is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.inference_input_type  = tf.float32   # keep float I/O
        converter.inference_output_type = tf.float32

        def _representative_dataset():
            n = min(200, len(representative_data))
            for i in range(n):
                yield [representative_data[i:i+1].astype(np.float32)]

        converter.representative_dataset = _representative_dataset
        logger.info("TFLite: INT8 quantization with %d calibration samples",
                    min(200, len(representative_data) if representative_data is not None else 0))
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    logger.info("TFLite model saved → %s (%.1f KB)", output_path, size_kb)


# ─── Real-Time Inference Engine ───────────────────────────────────────────────

class InferenceEngine:
    """
    Loads a TFLite model and runs reconstruction-error anomaly scoring.
    Maintains an EWM-smoothed stress score for output.
    """

    def __init__(
        self,
        tflite_path: Path = TFLITE_MODEL_PATH,
        threshold_path: Path = THRESHOLD_PATH,
        smoothing_alpha: float = 0.3,
    ):
        self._tflite_path    = tflite_path
        self._keras_path     = KERAS_MODEL_PATH      
        self._threshold_path = threshold_path
        self._alpha          = smoothing_alpha
        self._interpreter    = None
        self._keras_model    = None                 
        self._use_keras      = False             
        self._input_idx      = 0
        self._output_idx     = 0
        self._threshold      = 1.0
        self._smoothed_score = 0.0
        self._last_raw_score = 0.0
        self._ready          = False

    def load(self) -> bool:
        try:
            self._threshold = load_threshold(self._threshold_path) or 1.0

            # Try TFLite first
            try:
                self._interpreter = tf.lite.Interpreter(
                    model_path=str(self._tflite_path),
                    num_threads=2,
                )
                self._interpreter.allocate_tensors()
                details = self._interpreter.get_input_details()
                self._input_idx  = details[0]["index"]
                self._output_idx = self._interpreter.get_output_details()[0]["index"]
                self._use_keras  = False
                logger.info("InferenceEngine: TFLite loaded (threshold=%.6f)", self._threshold)

            except Exception as tflite_err:
                # Flex delegate not available — fall back to Keras
                logger.warning("TFLite unavailable (%s) — falling back to Keras", tflite_err)
                self._keras_model = tf.keras.models.load_model(
                    str(self._keras_path),
                    compile=False        # skips deserializing metrics/optimizer — fixes version mismatch
                )
                self._use_keras   = True
                logger.info("InferenceEngine: Keras fallback loaded (threshold=%.6f)", self._threshold)

            self._ready = True
            return True

        except Exception as exc:
            logger.error("InferenceEngine load failed: %s", exc)
            self._ready = False
            return False
    
    def infer(self, sequence: np.ndarray) -> Optional[dict]:
        """
        Run one inference pass.

        Args:
            sequence: (seq_len, n_feat) float32 array

        Returns:
            dict with keys: raw_score, smoothed_score, is_anomaly, stress_level
        """
        if not self._ready:
            logger.warning("InferenceEngine not ready")
            return None

        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]   # add batch dim

        t0 = time.perf_counter()

        try:
            if self._use_keras:
                reconstruction = self._keras_model.predict(
                    sequence.astype(np.float32), verbose=0
                )
            else:
                self._interpreter.set_tensor(self._input_idx, sequence.astype(np.float32))
                self._interpreter.invoke()
                reconstruction = self._interpreter.get_tensor(self._output_idx)
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return None

        latency_ms = (time.perf_counter() - t0) * 1000

        mse        = float(np.mean((sequence - reconstruction) ** 2))
        raw_score  = min(1.0, mse / (self._threshold + 1e-9))
        self._smoothed_score = (
            self._alpha * raw_score + (1 - self._alpha) * self._smoothed_score
        )
        self._last_raw_score = raw_score

        stress_level = _score_to_level(self._smoothed_score)

        if latency_ms > 500:
            logger.warning("Inference latency %.1f ms exceeds 500 ms target", latency_ms)
        else:
            logger.debug("Inference: mse=%.6f raw=%.3f smooth=%.3f latency=%.1f ms",
                         mse, raw_score, self._smoothed_score, latency_ms)

        return {
            "raw_score":     raw_score,
            "smoothed_score": self._smoothed_score,
            "mse":           mse,
            "threshold":     self._threshold,
            "is_anomaly":    mse > self._threshold,
            "stress_level":  stress_level,
            "latency_ms":    round(latency_ms, 1),
            "ts":            time.time(),
        }

    def update_threshold(self, new_threshold: float) -> None:
        self._threshold = new_threshold
        save_threshold(new_threshold)
        logger.info("Threshold updated: %.6f", new_threshold)

    @property
    def smoothed_score(self) -> float:
        return self._smoothed_score

    @property
    def is_ready(self) -> bool:
        return self._ready


def _score_to_level(score: float) -> str:
    from config import STRESS_LEVELS
    for (lo, hi), label in STRESS_LEVELS.items():
        if lo <= score < hi:
            return label
    return "high"


# ─── CLI: full training pipeline ─────────────────────────────────────────────
def run_training_pipeline(data_dir: Path, model_dir: Path) -> None:
    """
    Build → pre-train → compute threshold → export TFLite.
    Saves all artefacts to model_dir.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    model = build_autoencoder()
    model.summary()

    logger.info("=== Phase 1: Pre-training ===")
    pretrain(model, data_dir)

    # Save Keras model
    model.save(str(KERAS_MODEL_PATH))
    logger.info("Keras model saved → %s", KERAS_MODEL_PATH)

    # Compute threshold on synthetic normal data
    logger.info("=== Phase 2: Threshold computation ===")
    baseline_data = generate_synthetic_baseline(n_samples=2000)
    threshold = compute_threshold(model, baseline_data)
    save_threshold(threshold)

    # Export TFLite
    logger.info("=== Phase 3: TFLite export ===")
    export_tflite(model, TFLITE_MODEL_PATH, quantize=True,
                  representative_data=baseline_data[:200])

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    import sys
    data_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models")
    run_training_pipeline(data_dir, model_dir)
