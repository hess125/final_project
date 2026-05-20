"""
orchestrator.py — Top-level process that coordinates:
  - Behavioral data collection
  - Feature extraction (sliding window)
  - LSTM inference
  - LED feedback
  - Web server state updates
  - Health monitoring + watchdog
  - Graceful shutdown
"""

import json
import logging
import logging.handlers
import multiprocessing
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import (
    BASELINE_DAYS,
    FEATURES_DIR,
    HEALTH_CHECK_INTERVAL,
    INFERENCE_INTERVAL_SEC,
    KERAS_MODEL_PATH,
    LOG_BACKUPS,
    LOG_LEVEL,
    LOG_MAX_MB,
    LOGS_DIR,
    MODELS_DIR,
    N_FEATURES,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    STRESS_LEVELS,
    TFLITE_MODEL_PATH,
    THRESHOLD_PATH,
    WATCHDOG_TIMEOUT,
    WINDOW_SIZE_SECONDS,
    WINDOW_STEP_SECONDS,
)

# ─── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging() -> None:
    
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(name)-20s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file
    fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "coghealth.log",
        maxBytes=LOG_MAX_MB * 1024 * 1024,
        backupCount=LOG_BACKUPS,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


setup_logging()
logger = logging.getLogger("orchestrator")


# ─── Deferred imports (after logging) ────────────────────────────────────────

from collector import BehavioralCollector
from features  import (
    RollingNormalizer,
    SequenceBuilder,
    extract_features,
    load_feature_history,
    save_feature_vector,
)
#from led    import LEDController
from model  import (
    InferenceEngine,
    build_autoencoder,
    compute_threshold,
    export_tflite,
    fine_tune,
    generate_synthetic_baseline,
    load_threshold,
    pretrain,
    save_threshold,
)
import server as web_server


# ─── Status File ─────────────────────────────────────────────────────────────

STATUS_PATH = LOGS_DIR / "status.json"

def write_status(data: dict) -> None:
    try:
        with open(STATUS_PATH, "w") as fh:
            json.dump({**data, "written_ts": time.time()}, fh)
    except OSError as exc:
        logger.debug("Status write failed: %s", exc)


# ─── Model Bootstrap ──────────────────────────────────────────────────────────

def ensure_model_ready() -> bool:
    """
    Check if TFLite model exists; if not, run pre-training pipeline.
    Returns True when model is ready.
    """
    if TFLITE_MODEL_PATH.exists() and THRESHOLD_PATH.exists():
        logger.info("Found existing TFLite model and threshold")
        return True

    logger.info("No trained model found — running initial pre-training (this takes ~5 min)")
    try:
        import tensorflow as tf
        model = build_autoencoder()
        pretrain(model, MODELS_DIR)
        model.save(str(KERAS_MODEL_PATH))

        baseline = generate_synthetic_baseline(n_samples=2000)
        threshold = compute_threshold(model, baseline)
        save_threshold(threshold)
        export_tflite(model, TFLITE_MODEL_PATH, quantize=True,
                      representative_data=baseline[:200])
        logger.info("Initial model training complete")
        return True
    except Exception as exc:
        logger.error("Model training failed: %s", exc, exc_info=True)
        return False


# ─── Personalization ──────────────────────────────────────────────────────────

def attempt_personalization(normalizer: RollingNormalizer) -> bool:
    """
    Fine-tune the model on collected personal baseline sequences.
    Triggered when ≥ BASELINE_DAYS of feature data is available.
    """
    history = load_feature_history(days=BASELINE_DAYS)
    if len(history) < SEQUENCE_LENGTH * 2:
        logger.info("Not enough personal data for fine-tuning (%d sequences)", len(history))
        return False

    logger.info("Starting personalization on %d sequences", len(history))
    import numpy as np
    import tensorflow as tf

    # Build normalized sequences
    raw_vecs = [v for _, v in history]
    normalizer_vecs = []
    for raw in raw_vecs:
        norm = normalizer.transform(raw)
        normalizer_vecs.append(norm)

    import numpy as np
    vecs = np.stack(normalizer_vecs)

    # Sliding window into sequences
    sequences = []
    for i in range(0, len(vecs) - SEQUENCE_LENGTH + 1, 5):
        sequences.append(vecs[i:i + SEQUENCE_LENGTH])

    if len(sequences) < 10:
        return False

    seqs = np.stack(sequences)

    try:
        model = tf.keras.models.load_model(str(KERAS_MODEL_PATH))
        fine_tune(model, seqs)
        model.save(str(KERAS_MODEL_PATH))
        threshold = compute_threshold(model, seqs)
        save_threshold(threshold)
        export_tflite(model, TFLITE_MODEL_PATH, quantize=True,
                      representative_data=seqs[:100])
        logger.info("Personalization complete — new threshold: %.6f", threshold)
        return True
    except Exception as exc:
        logger.error("Personalization failed: %s", exc, exc_info=True)
        return False


# ─── Main Orchestrator ────────────────────────────────────────────────────────

class CogHealthOrchestrator:

    def __init__(self):
        self._collector   = BehavioralCollector()
        self._normalizer  = RollingNormalizer()
        self._seq_builder = SequenceBuilder(SEQUENCE_LENGTH)
        self._engine      = InferenceEngine()
        #self._led         = LEDController()

        self._start_ts    = time.time()
        self._calibrated  = False
        self._running     = False

        self._feature_thread:   Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._health_thread:    Optional[threading.Thread] = None
        self._server_thread:    Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Watchdog
        self._last_inference_ts: float = time.time()

    # ── Startup ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("=== CogHealth Orchestrator Starting ===")

        logger.info("Step 1: Restoring normalizer state")
        self._normalizer.load()

        for ts, raw in load_feature_history(days=7):
            self._normalizer.update(ts, raw)

        logger.info("Step 2: Ensuring model ready")
        if not ensure_model_ready():
            logger.critical("Cannot start without a trained model")
            sys.exit(1)

        logger.info("Step 3: Loading inference engine")
        if not self._engine.load():
            logger.critical(
                "Failed to load inference engine. Ensure models/autoencoder.h5 exists."
            )
            sys.exit(1)

        logger.info("Step 4: Starting collector")
        self._collector.start()

        logger.info("Step 5: Starting threads")
        self._running = True
        self._feature_thread = threading.Thread(
            target=self._feature_loop, daemon=True, name="features"
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="inference"
        )
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="health"
        )
        self._server_thread = threading.Thread(
            target=web_server.run_server, daemon=True, name="flask"
        )

        self._feature_thread.start()
        self._inference_thread.start()
        self._health_thread.start()
        self._server_thread.start()

        logger.info("All subsystems started — monitoring active")

        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info("Step 7: Entering main loop")
        self._main_loop()

    # ── Feature Extraction Loop ───────────────────────────────────────────────

    def _feature_loop(self) -> None:
        """Extract features on WINDOW_STEP_SECONDS cadence."""
        logger.info("Feature loop started")
        while not self._stop_evt.wait(WINDOW_STEP_SECONDS):
            try:
                end_ts   = time.time()
                start_ts = end_ts - WINDOW_SIZE_SECONDS
                t0 = time.perf_counter()

                raw_features = extract_features(start_ts, end_ts)
                if raw_features is None:
                    continue

                # Normalise and update rolling scaler
                self._normalizer.update(end_ts, raw_features)
                norm_features = self._normalizer.transform(raw_features)

                # Persist feature record
                save_feature_vector(end_ts, raw_features, norm_features)

                # Push to sequence builder
                self._seq_builder.push(norm_features)

                feat_latency = (time.perf_counter() - t0) * 1000
                logger.debug("Feature step: %.1f ms | seq fill: %.0f%%",
                             feat_latency, self._seq_builder.fill_ratio * 100)

                # Log to web API for latency tracking
                self._log_latency("feature_extraction", feat_latency)

            except Exception as exc:
                logger.error("Feature loop error: %s", exc, exc_info=True)

        logger.info("Feature loop exited")

    # ── Inference Loop ────────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """Run LSTM inference on INFERENCE_INTERVAL_SEC cadence."""
        logger.info("Inference loop started")
        personalization_done = False

        while not self._stop_evt.wait(INFERENCE_INTERVAL_SEC):
            try:
                sequence = self._seq_builder.get_sequence()
                if sequence is None:
                    fill = self._seq_builder.fill_ratio
                    logger.debug("Sequence not ready (%.0f%%)", fill * 100)
                    continue

                t0 = time.perf_counter()
                result = self._engine.infer(sequence)
                total_ms = (time.perf_counter() - t0) * 1000

                if result is None:
                    continue

                self._last_inference_ts = time.time()
                stress_level = result["stress_level"]
                stress_score = result["smoothed_score"]

                # Update LED
                #self._led.set_level(stress_level)

                # Update web server state
                import numpy as np
                from config import FEATURE_NAMES
                last_feat = self._seq_builder.get_sequence()
                feat_dict = {}
                if last_feat is not None:
                    for i, name in enumerate(FEATURE_NAMES):
                        feat_dict[name] = round(float(last_feat[-1, i]), 4)

                web_server.update_state({
                    "stress_score":  stress_score,
                    "stress_level":  stress_level,
                    "is_anomaly":    result["is_anomaly"],
                    "latency_ms":    result["latency_ms"],
                    "uptime_s":      int(time.time() - self._start_ts),
                    "features":      feat_dict,
                    "calibrated":    self._calibrated,
                })

                write_status({
                    "stress_score": stress_score,
                    "stress_level": stress_level,
                    "is_anomaly":   result["is_anomaly"],
                    "uptime_s":     int(time.time() - self._start_ts),
                    "calibrated":   self._calibrated,
                })

                self._log_latency("inference", result["latency_ms"])
                logger.info(
                    "Inference ▸ score=%.3f level=%-8s anomaly=%s lat=%.1fms",
                    stress_score, stress_level, result["is_anomaly"], result["latency_ms"]
                )

                # Trigger personalization after BASELINE_DAYS
                if not personalization_done and not self._calibrated:
                    days_collected = (time.time() - self._start_ts) / 86400
                    if days_collected >= BASELINE_DAYS:
                        logger.info("Attempting personalization after %.1f days", days_collected)
                        if attempt_personalization(self._normalizer):
                            self._engine.load()   # reload updated TFLite
                            self._calibrated = True
                            personalization_done = True
                            logger.info("System calibrated ✓")
                            self._normalizer.save()

            except Exception as exc:
                logger.error("Inference loop error: %s", exc, exc_info=True)

        logger.info("Inference loop exited")

    # ── Health Monitor ────────────────────────────────────────────────────────

    def _health_loop(self) -> None:
        """Monitor CPU/memory, check watchdog, log system health."""
        import psutil
        from config import CPU_WARN_THRESHOLD, MEM_WARN_THRESHOLD
        logger.info("Health monitor started")

        while not self._stop_evt.wait(HEALTH_CHECK_INTERVAL):
            try:
                cpu  = psutil.cpu_percent(interval=1)
                mem  = psutil.virtual_memory().percent
                disk = psutil.disk_usage("/").percent

                if cpu > CPU_WARN_THRESHOLD:
                    logger.warning("High CPU: %.0f%%", cpu)
                if mem > MEM_WARN_THRESHOLD:
                    logger.warning("High MEM: %.0f%%", mem)
                if disk > 90:
                    logger.warning("High DISK: %.0f%%", disk)

                # Watchdog: if inference hasn't run recently, reload engine
                since_last = time.time() - self._last_inference_ts
                if since_last > WATCHDOG_TIMEOUT:
                    logger.warning("Watchdog: inference stalled (%.0f s) — reloading engine", since_last)
                    self._engine.load()
                    self._last_inference_ts = time.time()

                logger.debug("Health: cpu=%.0f%% mem=%.0f%% disk=%.0f%%", cpu, mem, disk)

            except Exception as exc:
                logger.error("Health monitor error: %s", exc, exc_info=True)

        logger.info("Health monitor exited")

    # ── Main Loop (watchdog for threads) ─────────────────────────────────────

    def _main_loop(self) -> None:
        while self._running:
            time.sleep(10)

            # Restart crashed threads
            threads = {
                "features":  (self._feature_thread,   self._feature_loop),
                "inference": (self._inference_thread,  self._inference_loop),
                "health":    (self._health_thread,     self._health_loop),
            }
            for name, (thread, target) in threads.items():
                if thread and not thread.is_alive():
                    logger.warning("Thread '%s' died — restarting", name)
                    new_thread = threading.Thread(target=target, daemon=True, name=name)
                    new_thread.start()
                    if name == "features":   self._feature_thread = new_thread
                    elif name == "inference": self._inference_thread = new_thread
                    elif name == "health":    self._health_thread = new_thread

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def _shutdown_handler(self, sig, frame) -> None:
        logger.info("Signal %s — initiating graceful shutdown", sig)
        self._running = False
        self._stop_evt.set()

        self._collector.stop()
        self._normalizer.save()
        #self._led.off()
        time.sleep(3)

        logger.info("CogHealth shutdown complete")
        sys.exit(0)

    # ── Utility ──────────────────────────────────────────────────────────────

    def _log_latency(self, component: str, latency_ms: float) -> None:
        try:
            import requests
            requests.post(
                f"http://localhost:5000/api/latency",
                json={"component": component, "latency_ms": latency_ms},
                timeout=0.5,
            )
        except Exception:
            pass   # Non-critical; server may not be ready yet


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    orch = CogHealthOrchestrator()
    orch.start()


if __name__ == "__main__":
    main()
