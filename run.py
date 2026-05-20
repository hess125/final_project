"""
run.py — CogHealth demo launcher. Direct feature-based scoring.
Responds to behavior within 2 minutes. No LSTM warm-up issues.
"""
import json, logging, signal, sys, threading, time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-8s %(message)s"
)
logger = logging.getLogger("run")

# ── Config overrides — must happen before any other import ────────────────────
import config
config.WINDOW_SIZE_SECONDS    = 60
config.WINDOW_STEP_SECONDS    = 10
config.SEQUENCE_LENGTH        = 5
config.INFERENCE_INTERVAL_SEC = 10

from collector import BehavioralCollector
from features  import RollingNormalizer, SequenceBuilder, extract_features, save_feature_vector
import server as web_server
from config import WINDOW_SIZE_SECONDS, WINDOW_STEP_SECONDS, SEQUENCE_LENGTH, INFERENCE_INTERVAL_SEC, FEATURE_NAMES

# ── State ─────────────────────────────────────────────────────────────────────
smoothed_score = 0.0
start_ts       = time.time()
stop_evt       = threading.Event()
last_level     = "baseline"

# ── Direct feature-based scoring (no LSTM, no threshold needed) ───────────────
def run_inference(sequence: np.ndarray) -> dict:
    global smoothed_score

    latest = sequence[-1]   # most recent feature vector

    # Features by index — matches FEATURE_NAMES
    typing_speed  = float(latest[0])   # Z-scored WPM
    dwell_std     = float(latest[2])   # keystroke hold variability
    flight_std    = float(latest[4])   # inter-key timing variability
    error_rate    = float(latest[5])   # backspace ratio
    rhythm_cv     = float(latest[9])   # rhythm irregularity
    mouse_spd_std = float(latest[11])  # mouse speed variability
    mouse_acc     = float(latest[12])  # mouse acceleration
    idle_ratio    = float(latest[16])  # fraction of time mouse idle
    path_eff      = float(latest[17])  # mouse path straightness

    # Each component scored 0-1
    # Positive Z-score on typing speed means faster than YOUR baseline
    speed_stress  = max(0.0, min(1.0,  typing_speed * 0.4 + 0.2))
    error_stress  = max(0.0, min(1.0,  abs(error_rate) * 0.5 + abs(dwell_std) * 0.3))
    rhythm_stress = max(0.0, min(1.0,  abs(rhythm_cv)  * 0.4 + abs(flight_std) * 0.3))
    mouse_stress  = max(0.0, min(1.0,  abs(mouse_spd_std) * 0.3 + abs(mouse_acc) * 0.2 +
                                       (1.0 - min(1.0, max(0.0, path_eff + 1.0) / 2.0)) * 0.3))

    # Idle reduces stress — you're not doing anything stressful
    calm_penalty  = min(0.3, max(0.0, idle_ratio * 0.4))

    raw_score = (
        speed_stress  * 0.30 +
        error_stress  * 0.25 +
        rhythm_stress * 0.25 +
        mouse_stress  * 0.20 -
        calm_penalty
    )
    raw_score = max(0.0, min(1.0, raw_score))

    # Smooth with alpha=0.5 — fast reaction
    smoothed_score = 0.5 * raw_score + 0.5 * smoothed_score
    smoothed_score = max(0.0, min(1.0, smoothed_score))

    if   smoothed_score < 0.45: level = "low"
    elif smoothed_score < 0.55: level = "moderate"
    elif smoothed_score < 0.80: level = "elevated"
    else:                       level = "high"

    logger.info(
        "Score components: speed=%.2f error=%.2f rhythm=%.2f mouse=%.2f calm=-%.2f → raw=%.3f smooth=%.3f %s",
        speed_stress, error_stress, rhythm_stress, mouse_stress, calm_penalty,
        raw_score, smoothed_score, level
    )

    return {
        "smoothed_score": round(smoothed_score, 4),
        "raw_score":      round(raw_score, 4),
        "mse":            round(raw_score, 6),
        "stress_level":   level,
        "is_anomaly":     smoothed_score > 0.55,
        "latency_ms":     0.0,
    }

# ── Notification ───────────────────────────────────────────────────────────────
def notify(level, score):
    try:
        from plyer import notification
        notification.notify(
            title="CogHealth — Stress Alert",
            message=f"Level: {level.upper()}  |  Score: {score:.2f}\nConsider a short break.",
            app_name="CogHealth", timeout=10
        )
    except Exception:
        logger.warning("STRESS ALERT: level=%s score=%.2f", level, score)

# ── Components ─────────────────────────────────────────────────────────────────
collector   = BehavioralCollector()
normalizer  = RollingNormalizer()
seq_builder = SequenceBuilder(SEQUENCE_LENGTH)

# ── Feature loop ───────────────────────────────────────────────────────────────
def feature_loop():
    logger.info("Feature loop — window=%ds step=%ds", WINDOW_SIZE_SECONDS, WINDOW_STEP_SECONDS)
    while not stop_evt.wait(WINDOW_STEP_SECONDS):
        try:
            end_ts = time.time()
            raw = extract_features(end_ts - WINDOW_SIZE_SECONDS, end_ts)
            if raw is None:
                logger.info("Waiting for events...")
                continue
            normalizer.update(end_ts, raw)
            norm = normalizer.transform(raw)
            save_feature_vector(end_ts, raw, norm)
            seq_builder.push(norm)
            logger.info("Features ✓  fill=%.0f%%  typing_speed_raw=%.1f  error_raw=%.3f",
                        seq_builder.fill_ratio * 100, raw[0], raw[5])
        except Exception as e:
            logger.error("Feature error: %s", e, exc_info=True)

# ── Inference loop ─────────────────────────────────────────────────────────────
def inference_loop():
    global last_level
    logger.info("Inference loop — interval=%ds", INFERENCE_INTERVAL_SEC)
    while not stop_evt.wait(INFERENCE_INTERVAL_SEC):
        try:
            seq = seq_builder.get_sequence()
            if seq is None:
                logger.info("Sequence %.0f%% full...", seq_builder.fill_ratio * 100)
                continue

            result = run_inference(seq)

            feat_dict = {}
            last_seq = seq_builder.get_sequence()
            if last_seq is not None:
                for i, name in enumerate(FEATURE_NAMES):
                    feat_dict[name] = round(float(last_seq[-1, i]), 4)

            web_server.update_state({
                "stress_score":  result["smoothed_score"],
                "stress_level":  result["stress_level"],
                "is_anomaly":    result["is_anomaly"],
                "latency_ms":    result["latency_ms"],
                "uptime_s":      int(time.time() - start_ts),
                "features":      feat_dict,
                "calibrated":    True,
            })

            level = result["stress_level"]
            if level in ("elevated", "high") and level != last_level:
                notify(level, result["smoothed_score"])
            last_level = level

        except Exception as e:
            logger.error("Inference error: %s", e, exc_info=True)

def uptime_loop():
    while not stop_evt.wait(5):
        web_server._state["uptime_s"] = int(time.time() - start_ts)

# ── Start ──────────────────────────────────────────────────────────────────────
collector.start()
threading.Thread(target=feature_loop,          daemon=True, name="features").start()
threading.Thread(target=inference_loop,        daemon=True, name="inference").start()
threading.Thread(target=uptime_loop,           daemon=True, name="uptime").start()
threading.Thread(target=web_server.run_server, daemon=True, name="flask").start()

logger.info("=" * 55)
logger.info("  CogHealth DEMO MODE — direct feature scoring")
logger.info("  Dashboard → http://localhost:5000")
logger.info("  First scores appear in ~2 minutes")
logger.info("  IDLE=low | NORMAL=low-moderate | FAST+ERRATIC=elevated-high")
logger.info("=" * 55)

def _shutdown(sig, frame):
    stop_evt.set(); collector.stop(); sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

while True:
    time.sleep(10)