"""
config.py — Central configuration for the CogHealth edge system.
All tunable parameters, paths, and constants live here.
"""

import os
from pathlib import Path

# ─── Directories ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"          # auto-deleted after 1 h
FEATURES_DIR    = DATA_DIR / "features"
MODELS_DIR      = BASE_DIR / "models"
LOGS_DIR        = BASE_DIR / "logs"
STATIC_DIR      = BASE_DIR / "web" / "static"
TEMPLATES_DIR   = BASE_DIR / "web" / "templates"

for _d in [RAW_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Privacy ──────────────────────────────────────────────────────────────────
RAW_RETENTION_SECONDS   = 3600          # 1 hour hard limit
PURGE_INTERVAL_SECONDS  = 300           # check every 5 min
STORE_KEY_VALUES        = False         # NEVER store actual key characters
STORE_CLICK_TARGETS     = False         # NEVER store element targets

# ─── Feature Extraction ───────────────────────────────────────────────────────
WINDOW_SIZE_SECONDS     = 1800           # 30-minute sliding window [1800] (Currently 5 minutes instead of 30)
WINDOW_STEP_SECONDS     = 60            # 1-minute step [60] (Now extract every 20 seconds)
ROLLING_NORM_DAYS       = 7             # 7-day rolling Z-score
N_FEATURES              = 18

FEATURE_NAMES = [
    "typing_speed_wpm",
    "keystroke_dwell_mean",
    "keystroke_dwell_std",
    "keystroke_flight_mean",
    "keystroke_flight_std",
    "error_rate",
    "backspace_ratio",
    "burst_count",
    "pause_count_long",
    "typing_rhythm_cv",
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_acceleration_mean",
    "click_rate",
    "double_click_rate",
    "scroll_velocity_mean",
    "mouse_idle_ratio",
    "mouse_path_efficiency",
]

# ─── LSTM Autoencoder ─────────────────────────────────────────────────────────
SEQUENCE_LENGTH         = 30            # 30 time-steps per sequence (30 min) [30] (Now shorter sequence)
LSTM_UNITS              = [64, 32]      # encoder units (decoder mirrors)
LATENT_DIM              = 16
DROPOUT_RATE            = 0.2
LEARNING_RATE           = 1e-3
BATCH_SIZE              = 32
PRE_TRAIN_EPOCHS        = 50
FINE_TUNE_EPOCHS        = 20
FINE_TUNE_LR            = 1e-4
RECONSTRUCTION_THRESHOLD_PERCENTILE = 95  # anomaly if error > 95th pct of baseline

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_INTERVAL_SEC  = 60            # infer every 60 seconds [60] (Now every 20 seconds)
TFLITE_MODEL_PATH       = MODELS_DIR / "autoencoder.tflite"
KERAS_MODEL_PATH        = MODELS_DIR / "autoencoder.h5"
THRESHOLD_PATH          = MODELS_DIR / "threshold.json"
SCALER_PATH             = MODELS_DIR / "scaler.pkl"
BASELINE_DAYS           = 7

# ─── Anomaly Score ────────────────────────────────────────────────────────────
SCORE_SMOOTHING_ALPHA   = 0.3           # EWM smoothing for output
STRESS_LEVELS           = {             # score ranges → level label
    (0.0, 0.25): "low",
    (0.25, 0.55): "moderate",
    (0.55, 0.80): "elevated",
    (0.80, 1.01): "high",
}

# ─── RGB LED ─────────────────────────────────────────────────────────────────
'''
GPIO_RED   = None
GPIO_GREEN = None
GPIO_BLUE  = None              # Hz
PWM_FREQ = 1000

LED_COLORS = {                          # RGB 0-100 (duty cycle %)
    "low":      (0,  80, 40),           # calm teal
    "moderate": (60, 70,  0),           # amber-yellow
    "elevated": (90, 30,  0),           # orange-red
    "high":     (100, 0,  5),           # red with faint violet
    "baseline": (0,  20, 80),           # cool blue (calibrating)
    "offline":  (5,   5,  5),           # dim white
}
LED_TRANSITION_SEC      = 10            # cubic-ease duration
'''

# ─── Flask Web Server ─────────────────────────────────────────────────────────
FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 5000
SECRET_KEY      = os.environ.get("COGHEALTH_SECRET", "change-me-in-production")
POLL_INTERVAL   = 2000                  # ms — client polling interval

# ─── Self-Report ──────────────────────────────────────────────────────────────
SELF_REPORT_DB  = DATA_DIR / "self_reports.db"
REPORT_SCALE    = (1, 10)               # PSS-inspired 1-10 scale

# ─── System Health ────────────────────────────────────────────────────────────
HEALTH_CHECK_INTERVAL   = 30            # seconds
CPU_WARN_THRESHOLD      = 85            # %
MEM_WARN_THRESHOLD      = 85            # %
WATCHDOG_TIMEOUT        = 120           # seconds before restart

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL   = "INFO"
LOG_MAX_MB  = 10
LOG_BACKUPS = 5
