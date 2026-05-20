"""
features.py — Real-time 18-feature extraction pipeline.

Reads raw JSONL event files, computes behavioral features per 30-minute
sliding window, writes normalized feature vectors to FEATURES_DIR.
Adaptive normalization via rolling 7-day Z-score.
"""

import json
import logging
import math
import pickle
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    FEATURES_DIR,
    FEATURE_NAMES,
    N_FEATURES,
    RAW_DIR,
    ROLLING_NORM_DAYS,
    SCALER_PATH,
    WINDOW_SIZE_SECONDS,
    WINDOW_STEP_SECONDS,
)

logger = logging.getLogger(__name__)

# ─── Raw Event Loading ────────────────────────────────────────────────────────

def load_events_in_window(start_ts: float, end_ts: float) -> List[dict]:
    """Load all raw events whose timestamps fall within [start_ts, end_ts]."""
    events = []
    try:
        for fpath in sorted(RAW_DIR.glob("raw_*.jsonl")):
            with open(fpath) as fh:
                lines = fh.readlines()
            if not lines:
                continue
            meta = json.loads(lines[0])
            # Quick pre-filter: skip files created entirely outside window
            file_created = meta.get("created_ts", 0)
            if file_created > end_ts + WINDOW_SIZE_SECONDS:
                continue
            for line in lines[1:]:
                try:
                    ev = json.loads(line)
                    ts = ev.get("ts", 0)
                    if start_ts <= ts <= end_ts:
                        events.append(ev)
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("Event load error: %s", exc)

    events.sort(key=lambda e: e["ts"])
    return events


# ─── Feature Extraction Functions ────────────────────────────────────────────

def _typing_speed_wpm(key_events: List[dict], window_sec: float) -> float:
    """Words per minute: (key_press count / 5) / window_minutes."""
    presses = sum(1 for e in key_events if e["type"] == "key_press" and e["cat"] == "alpha")
    minutes = window_sec / 60.0
    return (presses / 5.0) / minutes if minutes > 0 else 0.0


def _dwell_times(key_events: List[dict]) -> List[float]:
    """Dwell = time between press and release of same key position."""
    return [e["dwell"] for e in key_events if e["type"] == "key_release"
            and e.get("dwell") is not None and 0 < e["dwell"] < 2.0]


def _flight_times(key_events: List[dict]) -> List[float]:
    """Flight = time between successive key presses."""
    press_ts = [e["ts"] for e in key_events if e["type"] == "key_press"]
    if len(press_ts) < 2:
        return []
    return [t2 - t1 for t1, t2 in zip(press_ts, press_ts[1:]) if 0 < t2 - t1 < 5.0]


def _error_rate(key_events: List[dict]) -> float:
    """Fraction of key presses that are backspace."""
    presses = [e for e in key_events if e["type"] == "key_press"]
    if not presses:
        return 0.0
    errors = sum(1 for e in presses if e["cat"] == "backspace")
    return errors / len(presses)


def _burst_count(key_events: List[dict], gap_threshold: float = 2.0) -> int:
    """Number of typing bursts (sequences separated by gaps > gap_threshold s)."""
    press_ts = [e["ts"] for e in key_events if e["type"] == "key_press"]
    if len(press_ts) < 2:
        return 0
    bursts = 1
    for t1, t2 in zip(press_ts, press_ts[1:]):
        if t2 - t1 > gap_threshold:
            bursts += 1
    return bursts


def _pause_count_long(key_events: List[dict], pause_threshold: float = 5.0) -> int:
    """Number of pauses > pause_threshold seconds between key events."""
    press_ts = [e["ts"] for e in key_events if e["type"] == "key_press"]
    if len(press_ts) < 2:
        return 0
    return sum(1 for t1, t2 in zip(press_ts, press_ts[1:]) if t2 - t1 > pause_threshold)


def _rhythm_cv(flight_times: List[float]) -> float:
    """Coefficient of variation of inter-key timing — high = erratic."""
    if len(flight_times) < 3:
        return 0.0
    arr = np.array(flight_times)
    mean = arr.mean()
    if mean < 1e-9:
        return 0.0
    return float(arr.std() / mean)


def _mouse_speeds(move_events: List[dict]) -> List[float]:
    return [e["speed"] for e in move_events if e["type"] == "mouse_move" and "speed" in e]


def _mouse_accelerations(move_events: List[dict]) -> List[float]:
    speeds = [e for e in move_events if e["type"] == "mouse_move" and "speed" in e]
    if len(speeds) < 2:
        return []
    accs = []
    for a, b in zip(speeds, speeds[1:]):
        dt = b["ts"] - a["ts"]
        if dt > 0.001:
            accs.append(abs(b["speed"] - a["speed"]) / dt)
    return accs


def _click_rate(click_events: List[dict], window_sec: float) -> float:
    clicks = sum(1 for e in click_events if e["type"] == "mouse_click" and e["pressed"])
    return clicks / (window_sec / 60.0) if window_sec > 0 else 0.0


def _double_click_rate(click_events: List[dict], window_sec: float,
                       dc_threshold: float = 0.3) -> float:
    press_ts = [e["ts"] for e in click_events
                if e["type"] == "mouse_click" and e["pressed"]]
    if len(press_ts) < 2:
        return 0.0
    dc = sum(1 for t1, t2 in zip(press_ts, press_ts[1:]) if t2 - t1 < dc_threshold)
    minutes = window_sec / 60.0
    return dc / minutes if minutes > 0 else 0.0


def _scroll_velocity(scroll_events: List[dict]) -> List[float]:
    return [abs(e["magnitude"]) for e in scroll_events
            if e["type"] == "mouse_scroll" and "magnitude" in e]


def _mouse_idle_ratio(move_events: List[dict], window_sec: float,
                      idle_threshold: float = 1.0) -> float:
    """Fraction of window with no mouse activity."""
    if not move_events or window_sec <= 0:
        return 1.0
    active_intervals = 0.0
    for a, b in zip(move_events, move_events[1:]):
        gap = b["ts"] - a["ts"]
        if gap <= idle_threshold:
            active_intervals += gap
    return max(0.0, 1.0 - active_intervals / window_sec)


def _path_efficiency(move_events: List[dict]) -> float:
    """
    Ratio of straight-line distance to actual path length.
    1.0 = perfectly straight, lower = curved/erratic.
    """
    moves = [e for e in move_events if e["type"] == "mouse_move" and "x" in e]
    if len(moves) < 2:
        return 1.0
    total_path = sum(m["dist"] for m in moves if "dist" in m)
    if total_path < 1.0:
        return 1.0
    start = (moves[0]["x"], moves[0]["y"])
    end   = (moves[-1]["x"], moves[-1]["y"])
    straight = math.hypot(end[0] - start[0], end[1] - start[1])
    return min(1.0, straight / total_path)


# ─── Main Extraction Entry-Point ──────────────────────────────────────────────

def extract_features(start_ts: float, end_ts: float) -> Optional[np.ndarray]:
    """
    Extract all 18 features for events in [start_ts, end_ts].

    Returns:
        np.ndarray of shape (N_FEATURES,) or None if insufficient data.
    """
    t0 = time.perf_counter()
    events = load_events_in_window(start_ts, end_ts)
    window_sec = end_ts - start_ts

    if len(events) < 10:
        logger.debug("Insufficient events (%d) for feature extraction", len(events))
        return None

    key_events    = [e for e in events if e["type"].startswith("key_")]
    move_events   = [e for e in events if e["type"] == "mouse_move"]
    click_events  = [e for e in events if e["type"] == "mouse_click"]
    scroll_events = [e for e in events if e["type"] == "mouse_scroll"]

    # Derived sequences
    dwells      = _dwell_times(key_events)
    flights     = _flight_times(key_events)
    speeds      = _mouse_speeds(move_events)
    accs        = _mouse_accelerations(move_events)
    scroll_vels = _scroll_velocity(scroll_events)

    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst):  return float(np.std(lst))  if len(lst) > 1 else 0.0

    features = np.array([
        _typing_speed_wpm(key_events, window_sec),          # 0
        safe_mean(dwells),                                   # 1
        safe_std(dwells),                                    # 2
        safe_mean(flights),                                  # 3
        safe_std(flights),                                   # 4
        _error_rate(key_events),                             # 5
        _error_rate(key_events),                             # 6  backspace_ratio (alias)
        float(_burst_count(key_events)),                     # 7
        float(_pause_count_long(key_events)),                # 8
        _rhythm_cv(flights),                                 # 9
        safe_mean(speeds),                                   # 10
        safe_std(speeds),                                    # 11
        safe_mean(accs),                                     # 12
        _click_rate(click_events, window_sec),               # 13
        _double_click_rate(click_events, window_sec),        # 14
        safe_mean(scroll_vels),                              # 15
        _mouse_idle_ratio(move_events, window_sec),          # 16
        _path_efficiency(move_events),                       # 17
    ], dtype=np.float32)

    # Clamp NaN / Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = time.perf_counter() - t0
    logger.debug("Feature extraction: %.1f ms for %d events", elapsed * 1000, len(events))

    if elapsed > 1.0:
        logger.warning("Feature extraction exceeded 1 s: %.2f s", elapsed)

    return features


# ─── Rolling Normalizer ───────────────────────────────────────────────────────

class RollingNormalizer:
    """
    Adaptive Z-score normalization using a 7-day rolling window of feature vectors.
    Persists state to disk so it survives restarts.
    """

    def __init__(self, window_days: int = ROLLING_NORM_DAYS):
        self._window_sec = window_days * 86400
        self._history: deque = deque()   # list of (ts, np.ndarray)
        self._lock = threading.Lock()

    def update(self, ts: float, features: np.ndarray) -> None:
        cutoff = ts - self._window_sec
        with self._lock:
            self._history.append((ts, features.copy()))
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()

    def transform(self, features: np.ndarray) -> np.ndarray:
        with self._lock:
            if len(self._history) < 5:
                return features.copy()          # not enough baseline yet
            arr = np.stack([v for _, v in self._history])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        std  = np.where(std < 1e-9, 1.0, std)
        return ((features - mean) / std).astype(np.float32)

    def save(self, path: Path = SCALER_PATH) -> None:
        with self._lock:
            state = list(self._history)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("Scaler state saved → %s", path)

    def load(self, path: Path = SCALER_PATH) -> None:
        if not path.exists():
            logger.info("No scaler state found — starting fresh")
            return
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        with self._lock:
            self._history = deque(state)
        logger.info("Scaler state loaded (%d samples)", len(self._history))

    @property
    def n_samples(self) -> int:
        with self._lock:
            return len(self._history)


# ─── Sequence Builder ─────────────────────────────────────────────────────────

class SequenceBuilder:
    """
    Maintains a ring of normalized feature vectors and assembles
    fixed-length sequences for LSTM inference.
    """

    def __init__(self, sequence_length: int = 30):
        self._seq_len = sequence_length
        self._window: deque = deque(maxlen=sequence_length)
        self._lock = threading.Lock()

    def push(self, features: np.ndarray) -> None:
        with self._lock:
            self._window.append(features)

    def get_sequence(self) -> Optional[np.ndarray]:
        """Return (sequence_length, N_FEATURES) array or None if not ready."""
        with self._lock:
            if len(self._window) < self._seq_len:
                return None
            return np.stack(list(self._window))  # shape: (seq_len, N_FEATURES)

    @property
    def fill_ratio(self) -> float:
        with self._lock:
            return len(self._window) / self._seq_len


# ─── Feature Writer ───────────────────────────────────────────────────────────

def save_feature_vector(ts: float, raw: np.ndarray, norm: np.ndarray) -> None:
    """Append a feature record to the daily feature JSONL file."""
    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    fpath = FEATURES_DIR / f"features_{date_str}.jsonl"
    record = {
        "ts": ts,
        "raw": raw.tolist(),
        "norm": norm.tolist(),
        "names": FEATURE_NAMES,
    }
    try:
        with open(fpath, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.error("Feature write error: %s", exc)


def load_feature_history(days: int = ROLLING_NORM_DAYS) -> List[Tuple[float, np.ndarray]]:
    """Load raw feature vectors from the past N days."""
    result = []
    now = time.time()
    cutoff = now - days * 86400
    for fpath in sorted(FEATURES_DIR.glob("features_*.jsonl")):
        try:
            with open(fpath) as fh:
                for line in fh:
                    rec = json.loads(line)
                    ts = rec.get("ts", 0)
                    if ts >= cutoff:
                        result.append((ts, np.array(rec["raw"], dtype=np.float32)))
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    result.sort(key=lambda x: x[0])
    return result
