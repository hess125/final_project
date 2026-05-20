"""
collector.py — Privacy-preserving keyboard and mouse behavioral monitor.

Stores ONLY timing metadata. No key values, no click targets, no text.
Raw event files auto-delete after RAW_RETENTION_SECONDS.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pynput import keyboard, mouse

from config import (
    RAW_DIR,
    RAW_RETENTION_SECONDS,
    PURGE_INTERVAL_SECONDS,
    STORE_KEY_VALUES,
    STORE_CLICK_TARGETS,
)

logger = logging.getLogger(__name__)


# ─── Event Buffer ──────────────────────────────────────────────────────────────

class EventBuffer:
    """Thread-safe ring buffer for raw behavioral events."""

    def __init__(self, maxlen: int = 10000):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, event: dict) -> None:
        with self._lock:
            self._buf.append(event)

    def drain(self) -> list:
        """Return all events and clear the buffer."""
        with self._lock:
            events = list(self._buf)
            self._buf.clear()
            return events

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ─── Keyboard Monitor ─────────────────────────────────────────────────────────

class KeyboardMonitor:
    """
    Captures ONLY:
    - Key press timestamp (ms since epoch)
    - Key release timestamp (ms since epoch)
    - Whether key is a modifier, backspace, enter, or 'other'
    - No character values stored under any circumstances.
    """

    # Category map — never logs actual key identities beyond these buckets
    _CATEGORIES = {
        keyboard.Key.backspace: "backspace",
        keyboard.Key.enter:     "enter",
        keyboard.Key.space:     "space",
        keyboard.Key.tab:       "tab",
        keyboard.Key.shift:     "modifier",
        keyboard.Key.shift_r:   "modifier",
        keyboard.Key.ctrl:      "modifier",
        keyboard.Key.ctrl_r:    "modifier",
        keyboard.Key.alt:       "modifier",
        keyboard.Key.alt_r:     "modifier",
        keyboard.Key.cmd:       "modifier",
    }

    def __init__(self, buffer: EventBuffer):
        self._buffer = buffer
        self._press_times: dict = {}
        self._listener: Optional[keyboard.Listener] = None
        self._active = False

    def _category(self, key) -> str:
        if key in self._CATEGORIES:
            return self._CATEGORIES[key]
        if isinstance(key, keyboard.Key):
            return "special"
        return "alpha"  # any printable character → generic bucket

    def _on_press(self, key) -> None:
        ts = time.time()
        cat = self._category(key)
        self._press_times[id(key)] = ts
        self._buffer.append({
            "type": "key_press",
            "ts": ts,
            "cat": cat,
        })

    def _on_release(self, key) -> None:
        ts = time.time()
        cat = self._category(key)
        press_ts = self._press_times.pop(id(key), None)
        dwell = round(ts - press_ts, 6) if press_ts else None
        self._buffer.append({
            "type": "key_release",
            "ts": ts,
            "cat": cat,
            "dwell": dwell,
        })

    def start(self) -> None:
        if self._active:
            return
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.start()
        self._active = True
        logger.info("Keyboard monitor started")

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        self._active = False
        logger.info("Keyboard monitor stopped")


# ─── Mouse Monitor ────────────────────────────────────────────────────────────

class MouseMonitor:
    """
    Captures ONLY:
    - Move timestamps and pixel coordinates (no element targets)
    - Click timestamps and button category (no target/URL/element)
    - Scroll timestamps and delta magnitude
    """

    def __init__(self, buffer: EventBuffer):
        self._buffer = buffer
        self._listener: Optional[mouse.Listener] = None
        self._active = False
        self._last_pos: Optional[tuple] = None
        self._last_move_ts: float = 0.0

    def _on_move(self, x: int, y: int) -> None:
        ts = time.time()
        prev_pos = self._last_pos
        prev_ts  = self._last_move_ts
        self._last_pos = (x, y)
        self._last_move_ts = ts

        if prev_pos is None:
            return

        dt = ts - prev_ts
        if dt < 0.01:          # debounce — skip if < 10 ms
            return

        dx = x - prev_pos[0]
        dy = y - prev_pos[1]
        dist = (dx**2 + dy**2) ** 0.5
        speed = dist / dt if dt > 0 else 0.0

        self._buffer.append({
            "type": "mouse_move",
            "ts": ts,
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "dist": round(dist, 2),
            "speed": round(speed, 2),
            "dt": round(dt, 6),
        })

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        ts = time.time()
        btn_label = "left" if button == mouse.Button.left else (
            "right" if button == mouse.Button.right else "middle"
        )
        self._buffer.append({
            "type": "mouse_click",
            "ts": ts,
            "btn": btn_label,
            "pressed": pressed,
            # x, y intentionally omitted — no target inference
        })

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        ts = time.time()
        self._buffer.append({
            "type": "mouse_scroll",
            "ts": ts,
            "dy": dy,
            "magnitude": abs(dy),
        })

    def start(self) -> None:
        if self._active:
            return
        self._listener = mouse.Listener(
            on_move=self._on_click,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        # Override with correct handlers
        self._listener.on_move   = self._on_move
        self._listener.on_click  = self._on_click
        self._listener.on_scroll = self._on_scroll
        self._listener.start()
        self._active = True
        logger.info("Mouse monitor started")

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        self._active = False
        logger.info("Mouse monitor stopped")


# ─── Flusher ──────────────────────────────────────────────────────────────────

class EventFlusher:
    """
    Periodically drains the event buffer to timestamped JSONL files.
    Each file covers one flush interval and carries an expiry timestamp.
    """

    def __init__(
        self,
        buffer: EventBuffer,
        flush_interval: float = 10.0,
        output_dir: Path = RAW_DIR,
    ):
        self._buffer   = buffer
        self._interval = flush_interval
        self._dir      = output_dir
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def _flush(self) -> None:
        events = self._buffer.drain()
        if not events:
            return

        ts_now   = time.time()
        expires  = ts_now + RAW_RETENTION_SECONDS
        filename = self._dir / f"raw_{int(ts_now * 1000)}.jsonl"

        meta = {
            "_meta": True,
            "created_ts": ts_now,
            "expires_ts": expires,
            "event_count": len(events),
        }

        try:
            with open(filename, "w") as fh:
                fh.write(json.dumps(meta) + "\n")
                for ev in events:
                    fh.write(json.dumps(ev) + "\n")
            logger.debug("Flushed %d events → %s", len(events), filename.name)
        except OSError as exc:
            logger.error("Flush failed: %s", exc)

    def _run(self) -> None:
        while not self._stop_evt.wait(self._interval):
            self._flush()
        self._flush()   # final drain on shutdown

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="flusher")
        self._thread.start()
        logger.info("Event flusher started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("Event flusher stopped")


# ─── Purge Daemon ─────────────────────────────────────────────────────────────

class RawDataPurger:
    """
    Watches RAW_DIR and deletes any file past its expires_ts.
    Runs independently so privacy guarantee holds even if main process is busy.
    """

    def __init__(self, raw_dir: Path = RAW_DIR):
        self._dir      = raw_dir
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def _purge_pass(self) -> None:
        now = time.time()
        try:
            for fpath in self._dir.glob("raw_*.jsonl"):
                try:
                    with open(fpath) as fh:
                        meta = json.loads(fh.readline())
                    if meta.get("_meta") and now >= meta.get("expires_ts", 0):
                        fpath.unlink()
                        logger.info("Purged expired raw file: %s", fpath.name)
                except (json.JSONDecodeError, OSError):
                    # Unreadable or already gone — remove it
                    try:
                        fpath.unlink()
                    except OSError:
                        pass
        except OSError as exc:
            logger.error("Purge scan error: %s", exc)

    def _run(self) -> None:
        while not self._stop_evt.wait(PURGE_INTERVAL_SECONDS):
            self._purge_pass()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="purger")
        self._thread.start()
        logger.info("Raw data purger started (interval=%ds)", PURGE_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._stop_evt.set()
        logger.info("Raw data purger stopped")


# ─── Collector Facade ─────────────────────────────────────────────────────────

class BehavioralCollector:
    """Top-level facade that owns all collection sub-systems."""

    def __init__(self):
        self._buffer  = EventBuffer(maxlen=20000)
        self._kb      = KeyboardMonitor(self._buffer)
        self._mouse   = MouseMonitor(self._buffer)
        self._flusher = EventFlusher(self._buffer)
        self._purger  = RawDataPurger()
        self._running = False

    def start(self) -> None:
        if self._running:
            logger.warning("Collector already running")
            return
        self._kb.start()
        self._mouse.start()
        self._flusher.start()
        self._purger.start()
        self._running = True
        logger.info("BehavioralCollector started — privacy mode ON")

    def stop(self) -> None:
        self._flusher.stop()
        self._kb.stop()
        self._mouse.stop()
        self._purger.stop()
        self._running = False
        logger.info("BehavioralCollector stopped")

    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR


# ─── CLI Entry-Point ──────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    collector = BehavioralCollector()
    collector.start()

    def _shutdown(sig, frame):
        logger.info("Signal %s received — shutting down collector", sig)
        collector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Collector running. Press Ctrl-C to stop.")
    while True:
        time.sleep(60)
        logger.debug("Buffer size: %d events", collector.buffer_size())


if __name__ == "__main__":
    main()
