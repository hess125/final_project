"""
led.py — RGB LED controller with smooth cubic-ease color transitions.

Uses pigpio for hardware PWM to avoid software PWM flicker.
Falls back to mock mode when GPIO is unavailable (development/testing).
"""

import logging
import math
import threading
import time
from typing import Optional, Tuple

'''
from config import (
    GPIO_BLUE,
    GPIO_GREEN,
    GPIO_RED,
    LED_COLORS,
    LED_TRANSITION_SEC,
    PWM_FREQ,
)
'''
logger = logging.getLogger(__name__)

# ─── GPIO Backend ─────────────────────────────────────────────────────────────

class _MockGPIO:
    """Silent mock when running outside Raspberry Pi."""
    def __init__(self):
        logger.info("LED: using mock GPIO backend")
    def set_PWM_dutycycle(self, pin, dc): pass
    def set_PWM_frequency(self, pin, freq): pass
    def set_mode(self, pin, mode): pass
    def stop(self): pass


def _init_gpio() -> Tuple[object, bool]:
    """
    Attempt to initialize pigpio. Returns (pi, is_real).
    """
    try:
        import pigpio
        pi = pigpio.pi()
        if not pi.connected:
            raise RuntimeError("pigpio daemon not running — start with: sudo pigpiod")
        for pin in (GPIO_RED, GPIO_GREEN, GPIO_BLUE):
            pi.set_mode(pin, pigpio.OUTPUT)
            pi.set_PWM_frequency(pin, PWM_FREQ)
        logger.info("LED: pigpio connected (R=%d G=%d B=%d)", GPIO_RED, GPIO_GREEN, GPIO_BLUE)
        return pi, True
    except Exception as exc:
        logger.warning("LED: GPIO unavailable (%s) — mock mode", exc)
        return _MockGPIO(), False


# ─── Color Utilities ──────────────────────────────────────────────────────────

def _cubic_ease(t: float) -> float:
    """Cubic ease-in-out: smooth S-curve, t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _dc(percent: float) -> int:
    """Convert 0-100% to 0-255 pigpio duty cycle."""
    return int(max(0, min(100, percent)) * 255 / 100)


# ─── LED Controller ───────────────────────────────────────────────────────────

class LEDController:
    """
    Manages RGB LED with:
    - Smooth 10-second cubic-ease transitions between stress levels
    - Thread-safe non-blocking operation
    - Graceful degradation to mock mode on non-Pi hardware
    """

    _UPDATE_INTERVAL = 0.05   # 20 Hz transition update rate

    def __init__(self):
        self._pi, self._real = _init_gpio()
        self._current_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._target_rgb:  Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._start_rgb:   Tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._transition_start:    float = 0.0
        self._transition_duration: float = LED_TRANSITION_SEC
        self._in_transition:       bool  = False

        self._current_level: str  = "offline"
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._run, daemon=True, name="led-ctrl")
        self._stop   = threading.Event()
        self._thread.start()

        # Start in baseline (blue) mode
        self.set_level("baseline")
        logger.info("LEDController started")

    def set_level(self, level: str, duration: float = LED_TRANSITION_SEC) -> None:
        """
        Smoothly transition to the color associated with *level*.
        Thread-safe; can be called from any thread.
        """
        color = LED_COLORS.get(level, LED_COLORS["offline"])
        with self._lock:
            self._start_rgb          = self._current_rgb
            self._target_rgb         = color
            self._transition_start   = time.time()
            self._transition_duration = max(0.1, duration)
            self._in_transition      = True
            self._current_level      = level
        logger.debug("LED: %s → %s", level, color)

    def set_rgb_direct(self, r: float, g: float, b: float) -> None:
        """Immediate (no transition) raw RGB set, values 0-100."""
        with self._lock:
            self._in_transition = False
            self._current_rgb   = (r, g, b)
        self._write(r, g, b)

    def off(self) -> None:
        self.set_level("offline", duration=2.0)

    def _write(self, r: float, g: float, b: float) -> None:
        try:
            self._pi.set_PWM_dutycycle(GPIO_RED,   _dc(r))
            self._pi.set_PWM_dutycycle(GPIO_GREEN, _dc(g))
            self._pi.set_PWM_dutycycle(GPIO_BLUE,  _dc(b))
        except Exception as exc:
            logger.warning("LED write error: %s", exc)

    def _run(self) -> None:
        while not self._stop.wait(self._UPDATE_INTERVAL):
            with self._lock:
                if not self._in_transition:
                    r, g, b = self._current_rgb
                else:
                    elapsed = time.time() - self._transition_start
                    progress = elapsed / self._transition_duration
                    if progress >= 1.0:
                        self._in_transition = False
                        self._current_rgb   = self._target_rgb
                        r, g, b = self._current_rgb
                    else:
                        ease = _cubic_ease(progress)
                        sr, sg, sb = self._start_rgb
                        tr, tg, tb = self._target_rgb
                        r = sr + (tr - sr) * ease
                        g = sg + (tg - sg) * ease
                        b = sb + (tb - sb) * ease
                        self._current_rgb = (r, g, b)

            self._write(r, g, b)

    def stop(self) -> None:
        self.off()
        time.sleep(2.5)
        self._stop.set()
        self._thread.join(timeout=5)
        if self._real:
            try:
                self._pi.stop()
            except Exception:
                pass
        logger.info("LEDController stopped")

    @property
    def current_level(self) -> str:
        with self._lock:
            return self._current_level

    @property
    def current_rgb(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._current_rgb


# ─── Breathing Animation ─────────────────────────────────────────────────────

class BreathingAnimation:
    """
    Optional calming breathing-guide animation overlaid on the base color.
    4-7-8 breathing pattern (inhale 4s, hold 7s, exhale 8s).
    """

    def __init__(self, controller: LEDController, base_level: str = "low"):
        self._ctrl  = controller
        self._base  = base_level
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="breathing")
        self._thread.start()

    def _run(self) -> None:
        base_rgb = LED_COLORS.get(self._base, (0, 80, 40))
        while not self._stop.is_set():
            # Inhale: 4 s (brighten)
            self._phase(base_rgb, 1.0, 4.0)
            if self._stop.is_set():
                break
            # Hold: 7 s (steady bright)
            self._phase(base_rgb, 1.0, 7.0)
            if self._stop.is_set():
                break
            # Exhale: 8 s (dim)
            self._phase(base_rgb, 0.3, 8.0)

    def _phase(self, base_rgb, target_scale, duration):
        start_rgb   = self._ctrl.current_rgb
        target_rgb  = tuple(c * target_scale for c in base_rgb)
        t0 = time.time()
        while not self._stop.is_set():
            elapsed  = time.time() - t0
            progress = min(1.0, elapsed / duration)
            ease     = _cubic_ease(progress)
            r = start_rgb[0] + (target_rgb[0] - start_rgb[0]) * ease
            g = start_rgb[1] + (target_rgb[1] - start_rgb[1]) * ease
            b = start_rgb[2] + (target_rgb[2] - start_rgb[2]) * ease
            self._ctrl.set_rgb_direct(r, g, b)
            if progress >= 1.0:
                break
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=20)
