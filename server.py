"""
server.py — Flask web server providing:
  - REST API for real-time stress data
  - Self-report data collection (SQLite)
  - WebSocket streaming for live dashboard
  - Three.js 3D visualization page
  - SUS questionnaire endpoint
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template_string, request, Response

from config import (
    FLASK_HOST,
    FLASK_PORT,
    LOGS_DIR,
    POLL_INTERVAL,
    REPORT_SCALE,
    SECRET_KEY,
    SELF_REPORT_DB,
    STATIC_DIR,
    TEMPLATES_DIR,
)

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATES_DIR))
app.secret_key = SECRET_KEY

# ─── Shared State (written by orchestrator, read by server) ──────────────────

_state: dict = {
    "stress_score":  0.0,
    "stress_level":  "baseline",
    "is_anomaly":    False,
    "latency_ms":    0.0,
    "ts":            time.time(),
    "uptime_s":      0,
    "features":      {},
    "calibrated":    False,
}

def update_state(new_state: dict) -> None:
    """Called by orchestrator thread to push new inference results."""
    global _state
    _state = {**_state, **new_state, "ts": time.time()}


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS self_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                score       INTEGER NOT NULL,
                notes       TEXT,
                context     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sus_responses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                participant TEXT,
                q1  INTEGER, q2  INTEGER, q3  INTEGER, q4  INTEGER, q5  INTEGER,
                q6  INTEGER, q7  INTEGER, q8  INTEGER, q9  INTEGER, q10 INTEGER,
                sus_score   REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS latency_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                component   TEXT,
                latency_ms  REAL
            )
        """)
        conn.commit()


@contextmanager
def _db():
    conn = sqlite3.connect(str(SELF_REPORT_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """
    GET /api/status
    Returns current stress score, level, and system metadata.
    """
    return jsonify({
        "stress_score":  round(_state.get("stress_score", 0.0), 4),
        "stress_level":  _state.get("stress_level", "baseline"),
        "is_anomaly":    _state.get("is_anomaly", False),
        "calibrated":    _state.get("calibrated", False),
        "latency_ms":    _state.get("latency_ms", 0),
        "uptime_s":      _state.get("uptime_s", 0),
        "ts":            _state.get("ts", time.time()),
        "poll_interval": POLL_INTERVAL,
    })


@app.route("/api/features")
def api_features():
    """
    GET /api/features
    Returns the latest normalized feature vector.
    """
    return jsonify({
        "features": _state.get("features", {}),
        "ts":       _state.get("ts", time.time()),
    })


@app.route("/api/history")
def api_history():
    """
    GET /api/history?hours=1
    Returns self-report and stress score history for correlation display.
    """
    hours = float(request.args.get("hours", 1))
    cutoff = time.time() - hours * 3600
    with _db() as conn:
        rows = conn.execute(
            "SELECT ts, score, notes FROM self_reports WHERE ts >= ? ORDER BY ts",
            (cutoff,)
        ).fetchall()
    return jsonify({"reports": [dict(r) for r in rows]})


@app.route("/api/report", methods=["POST"])
def api_report():
    """
    POST /api/report
    Body: {"score": 1-10, "notes": "optional text", "context": "study|break|other"}
    Record a self-reported stress measurement.
    """
    data = request.get_json(force=True, silent=True) or {}
    score = data.get("score")
    if not isinstance(score, int) or not (REPORT_SCALE[0] <= score <= REPORT_SCALE[1]):
        return jsonify({"error": f"score must be integer {REPORT_SCALE[0]}-{REPORT_SCALE[1]}"}), 400

    notes   = str(data.get("notes", ""))[:500]
    context = str(data.get("context", ""))[:50]
    ts      = time.time()

    with _db() as conn:
        conn.execute(
            "INSERT INTO self_reports (ts, score, notes, context) VALUES (?,?,?,?)",
            (ts, score, notes, context)
        )
        conn.commit()

    logger.info("Self-report recorded: score=%d context=%s", score, context)
    return jsonify({"status": "ok", "ts": ts})


@app.route("/api/sus", methods=["POST"])
def api_sus():
    """
    POST /api/sus
    Body: {"participant": "P1", "q1": 1..5, ..., "q10": 1..5}
    Compute and store SUS score.
    SUS = ((odd_sum - 5) + (25 - even_sum)) * 2.5
    """
    data    = request.get_json(force=True, silent=True) or {}
    answers = {}
    for i in range(1, 11):
        key = f"q{i}"
        val = data.get(key)
        if not isinstance(val, int) or not (1 <= val <= 5):
            return jsonify({"error": f"{key} must be integer 1-5"}), 400
        answers[key] = val

    odd_sum  = sum(answers[f"q{i}"] for i in [1,3,5,7,9])
    even_sum = sum(answers[f"q{i}"] for i in [2,4,6,8,10])
    sus_score = ((odd_sum - 5) + (25 - even_sum)) * 2.5

    participant = str(data.get("participant", "anonymous"))[:50]
    ts = time.time()

    with _db() as conn:
        conn.execute("""
            INSERT INTO sus_responses
            (ts, participant, q1,q2,q3,q4,q5,q6,q7,q8,q9,q10, sus_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, participant,
              answers["q1"], answers["q2"], answers["q3"], answers["q4"], answers["q5"],
              answers["q6"], answers["q7"], answers["q8"], answers["q9"], answers["q10"],
              sus_score))
        conn.commit()

    logger.info("SUS recorded: participant=%s score=%.1f", participant, sus_score)
    return jsonify({"status": "ok", "sus_score": round(sus_score, 1), "ts": ts})


@app.route("/api/latency", methods=["POST"])
def api_latency():
    """POST /api/latency — log a latency measurement."""
    data = request.get_json(force=True, silent=True) or {}
    component  = str(data.get("component", "unknown"))[:50]
    latency_ms = float(data.get("latency_ms", 0))
    ts = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO latency_log (ts, component, latency_ms) VALUES (?,?,?)",
            (ts, component, latency_ms)
        )
        conn.commit()
    return jsonify({"status": "ok"})


@app.route("/api/analytics")
def api_analytics():
    """
    GET /api/analytics?days=7
    Returns correlation, mean SUS, uptime stats for researcher view.
    """
    days   = float(request.args.get("days", 7))
    cutoff = time.time() - days * 86400
    with _db() as conn:
        sus_rows = conn.execute(
            "SELECT sus_score FROM sus_responses WHERE ts >= ?", (cutoff,)
        ).fetchall()
        lat_rows = conn.execute(
            "SELECT component, AVG(latency_ms) as avg_ms FROM latency_log "
            "WHERE ts >= ? GROUP BY component", (cutoff,)
        ).fetchall()
    mean_sus = (
        sum(r["sus_score"] for r in sus_rows) / len(sus_rows)
        if sus_rows else None
    )
    latencies = {r["component"]: round(r["avg_ms"], 1) for r in lat_rows}
    return jsonify({
        "mean_sus_score": round(mean_sus, 1) if mean_sus else None,
        "n_sus_responses": len(sus_rows),
        "avg_latencies_ms": latencies,
    })


# ─── SSE Stream ───────────────────────────────────────────────────────────────

@app.route("/api/stream")
def api_stream():
    """
    GET /api/stream
    Server-Sent Events for real-time dashboard updates.
    """
    def _generator():
        last_ts = 0.0
        while True:
            current_ts = _state.get("ts", 0)
            if current_ts != last_ts:
                payload = json.dumps({
                    "stress_score": round(_state.get("stress_score", 0.0), 4),
                    "stress_level": _state.get("stress_level", "baseline"),
                    "is_anomaly":   _state.get("is_anomaly", False),
                    "ts":           current_ts,
                })
                yield f"data: {payload}\n\n"
                last_ts = current_ts
            time.sleep(1.0)

    return Response(_generator(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Dashboard Page ───────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CogHealth Monitor</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #6ee7b7;
    --text: #e2e8f0;
    --muted: #64748b;
    --low: #6ee7b7;
    --moderate: #fbbf24;
    --elevated: #f97316;
    --high: #ef4444;
    --baseline: #60a5fa;
  }
  :root.light-mode {
    --bg: #f8f9fa;
    --surface: #ffffff;
    --border: #e5e7eb;
    --accent: #10b981;
    --text: #1f2937;
    --muted: #6b7280;
    --low: #10b981;
    --moderate: #f59e0b;
    --elevated: #f97316;
    --high: #ef4444;
    --baseline: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
    transition: background 0.3s ease, color 0.3s ease;
  }
  header {
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--surface);
    transition: background 0.3s ease, border-color 0.3s ease;
  }
  .logo { font-size: 1.2rem; letter-spacing: 0.15em; color: var(--accent); transition: color 0.3s ease; }
  .theme-toggle {
    background: none;
    border: 1px solid var(--border);
    color: var(--text);
    width: 36px;
    height: 36px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    transition: background 0.2s, border-color 0.2s, color 0.2s;
    padding: 0;
  }
  .theme-toggle:hover {
    background: var(--border);
    color: var(--accent);
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  main {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 0;
    height: calc(100vh - 57px);
  }
  #canvas-container {
    position: relative;
    background: radial-gradient(ellipse at center, #0d1117 0%, #0a0a0f 100%);
    transition: background 0.3s ease;
  }
  :root.light-mode #canvas-container {
    background: radial-gradient(ellipse at center, #f0f4f8 0%, #f8f9fa 100%);
  }
  canvas { display: block; width: 100%; height: 100%; }
  .sidebar {
    background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0;
    overflow-y: auto;
    transition: background 0.3s ease, border-color 0.3s ease;
  }
  .panel {
    padding: 1.25rem;
    border-bottom: 1px solid var(--border);
  }
  .panel-title {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    margin-bottom: 1rem;
    text-transform: uppercase;
  }
  .score-display {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    transition: color 0.5s ease;
  }
  .level-badge {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border: 1px solid currentColor;
  }
  .feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
  .feat-item { font-size: 0.7rem; }
  .feat-name { color: var(--muted); }
  .feat-val  { color: var(--text); font-weight: 600; }
  .bar-container { width: 100%; height: 4px; background: var(--border); border-radius: 2px; margin-top: 3px; }
  .bar { height: 4px; border-radius: 2px; background: var(--accent); transition: width 0.8s ease; }
  .report-form { display: flex; flex-direction: column; gap: 0.75rem; }
  .slider-row { display: flex; align-items: center; gap: 0.75rem; }
  input[type=range] {
    flex: 1; -webkit-appearance: none;
    height: 4px; background: var(--border); border-radius: 2px; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px;
    border-radius: 50%; background: var(--accent); cursor: pointer;
  }
  .score-label {
    min-width: 2rem; text-align: right;
    font-size: 1.1rem; font-weight: 700; color: var(--accent);
  }
  textarea {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); border-radius: 4px; padding: 0.5rem;
    font-family: inherit; font-size: 0.75rem; resize: vertical; min-height: 60px;
  }
  select {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); border-radius: 4px; padding: 0.4rem 0.5rem;
    font-family: inherit; font-size: 0.75rem;
  }
  button {
    padding: 0.6rem 1.2rem; border: 1px solid var(--accent);
    background: transparent; color: var(--accent); border-radius: 4px;
    cursor: pointer; font-family: inherit; font-size: 0.75rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    transition: background 0.2s, color 0.2s;
  }
  button:hover { background: var(--accent); color: var(--bg); }
  .msg { font-size: 0.7rem; color: var(--accent); min-height: 1.2em; }
  .latency-row { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); }

  /* ── Toast notification ───────────────────────────────────────── */
  #stressToast {
    position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    opacity: 0; pointer-events: none;
    transition: opacity 0.4s ease, transform 0.4s ease;
    transform: translateX(-50%) translateY(20px);
    z-index: 999; min-width: 320px; max-width: 500px;
  }
  #stressToast.visible {
    opacity: 1; pointer-events: auto;
    transform: translateX(-50%) translateY(0);
  }
  .toast-orb {
    width: 44px; height: 44px; border-radius: 50%; flex-shrink: 0;
    animation: toastPulse 1.5s ease-in-out infinite;
  }
  @keyframes toastPulse {
    0%,100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.15); opacity: 0.7; }
  }
  .toast-body { flex: 1; }
  .toast-title { font-size: 0.85rem; font-weight: 600; margin-bottom: 2px; }
  .toast-msg   { font-size: 0.75rem; color: var(--muted); line-height: 1.4; }
  .toast-close {
    background: none; border: none; color: var(--muted);
    font-size: 1.1rem; cursor: pointer; padding: 0 0.25rem; line-height: 1;
  }
  .toast-close:hover { color: var(--text); background: none; }

  /* ── Advice panel ─────────────────────────────────────────────── */
  #advicePanel {
    position: absolute; bottom: 1.5rem; left: 50%;
    transform: translateX(-50%);
    text-align: center; max-width: 420px;
    font-size: 0.75rem; color: var(--muted);
    line-height: 1.6; letter-spacing: 0.03em;
    transition: opacity 0.8s ease;
  }
  .advice-label {
    font-size: 0.6rem; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 4px;
    color: var(--muted);
  }
  .advice-text { color: var(--text); }

  /* ── History chart ────────────────────────────────────────────── */
  .hist-panel { padding: 1.25rem; border-top: 1px solid var(--border); }
  .hist-bars  {
    display: flex; align-items: flex-end;
    gap: 4px; height: 48px; margin-top: 0.75rem;
  }
  .hist-bar {
    flex: 1; border-radius: 3px 3px 0 0; min-height: 3px;
    transition: height 0.5s ease, background 0.5s ease;
    cursor: default; position: relative;
  }
  .hist-bar:hover::after {
    content: attr(data-tip);
    position: absolute; bottom: calc(100% + 6px); left: 50%;
    transform: translateX(-50%);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; padding: 3px 8px;
    font-size: 0.65rem; white-space: nowrap; color: var(--text);
    pointer-events: none; z-index: 10;
  }
</style>
</head>
<body>
<header>
  <div class="status-dot" id="statusDot"></div>
  <div class="logo">COGHEALTH</div>
  <div style="flex:1"></div>
  <button class="theme-toggle" id="themeToggle" title="Toggle light/dark mode">🌙</button>
  <div style="font-size:0.7rem;color:var(--muted)" id="uptimeLabel">—</div>
</header>
<main>
  <div id="canvas-container">
    <canvas id="c"></canvas>

    <!-- ── Advice overlay (inside canvas container) ──────────────────── -->

    <div id="advicePanel">
      <div class="advice-label">System Insight</div>
      <div class="advice-text" id="adviceText">Monitoring your behavioral patterns…</div>
    </div>
  </div>
  <div class="sidebar">
    <!-- Stress Score -->
    <div class="panel">
      <div class="panel-title">Stress Indicator</div>
      <div class="score-display" id="scoreDisplay">0.00</div>
      <div><span class="level-badge" id="levelBadge">baseline</span></div>
    </div>

    <!-- Feature Overview -->
    
    <div class="panel">
      <div class="panel-title">Behavioral Features</div>
      <div class="feat-grid" id="featGrid"></div>
    </div>

    <!-- Self-Report -->
    
    <div class="panel">
      <div class="panel-title">Self-Report</div>
      <div class="report-form">
        <div class="slider-row">
          <input type="range" id="stressSlider" min="1" max="10" value="5">
          <div class="score-label" id="sliderLabel">5</div>
        </div>
        <div style="font-size:0.65rem;color:var(--muted)">1 = Very calm — 10 = Extremely stressed</div>
        <select id="contextSel">
          <option value="study">Studying</option>
          <option value="assignment">Assignment deadline</option>
          <option value="exam">Exam week</option>
          <option value="break">Break</option>
          <option value="other">Other</option>
        </select>
        <textarea id="notesInput" placeholder="Optional notes…"></textarea>
        <button onclick="submitReport()">Submit Report</button>
        <div class="msg" id="reportMsg"></div>
      </div>
    </div>

    <!-- System Stats -->
    
    <div class="panel">
      <div class="panel-title">System</div>
      <div class="latency-row"><span>Inference latency</span><span id="latLabel">—</span></div>
      <div class="latency-row"><span>Uptime</span><span id="uptLabel">—</span></div>
      <div class="latency-row"><span>Calibrated</span><span id="calLabel">—</span></div>
    </div>

    <!-- ── Inference History ──────────────────────────────────────── -->
    <div class="hist-panel">
      <div class="panel-title">Last 10 Readings</div>
      <div class="hist-bars" id="histBars"></div>
    </div>
  </div>
  </div><!-- end sidebar -->
</main>

<!-- ── Stress Toast ─────────────────────────────────────────────────── -->

<div id="stressToast">
  <div class="toast-orb" id="toastOrb"></div>
  <div class="toast-body">
    <div class="toast-title" id="toastTitle">Stress Alert</div>
    <div class="toast-msg"  id="toastMsg"></div>
  </div>
  <button class="toast-close" onclick="dismissToast()">✕</button>
</div>
</main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── Theme Toggle ────────────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('coghealth-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const isDark = saved ? saved === 'dark' : prefersDark;
  
  if (!isDark) {
    document.documentElement.classList.add('light-mode');
    updateThemeButton();
  }
}

function toggleTheme() {
  const html = document.documentElement;
  const isLight = html.classList.contains('light-mode');
  
  if (isLight) {
    html.classList.remove('light-mode');
    localStorage.setItem('coghealth-theme', 'dark');
  } else {
    html.classList.add('light-mode');
    localStorage.setItem('coghealth-theme', 'light');
  }
  updateThemeButton();
}

function updateThemeButton() {
  const btn = document.getElementById('themeToggle');
  const isLight = document.documentElement.classList.contains('light-mode');
  btn.textContent = isLight ? '☀️' : '🌙';
}

document.addEventListener('DOMContentLoaded', function() {
  initTheme();
  const btn = document.getElementById('themeToggle');
  if (btn) btn.addEventListener('click', toggleTheme);
});



// ── Three.js Visualization ───────────────────────────────────────────────────
// Update Three.js colors based on theme
function getThemeColor(colorName) {
  const isLight = document.documentElement.classList.contains('light-mode');
  const colors = {
    accent: isLight ? 0x10b981 : 0x6ee7b7,
    baseline: isLight ? 0x3b82f6 : 0x60a5fa,
    low: isLight ? 0x10b981 : 0x6ee7b7,
  };
  return colors[colorName] || 0x6ee7b7;
}


const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
camera.position.set(0, 0, 4);

// ── Leaf Geometry ────────────────────────────────────────────────────────────
function createLeafGeometry() {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(0.5, 0.5, 1, 1.5, 0, 3);
  shape.bezierCurveTo(-1, 1.5, -0.5, 0.5, 0, 0);
  
  const extrudeSettings = { depth: 0.1, bevelEnabled: true, bevelSegments: 2, steps: 2, bevelSize: 0.1, bevelThickness: 0.1 };
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  geometry.center();
  geometry.scale(0.8, 0.8, 0.8);
  return geometry;
}

const leafGeo = createLeafGeometry();
const ballGeo = new THREE.IcosahedronGeometry(1.2, 4);

const accentColor = getThemeColor('accent');
const mat = new THREE.MeshPhongMaterial({
  color: accentColor,
  wireframe: false,
  shininess: 80,
  transparent: true,
  opacity: 0.85,
});
const mesh = new THREE.Mesh(leafGeo, mat);
scene.add(mesh);

// Wireframe overlay
const wireMat = new THREE.MeshBasicMaterial({ color: accentColor, wireframe: true, transparent: true, opacity: 0.15 });
const wireMesh = new THREE.Mesh(leafGeo, wireMat);
scene.add(wireMesh);

// Particle field
const particleGeo = new THREE.BufferGeometry();
const pCount = 400;
const pPos = new Float32Array(pCount * 3);
for (let i = 0; i < pCount * 3; i++) pPos[i] = (Math.random() - 0.5) * 12;
particleGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
const particles = new THREE.Points(
  particleGeo,
  new THREE.PointsMaterial({ color: accentColor, size: 0.03, transparent: true, opacity: 0.4 })
);
scene.add(particles);

// Lights
scene.add(new THREE.AmbientLight(0x333333));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 5, 5);
scene.add(dirLight);
const pointLight = new THREE.PointLight(accentColor, 1.5, 10);
pointLight.position.set(-3, 2, 3);
scene.add(pointLight);

// Store original vertex positions for ball morphing
const ballPosAttr = ballGeo.attributes.position;
const ballOrigPos = ballPosAttr.array.slice();

let targetScore = 0;
let currentScore = 0;
const LEVEL_COLORS = {
  low:      new THREE.Color(0x6ee7b7),
  moderate: new THREE.Color(0xfbbf24),
  elevated: new THREE.Color(0xf97316),
  high:     new THREE.Color(0xef4444),
  baseline: new THREE.Color(0x60a5fa),
  offline:  new THREE.Color(0x334155),
};
let targetColor = LEVEL_COLORS.baseline.clone();
let currentColor = LEVEL_COLORS.baseline.clone();

function resize() {
  const c = canvas.parentElement;
  renderer.setSize(c.clientWidth, c.clientHeight);
  camera.aspect = c.clientWidth / c.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let t = 0;
function animate() {
  requestAnimationFrame(animate);
  t += 0.008;

  // Interpolate score
  currentScore += (targetScore - currentScore) * 0.05;

  // ── Geometry Transition ─────────────────────────────────────────────────
  // Switch geometry based on stress score (threshold 0.4)
  const transitionThreshold = 0.4;
  if (currentScore < transitionThreshold && mesh.geometry !== leafGeo) {
    mesh.geometry = leafGeo;
    wireMesh.geometry = leafGeo;
  } else if (currentScore >= transitionThreshold && mesh.geometry !== ballGeo) {
    mesh.geometry = ballGeo;
    wireMesh.geometry = ballGeo;
  }

  if (currentScore < transitionThreshold) {
    // Leaf animation: gentle swaying
    const sway = Math.sin(t * 1.5) * 0.15;
    mesh.rotation.z = sway;
    mesh.rotation.x = Math.PI / 6 + Math.cos(t) * 0.1;
    mesh.rotation.y = t * 0.2;
    
    // Pulse scale slightly
    const s = 1 + Math.sin(t * 2) * 0.03;
    mesh.scale.set(s, s, s);
  } else {
    // Ball animation: rapid stretching (current behavior)
    const ballScore = (currentScore - transitionThreshold) / (1 - transitionThreshold);
    const distort = 0.15 + ballScore * 0.55;
    const speed   = 0.5 + ballScore * 2.0;
    const posAttr = ballGeo.attributes.position;
    
    for (let i = 0; i < posAttr.count; i++) {
      const ix = i * 3, iy = ix + 1, iz = ix + 2;
      const ox = ballOrigPos[ix], oy = ballOrigPos[iy], oz = ballOrigPos[iz];
      const len = Math.sqrt(ox*ox + oy*oy + oz*oz);
      const n = Math.sin(ox * 2.1 + t * speed) *
                Math.cos(oy * 1.9 + t * speed * 0.7) *
                Math.sin(oz * 2.3 + t * speed * 1.3);
      const scale = 1 + n * distort;
      posAttr.array[ix] = ox * scale / len;
      posAttr.array[iy] = oy * scale / len;
      posAttr.array[iz] = oz * scale / len;
    }
    posAttr.needsUpdate = true;
    ballGeo.computeVertexNormals();
    
    mesh.rotation.x += 0.003 + ballScore * 0.008;
    mesh.rotation.y += 0.005 + ballScore * 0.012;
    mesh.scale.set(1, 1, 1);
  }

  // Color interpolation
  currentColor.lerp(targetColor, 0.03);
  mat.color.copy(currentColor);
  wireMat.color.copy(currentColor);
  pointLight.color.copy(currentColor);

  wireMesh.rotation.copy(mesh.rotation);
  wireMesh.scale.copy(mesh.scale);

  // Particles drift
  particles.rotation.y += 0.0008;

  renderer.render(scene, camera);
}
animate();

// ── API Polling ───────────────────────────────────────────────────────────────
const LEVEL_CSS = { low:'#6ee7b7', moderate:'#fbbf24', elevated:'#f97316', high:'#ef4444', baseline:'#60a5fa', offline:'#475569' };

const featNames = [
  'typing_speed_wpm','keystroke_dwell_mean','keystroke_flight_mean','error_rate',
  'burst_count','pause_count_long','typing_rhythm_cv','mouse_speed_mean',
  'click_rate','scroll_velocity_mean','mouse_idle_ratio','mouse_path_efficiency'
];

function buildFeatGrid(features) {
  const grid = document.getElementById('featGrid');
  grid.innerHTML = '';
  featNames.forEach(name => {
    const val = features[name];
    if (val === undefined) return;
    const short = name.replace(/_/g,' ').replace('keystroke ','').replace('mouse ','');
    const norm = Math.min(1, Math.abs(val) / 3);   // rough viz normalize
    grid.innerHTML += `
      <div class="feat-item">
        <div class="feat-name">${short}</div>
        <div class="feat-val">${typeof val === 'number' ? val.toFixed(2) : val}</div>
        <div class="bar-container"><div class="bar" style="width:${norm*100}%"></div></div>
      </div>`;
  });
}

function formatUptime(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

// ── Advice messages per level ─────────────────────────────────────────────
const ADVICE = {
  baseline: "System is calibrating — use your keyboard and mouse normally.",
  low:      "Behavioral patterns are stable. You appear calm and focused.",
  moderate: "Slight irregularity detected. Consider a short pause if needed.",
  elevated: "Elevated stress detected. Try a 2-minute breathing break: inhale 4s, hold 4s, exhale 6s.",
  high:     "Strong stress signal. Step away from the screen for 5 minutes. Hydrate. Breathe slowly.",
};

const TOAST_COLORS = {
  low:      "#6ee7b7",
  moderate: "#fbbf24",
  elevated: "#f97316",
  high:     "#ef4444",
  baseline: "#60a5fa",
};

// ── Inference history ─────────────────────────────────────────────────────
const inferenceHistory = [];
let toastTimer = null;
let lastToastLevel = null;

function updateHistoryBars() {
  const container = document.getElementById('histBars');
  if (!container) return;
  container.innerHTML = '';
  // Fill empty slots first
  const slots = 10;
  for (let i = 0; i < slots; i++) {
    const bar = document.createElement('div');
    bar.className = 'hist-bar';
    const entry = inferenceHistory[i];
    if (entry) {
      const pct = Math.max(4, Math.round(entry.score * 100));
      bar.style.height    = pct + '%';
      bar.style.background = LEVEL_CSS[entry.level] || '#475569';
      bar.setAttribute('data-tip',
        entry.level + ' ' + entry.score.toFixed(2) + ' @ ' + entry.time);
    } else {
      bar.style.height    = '4px';
      bar.style.background = 'var(--border)';
    }
    container.appendChild(bar);
  }
}

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(level, score) {
  const color   = TOAST_COLORS[level] || '#6ee7b7';
  const titles  = {
    moderate: "Moderate Stress Detected",
    elevated: "Elevated Stress — Take a Break",
    high:     "High Stress Alert",
  };
  document.getElementById('toastOrb').style.background   = color;
  document.getElementById('toastTitle').style.color       = color;
  document.getElementById('toastTitle').textContent       = titles[level] || "Stress Update";
  document.getElementById('toastMsg').textContent         = ADVICE[level];
  document.getElementById('stressToast').classList.add('visible');

  clearTimeout(toastTimer);
  toastTimer = setTimeout(dismissToast, 8000);
}

function dismissToast() {
  document.getElementById('stressToast').classList.remove('visible');
}

// ── Poll ───────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    const score = d.stress_score || 0;
    const level = d.stress_level || 'baseline';
    targetScore = score;
    targetColor = (LEVEL_COLORS[level] || LEVEL_COLORS.baseline).clone();

    // Score and badge
    document.getElementById('scoreDisplay').textContent = score.toFixed(2);
    document.getElementById('scoreDisplay').style.color = LEVEL_CSS[level] || '#e2e8f0';
    document.getElementById('levelBadge').textContent   = level;
    document.getElementById('levelBadge').style.color   = LEVEL_CSS[level] || '#e2e8f0';

    // System stats
    document.getElementById('latLabel').textContent      = d.latency_ms + ' ms';
    document.getElementById('uptLabel').textContent      = formatUptime(d.uptime_s || 0);
    document.getElementById('uptimeLabel').textContent   = formatUptime(d.uptime_s || 0);
    document.getElementById('calLabel').textContent      = d.calibrated ? '✓ Yes' : '… Calibrating';

    // Advice text
    const advEl = document.getElementById('adviceText');
    if (advEl) advEl.textContent = ADVICE[level] || ADVICE.baseline;

    // Inference history
    if (score > 0 || level !== 'baseline') {
      const now = new Date();
      const timeStr = now.getHours() + ':' +
                      String(now.getMinutes()).padStart(2,'0') + ':' +
                      String(now.getSeconds()).padStart(2,'0');
      inferenceHistory.push({ score, level, time: timeStr });
      if (inferenceHistory.length > 10) inferenceHistory.shift();
      updateHistoryBars();
    }

    // Toast — only show when level rises to moderate/elevated/high and changes
    if (['moderate','elevated','high'].includes(level) && level !== lastToastLevel) {
      showToast(level, score);
    }
    if (level === 'low' || level === 'baseline') {
      dismissToast();
    }
    lastToastLevel = level;

  } catch(e) { console.warn('poll error', e); }

  try {
    const r2 = await fetch('/api/features');
    const d2 = await r2.json();
    if (d2.features) buildFeatGrid(d2.features);
  } catch(e) {}s
}

setInterval(poll, {{ poll_interval }});
poll();


// ── Self-Report ───────────────────────────────────────────────────────────────
document.getElementById('stressSlider').addEventListener('input', function() {
  document.getElementById('sliderLabel').textContent = this.value;
});

async function submitReport() {
  const score   = parseInt(document.getElementById('stressSlider').value);
  const notes   = document.getElementById('notesInput').value;
  const context = document.getElementById('contextSel').value;
  const msg     = document.getElementById('reportMsg');
  try {
    const r = await fetch('/api/report', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({score, notes, context})
    });
    const d = await r.json();
    if (d.status === 'ok') {
      msg.textContent = '✓ Saved';
      document.getElementById('notesInput').value = '';
      setTimeout(() => msg.textContent='', 3000);
    } else { msg.textContent = d.error || 'Error'; }
  } catch(e) { msg.textContent = 'Network error'; }
}
</script>
</body>
</html>"""

SUS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>System Usability Scale — CogHealth</title>
<style>
:root { --bg:#0a0a0f; --surface:#12121a; --border:#1e1e2e; --accent:#6ee7b7; --text:#e2e8f0; --muted:#64748b; }
:root.light-mode { --bg:#f8f9fa; --surface:#ffffff; --border:#e5e7eb; --accent:#10b981; --text:#1f2937; --muted:#6b7280; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); font-family:'JetBrains Mono',monospace; padding:2rem; max-width:700px; margin:0 auto; transition:background 0.3s ease,color 0.3s ease; }
.header-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; }
.theme-toggle { background:none; border:1px solid var(--border); color:var(--text); width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:1rem; transition:background 0.2s,border-color 0.2s,color 0.2s; padding:0; display:flex; align-items:center; justify-content:center; }
.theme-toggle:hover { background:var(--border); color:var(--accent); }
h1 { color:var(--accent); font-size:1.1rem; letter-spacing:0.15em; margin-bottom:0.5rem; }
p.sub { color:var(--muted); font-size:0.75rem; margin-bottom:2rem; }
.q { background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:1rem; margin-bottom:1rem; }
.q-text { font-size:0.85rem; margin-bottom:0.75rem; }
.scale { display:flex; gap:0.5rem; align-items:center; }
.scale label { display:flex; flex-direction:column; align-items:center; gap:0.25rem; cursor:pointer; }
.scale label span { font-size:0.65rem; color:var(--muted); }
input[type=radio] { accent-color:var(--accent); width:18px; height:18px; cursor:pointer; }
input[type=text] { background:var(--bg); border:1px solid var(--border); color:var(--text); border-radius:4px; padding:0.5rem; font-family:inherit; font-size:0.8rem; width:100%; margin-bottom:1.5rem; }
button { padding:0.8rem 2rem; border:1px solid var(--accent); background:transparent; color:var(--accent); border-radius:4px; cursor:pointer; font-family:inherit; font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase; transition:background 0.2s,color 0.2s; }
button:hover { background:var(--accent); color:var(--bg); }
.result { margin-top:1rem; padding:1rem; background:var(--surface); border-radius:6px; display:none; }
</style>
</head>
<body>
<div class="header-row">
  <h1>SYSTEM USABILITY SCALE</h1>
  <button class="theme-toggle" id="themeToggle" title="Toggle light/dark mode">🌙</button>
</div>
<p class="sub">Please rate your experience with the CogHealth monitoring system.</p>
<label style="font-size:0.75rem;color:var(--muted)">Participant ID</label>
<input type="text" id="pid" placeholder="e.g. P1" style="margin-top:0.25rem">
<div id="questions"></div>
<button onclick="submitSUS()">Submit</button>
<div class="result" id="result"></div>
<script>
// Theme Toggle
function initTheme() {
  const saved = localStorage.getItem('coghealth-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const isDark = saved ? saved === 'dark' : prefersDark;
  
  if (!isDark) {
    document.documentElement.classList.add('light-mode');
    updateThemeButton();
  }
}

function toggleTheme() {
  const html = document.documentElement;
  const isLight = html.classList.contains('light-mode');
  
  if (isLight) {
    html.classList.remove('light-mode');
    localStorage.setItem('coghealth-theme', 'dark');
  } else {
    html.classList.add('light-mode');
    localStorage.setItem('coghealth-theme', 'light');
  }
  updateThemeButton();
}

function updateThemeButton() {
  const btn = document.getElementById('themeToggle');
  const isLight = document.documentElement.classList.contains('light-mode');
  btn.textContent = isLight ? '☀️' : '🌙';
}

const questions = [
  "I think that I would like to use this system frequently.",
  "I found the system unnecessarily complex.",
  "I thought the system was easy to use.",
  "I think that I would need the support of a technical person to be able to use this system.",
  "I found the various functions in this system were well integrated.",
  "I thought there was too much inconsistency in this system.",
  "I would imagine that most people would learn to use this system very quickly.",
  "I found the system very cumbersome to use.",
  "I felt very confident using the system.",
  "I needed to learn a lot of things before I could get going with this system."
];
const container = document.getElementById('questions');
questions.forEach((q, i) => {
  const num = i + 1;
  container.innerHTML += `
    <div class="q">
      <div class="q-text">${num}. ${q}</div>
      <div class="scale">
        <span style="font-size:0.65rem;color:var(--muted);min-width:80px">Strongly Disagree</span>
        ${[1,2,3,4,5].map(v => `
          <label>
            <input type="radio" name="q${num}" value="${v}" required>
            <span>${v}</span>
          </label>`).join('')}
        <span style="font-size:0.65rem;color:var(--muted);min-width:80px;text-align:right">Strongly Agree</span>
      </div>
    </div>`;
});

async function submitSUS() {
  const pid = document.getElementById('pid').value.trim() || 'anonymous';
  const data = { participant: pid };
  for (let i = 1; i <= 10; i++) {
    const sel = document.querySelector(`input[name="q${i}"]:checked`);
    if (!sel) { alert(`Please answer question ${i}`); return; }
    data[`q${i}`] = parseInt(sel.value);
  }
  const r = await fetch('/api/sus', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) });
  const d = await r.json();
  const res = document.getElementById('result');
  res.style.display = 'block';
  res.innerHTML = `<strong style="color:var(--accent)">SUS Score: ${d.sus_score}</strong><br>
    <span style="font-size:0.75rem;color:var(--muted)">${d.sus_score >= 70 ? 'Good' : d.sus_score >= 50 ? 'Acceptable' : 'Below threshold'}</span>`;
}

document.addEventListener('DOMContentLoaded', function() {
  initTheme();
  const btn = document.getElementById('themeToggle');
  if (btn) btn.addEventListener('click', toggleTheme);
});
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(
        DASHBOARD_HTML.replace("{{ poll_interval }}", str(POLL_INTERVAL))
    )


@app.route("/sus")
def sus_page():
    return render_template_string(SUS_HTML)


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.exception("Server error")
    return jsonify({"error": "internal server error"}), 500


def run_server() -> None:
    init_db()
    logger.info("Flask server on %s:%d", FLASK_HOST, FLASK_PORT)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)
