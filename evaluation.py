"""
evaluation.py — Quantitative and qualitative evaluation tools.

Provides:
  - Pearson correlation between system scores and self-reports
  - SUS score aggregation and export
  - Latency performance report
  - Interview guide (printed)
  - Thematic analysis scaffold
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import DATA_DIR, LOGS_DIR, SELF_REPORT_DB

logger = logging.getLogger(__name__)


# ─── Database Helpers ─────────────────────────────────────────────────────────

def _db_connect():
    conn = sqlite3.connect(str(SELF_REPORT_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Correlation Analysis ─────────────────────────────────────────────────────

def load_self_reports(days: float = 14) -> List[Tuple[float, float]]:
    """Load (ts, score) tuples from self_reports within the past N days."""
    cutoff = time.time() - days * 86400
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT ts, score FROM self_reports WHERE ts >= ? ORDER BY ts",
            (cutoff,)
        ).fetchall()
    return [(r["ts"], r["score"]) for r in rows]


def load_system_scores(days: float = 14) -> List[Tuple[float, float]]:
    """
    Load (ts, stress_score) from the feature JSONL files.
    Falls back to status log if available.
    """
    results = []
    cutoff = time.time() - days * 86400
    features_dir = DATA_DIR / "features"

    for fpath in sorted(features_dir.glob("features_*.jsonl")):
        try:
            with open(fpath) as fh:
                for line in fh:
                    rec = json.loads(line)
                    ts = rec.get("ts", 0)
                    if ts >= cutoff and "norm" in rec:
                        norm = rec["norm"]
                        # Proxy score: mean absolute normalized deviation
                        proxy = float(np.mean(np.abs(norm)))
                        results.append((ts, proxy))
        except (json.JSONDecodeError, OSError):
            continue

    results.sort(key=lambda x: x[0])
    return results


def compute_correlation(
    reports: List[Tuple[float, float]],
    scores:  List[Tuple[float, float]],
    window_minutes: float = 10.0,
) -> Optional[dict]:
    """
    Align self-reports with nearest system score within ±window_minutes.
    Compute Pearson r, p-value, and 95% CI via bootstrap.

    Returns dict with keys: r, p_value, n, ci_low, ci_high, pairs
    """
    if not reports or not scores:
        logger.warning("Insufficient data for correlation (reports=%d scores=%d)",
                       len(reports), len(scores))
        return None

    window_sec = window_minutes * 60
    score_arr  = np.array(scores)        # (N, 2)

    aligned_reports = []
    aligned_scores  = []

    for rep_ts, rep_score in reports:
        # Find nearest system score
        time_diffs = np.abs(score_arr[:, 0] - rep_ts)
        nearest_idx = int(np.argmin(time_diffs))
        if time_diffs[nearest_idx] <= window_sec:
            aligned_reports.append(rep_score)
            aligned_scores.append(score_arr[nearest_idx, 1])

    n = len(aligned_reports)
    if n < 5:
        logger.warning("Only %d aligned pairs — need ≥5 for correlation", n)
        return None

    r_arr = np.array(aligned_reports, dtype=float)
    s_arr = np.array(aligned_scores,  dtype=float)

    # Pearson r
    r = float(np.corrcoef(r_arr, s_arr)[0, 1])

    # p-value approximation (t-distribution)
    from scipy import stats as sp_stats
    t_stat, p_val = sp_stats.pearsonr(r_arr, s_arr)

    # Bootstrap 95% CI
    boot_r = []
    rng = np.random.default_rng(42)
    for _ in range(2000):
        idx = rng.choice(n, size=n, replace=True)
        if np.std(r_arr[idx]) < 1e-9 or np.std(s_arr[idx]) < 1e-9:
            continue
        boot_r.append(float(np.corrcoef(r_arr[idx], s_arr[idx])[0, 1]))

    ci_low  = float(np.percentile(boot_r, 2.5))
    ci_high = float(np.percentile(boot_r, 97.5))

    result = {
        "r":        round(r, 4),
        "p_value":  round(float(p_val), 6),
        "n":        n,
        "ci_low":   round(ci_low, 4),
        "ci_high":  round(ci_high, 4),
        "pairs":    list(zip(r_arr.tolist(), s_arr.tolist())),
        "meets_target": abs(r) >= 0.55,
    }
    logger.info("Correlation: r=%.3f p=%.4f n=%d CI=[%.3f, %.3f]",
                r, p_val, n, ci_low, ci_high)
    return result


# ─── SUS Export ───────────────────────────────────────────────────────────────

def export_sus_results(output_path: Optional[Path] = None) -> dict:
    """Aggregate SUS scores by participant and overall."""
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT participant, sus_score, ts FROM sus_responses ORDER BY ts"
        ).fetchall()

    if not rows:
        return {"mean": None, "n": 0, "by_participant": {}}

    scores = [r["sus_score"] for r in rows]
    by_part = {}
    for r in rows:
        by_part.setdefault(r["participant"], []).append(r["sus_score"])

    result = {
        "mean":             round(float(np.mean(scores)), 1),
        "std":              round(float(np.std(scores)), 1),
        "median":           round(float(np.median(scores)), 1),
        "n":                len(scores),
        "meets_target":     float(np.mean(scores)) >= 65,
        "by_participant":   {p: round(float(np.mean(s)), 1) for p, s in by_part.items()},
        "individual_scores": scores,
    }

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info("SUS results exported → %s", output_path)

    return result


# ─── Latency Report ───────────────────────────────────────────────────────────

def latency_report(days: float = 7) -> dict:
    """Aggregate latency statistics by component."""
    cutoff = time.time() - days * 86400
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT component, latency_ms FROM latency_log WHERE ts >= ?",
            (cutoff,)
        ).fetchall()

    by_component = {}
    for r in rows:
        by_component.setdefault(r["component"], []).append(r["latency_ms"])

    report = {}
    for comp, lats in by_component.items():
        arr = np.array(lats)
        report[comp] = {
            "mean_ms":  round(float(arr.mean()), 1),
            "p50_ms":   round(float(np.percentile(arr, 50)), 1),
            "p95_ms":   round(float(np.percentile(arr, 95)), 1),
            "p99_ms":   round(float(np.percentile(arr, 99)), 1),
            "max_ms":   round(float(arr.max()), 1),
            "n":        len(arr),
        }

    return report


# ─── Interview Guide ─────────────────────────────────────────────────────────

INTERVIEW_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║         CogHealth Semi-Structured Interview Guide            ║
║         Duration: 20-30 minutes | Format: 1-on-1            ║
╚══════════════════════════════════════════════════════════════╝

SECTION 1: GENERAL EXPERIENCE (5 min)
─────────────────────────────────────
1. How would you describe your overall experience using the
   CogHealth monitoring system over the past [N] days?

2. Did the system fit naturally into your daily routine?
   What worked well, and what felt disruptive?

3. Were there times when you were particularly aware of the
   system running in the background?

SECTION 2: Light FEEDBACK (5 min)
────────────────────────────────
4. What did you notice about the light changes?
   Could you interpret the color shifts intuitively?

5. Did the ambient Light feedback ever influence your behavior
   or awareness? If so, how?

6. Were there times when the light color seemed misaligned with
   how you actually felt? Can you describe those moments?

SECTION 3: STRESS ALIGNMENT (8 min)
─────────────────────────────────────
7. When you submitted self-reports, how did the system's
   stress indicator compare to your subjective feeling?

8. Can you recall a specific high-stress moment (deadline,
   exam, etc.)? Did the system respond as you would expect?

9. Were there false alarms — times the system indicated high
   stress when you felt calm? How often did this happen?

10. Did having access to the dashboard change how you think
    about your own stress levels or work patterns?

SECTION 4: PRIVACY AND TRUST (5 min)
──────────────────────────────────────
11. How comfortable were you knowing the system was monitoring
    your keyboard and mouse behavior? Did your comfort level
    change over time?

12. Did the privacy safeguards (local processing, no key logging,
    auto-deletion) affect your trust in the system?

13. Is there any aspect of data collection you would want to
    change or have more control over?

SECTION 5: USABILITY AND FUTURE USE (5 min)
─────────────────────────────────────────────
14. If this system were available as a product, would you use it?
    What would need to change?

15. Who do you think would benefit most from this system?

16. Is there anything else about your experience you'd like
    to share that we haven't covered?

─────────────────────────────────────────────────────────────
THEMATIC ANALYSIS CODES (use for NVivo / manual coding):
  [PRIVA] Privacy concerns or comfort
  [ALIGN] System-self report alignment / misalignment
  [LED]   LED feedback interpretation and influence
  [USAB]  Usability, integration into routine
  [TRUST] Trust in technology / data handling
  [AWAR]  Increased self-awareness of stress
  [DISRT] Disruption or intrusiveness
  [IMPRO] Suggestions for improvement
─────────────────────────────────────────────────────────────
"""


def print_interview_guide() -> None:
    print(INTERVIEW_GUIDE)


# ─── Full Evaluation Report ───────────────────────────────────────────────────

def generate_report(output_path: Optional[Path] = None) -> dict:
    """Generate a comprehensive evaluation report."""
    logger.info("Generating evaluation report")

    reports = load_self_reports(days=14)
    scores  = load_system_scores(days=14)
    corr    = compute_correlation(reports, scores)
    sus     = export_sus_results()
    lats    = latency_report()

    report = {
        "generated_ts": time.time(),
        "correlation": corr,
        "sus": sus,
        "latency": lats,
        "n_self_reports": len(reports),
        "n_system_scores": len(scores),
        "summary": {
            "correlation_r":     corr["r"] if corr else None,
            "meets_correlation": corr["meets_target"] if corr else False,
            "mean_sus":          sus["mean"],
            "meets_sus":         sus["meets_target"],
            "inference_p95_ms":  lats.get("inference", {}).get("p95_ms"),
        }
    }

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Evaluation report → %s", output_path)

    # Print summary
    print("\n" + "═" * 60)
    print("  COGHEALTH EVALUATION SUMMARY")
    print("═" * 60)
    if corr:
        r_ok = "✓" if corr["meets_target"] else "✗"
        print(f"  Correlation r = {corr['r']:.3f} {r_ok} (target ≥0.55)")
        print(f"  p-value = {corr['p_value']:.4f}  n = {corr['n']}")
        print(f"  95% CI = [{corr['ci_low']:.3f}, {corr['ci_high']:.3f}]")
    else:
        print("  Correlation: insufficient data")
    sus_ok = "✓" if sus["meets_target"] else "✗"
    print(f"  Mean SUS = {sus['mean']} {sus_ok} (target ≥65, n={sus['n']})")
    if lats.get("inference"):
        print(f"  Inference P95 = {lats['inference']['p95_ms']} ms (target <500)")
    print("═" * 60 + "\n")

    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if "--interview" in sys.argv:
        print_interview_guide()
    else:
        out = Path("evaluation_report.json")
        generate_report(output_path=out)
