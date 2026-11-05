"""
End-to-end pipeline runner.

Runs every stage of the lab in the correct order:
  1. generate_data.py        - synthetic dataset
  2. eda.py                  - exploratory plots
  3. baseline.py             - naive/seasonal/rolling baselines
  4. model_linear.py         - Linear / Ridge / Lasso
  5. model_lgbm.py           - LightGBM global + per-SKU
  6. model_prophet.py        - Prophet (+ RU holidays)
  7. model_lstm.py           - LSTM (PyTorch)
  8. model_bayesian.py       - BayesianRidge + intervals
  9. ensemble.py             - weighted ensemble
 10. summarize.py            - summary table + bar chart
 11. forecast.py             - interactive forecast (auto-answered)
 12. build_report.py         - generate Word report

Run from the project root:
    python run_pipeline.py                    # full pipeline, forecast=30d, all SKUs
    python run_pipeline.py --horizon 90       # different forecast horizon
    python run_pipeline.py --skip-forecast    # skip interactive forecast
    python run_pipeline.py --skip-report      # skip Word report
    python run_pipeline.py --only model_lgbm  # run a single stage
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (short_name, script, description)
STAGES = [
    ("generate",    "generate_data.py",   "Generate synthetic sales data"),
    ("eda",         "eda.py",             "Exploratory data analysis"),
    ("baseline",    "baseline.py",        "Baseline models"),
    ("linear",      "model_linear.py",    "Linear / Ridge / Lasso"),
    ("lgbm",        "model_lgbm.py",      "LightGBM global + per-SKU"),
    ("prophet",     "model_prophet.py",   "Prophet + RU holidays"),
    ("lstm",        "model_lstm.py",      "LSTM (PyTorch)"),
    ("bayesian",    "model_bayesian.py",  "Bayesian Ridge + intervals"),
    ("ensemble",    "ensemble.py",        "Weighted ensemble"),
    ("summarize",   "summarize.py",       "Summary table + bar chart"),
    ("forecast",    "forecast.py",        "Interactive forecast"),
    ("report",      "build_report.py",    "Build Word report"),
]


def run_stage(name: str, script: str, stdin_text: str | None = None) -> tuple[bool, float]:
    script_path = ROOT / script
    if not script_path.exists():
        print(f"  [SKIP] {script} not found")
        return False, 0.0

    print(f"\n{'=' * 70}\n  {name.upper()} :: {script}\n{'=' * 70}")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            input=stdin_text,
            text=True,
            capture_output=False,   # let output stream live
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  [FAIL] {script} exited with code {result.returncode}")
            return False, elapsed
        print(f"  [OK]   {script}  ({elapsed:.1f}s)")
        return True, elapsed
    except Exception as e:
        print(f"  [FAIL] {script}: {e}")
        return False, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Run the full lab pipeline.")
    ap.add_argument("--horizon", type=int, default=30,
                    help="forecast horizon in days (default 30)")
    ap.add_argument("--sku", default="all",
                    help="SKU to forecast (default 'all')")
    ap.add_argument("--skip-forecast", action="store_true",
                    help="skip the interactive forecast stage")
    ap.add_argument("--skip-report", action="store_true",
                    help="skip the Word report stage")
    ap.add_argument("--only", default=None,
                    help="run only one stage (name from --list)")
    ap.add_argument("--list", action="store_true",
                    help="list stage names and exit")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="do not stop on first failure")
    args = ap.parse_args()

    if args.list:
        print("Available stages:")
        for n, s, d in STAGES:
            print(f"  {n:12s} {s:25s} {d}")
        return

    targets = STAGES
    if args.only:
        targets = [s for s in STAGES if s[0] == args.only]
        if not targets:
            print(f"Unknown stage '{args.only}'. Use --list.")
            sys.exit(1)

    # Skip forecast/report if requested
    if args.skip_forecast:
        targets = [s for s in targets if s[0] != "forecast"]
    if args.skip_report:
        targets = [s for s in targets if s[0] != "report"]

    print(f"Pipeline: {len(targets)} stages  |  Python: {sys.executable}")
    print(f"Root    : {ROOT}")
    print(f"Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    forecast_stdin = f"{args.horizon}\n{args.sku}\n"
    timings = []
    failed = []

    t_start = time.time()
    for name, script, _ in targets:
        stdin = forecast_stdin if name == "forecast" else None
        ok, dt = run_stage(name, script, stdin_text=stdin)
        timings.append((name, ok, dt))
        if not ok:
            failed.append(name)
            if not args.continue_on_error:
                print(f"\nStopping on failure in '{name}'. "
                      f"Use --continue-on-error to ignore.")
                break

    total = time.time() - t_start

    # ---- final report ----
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for name, ok, dt in timings:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {name:12s} {dt:7.1f}s")
    print(f"\n  Total time: {total:.1f}s")
    print(f"  Finished  : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if failed:
        print(f"\n  Failed stages: {', '.join(failed)}")
        sys.exit(1)

    # ---- artifacts check ----
    print("\n" + "=" * 70)
    print("OUTPUT ARTIFACTS")
    print("=" * 70)

    reports_dir = ROOT / "reports"
    if reports_dir.exists():
        files = sorted(reports_dir.glob("*"))
        png = [f for f in files if f.suffix == ".png"]
        csv = [f for f in files if f.suffix == ".csv"]
        print(f"  reports/  {len(files)} files  "
              f"({len(png)} PNG, {len(csv)} CSV)")

    docx = ROOT / "Отчет_лабораторная.docx"
    if docx.exists():
        size_kb = docx.stat().st_size / 1024
        print(f"  Report    : {docx.name}  ({size_kb:.1f} KB)")

    data = ROOT / "data" / "sales_data.csv"
    if data.exists():
        print(f"  Dataset   : {data.name}  ({data.stat().st_size / 1024:.1f} KB)")

    print("\nDone. Open 'reports/' or 'Отчет_лабораторная.docx'.")


if __name__ == "__main__":
    main()