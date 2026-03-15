#!/usr/bin/env python3
"""
Master Pipeline Runner
=======================
Executes the entire ML pipeline end-to-end:
  1. Preprocess raw CSV
  2. Exploratory Data Analysis
  3. DBSCAN spatial clustering
  4. Random Forest severity classification
  5. Accident Risk Index calculation
  6. (Optional) Seed MySQL database

Usage:
    python run_pipeline.py              # Run steps 1-5 (no DB)
    python run_pipeline.py --seed-db    # Run steps 1-6 (with DB)
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run the accident prediction ML pipeline")
    parser.add_argument("--seed-db", action="store_true",
                        help="Also seed results into MySQL after pipeline completes")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (use existing processed CSV)")
    args = parser.parse_args()

    t0 = time.time()

    # Step 1 — Preprocessing
    if not args.skip_preprocess:
        print("=" * 60)
        print("STEP 1 / 5  —  Data Preprocessing")
        print("=" * 60)
        from scripts.preprocess import run as preprocess_run
        preprocess_run()
        print()

    # Step 2 — EDA
    print("=" * 60)
    print("STEP 2 / 5  —  Exploratory Data Analysis")
    print("=" * 60)
    from scripts.eda import run as eda_run
    eda_run()
    print()

    # Step 3 — DBSCAN Clustering
    print("=" * 60)
    print("STEP 3 / 5  —  DBSCAN Spatial Clustering")
    print("=" * 60)
    from scripts.clustering import run as clustering_run
    clustering_run()
    print()

    # Step 4 — Random Forest Classification
    print("=" * 60)
    print("STEP 4 / 5  —  Random Forest Classification")
    print("=" * 60)
    from scripts.classifier import run as classifier_run
    classifier_run()
    print()

    # Step 5 — ARI Calculation
    print("=" * 60)
    print("STEP 5 / 5  —  Accident Risk Index (ARI)")
    print("=" * 60)
    from scripts.ari import run as ari_run
    ari_run()
    print()

    # Optional Step 6 — Seed MySQL
    if args.seed_db:
        print("=" * 60)
        print("STEP 6 (Optional)  —  Seed MySQL Database")
        print("=" * 60)
        from scripts.seed_db import run as seed_run
        seed_run()
        print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start the API server:  python app.py")
    print("  2. Visit http://localhost:5000/ to see available endpoints")
    if not args.seed_db:
        print("  3. (Optional) Seed MySQL:  python run_pipeline.py --seed-db")


if __name__ == "__main__":
    main()
