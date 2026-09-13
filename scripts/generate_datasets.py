"""
Dataset generation entry point.

Preferred path: build from real Kaggle Customer Support on Twitter data.
Fallback: synthetic AppleSupport-style pairs (offline, no download required).
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_kaggle_build() -> bool:
    script = os.path.join(ROOT, "scripts", "build_corpus_from_kaggle.py")
    result = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True
    print("Kaggle build unavailable; using synthetic fallback.")
    if result.stderr:
        print(result.stderr[:500])
    return False


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    if not run_kaggle_build():
        fallback = os.path.join(ROOT, "scripts", "_synthetic_datasets.py")
        if os.path.exists(fallback):
            subprocess.run([sys.executable, fallback], cwd=ROOT, check=True)
        else:
            print("ERROR: No dataset source available.")
            sys.exit(1)
