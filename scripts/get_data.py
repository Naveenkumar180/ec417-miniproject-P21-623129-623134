"""Download the ECG Heartbeat Categorization CSVs into data/raw/.

Uses the Kaggle CLI (needs a free Kaggle account + API token, see
https://www.kaggle.com/docs/api).  Dataset slug used by the course materials'
"commonly used 109,446-beat CSV": shayanfazeli/heartbeat  -- CHECK on kaggle.com that
this is the copy your instructor approved, and record the access date in data/README.md.

Manual fallback: download the zip from the Kaggle page in a browser and put
mitbih_train.csv and mitbih_test.csv into data/raw/ yourself, then re-run this
script to verify shapes and write checksums.

    python scripts/get_data.py
"""
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

SLUG = "shayanfazeli/heartbeat"
RAW = Path("data/raw")
NEEDED = ["mitbih_train.csv", "mitbih_test.csv"]
EXPECTED_ROWS = {"mitbih_train.csv": 87554, "mitbih_test.csv": 21892}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if not all((RAW / f).exists() for f in NEEDED):
        print(f"Downloading {SLUG} with the Kaggle CLI ...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", SLUG, "-p", str(RAW)], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            sys.exit(f"Kaggle download failed ({e}). Download manually and place "
                     f"{NEEDED} in {RAW}/, then re-run.")
        for z in RAW.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(RAW)
            z.unlink()

    sums = []
    for f in NEEDED:
        p = RAW / f
        if not p.exists():
            sys.exit(f"missing {p}")
        df = pd.read_csv(p, header=None)
        ok = df.shape == (EXPECTED_ROWS[f], 188)
        print(f"{f}: shape {df.shape} {'OK' if ok else 'UNEXPECTED (expected %d x 188)' % EXPECTED_ROWS[f]}")
        sums.append(f"{sha256(p)}  {f}")
    (RAW / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    total = sum(EXPECTED_ROWS.values())
    print(f"Expected total beats: {total} (should equal 109,446). Checksums written to {RAW}/SHA256SUMS.txt")


if __name__ == "__main__":
    main()
