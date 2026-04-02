#!/usr/bin/env python3
"""Fetch EMNLP 2019 BASIL into ~/basil_workspace/emnlp19-BASIL/data/."""

from __future__ import annotations

import zipfile
from pathlib import Path
import urllib.request

ARCHIVE_URL = "https://github.com/marshallwhiteorg/emnlp19-media-bias/archive/refs/heads/master.zip"


def main() -> None:
    root = Path.home() / "basil_workspace"
    root.mkdir(parents=True, exist_ok=True)

    archive_path = root / "basil_source.zip"
    extract_dir = root / "basil_source"
    combined_zip = extract_dir / "emnlp19-media-bias-master" / "emnlp19-BASIL.zip"
    combined_dir = root / "emnlp19-BASIL"
    dataset_dir = combined_dir / "data"

    if not archive_path.exists():
        print("Downloading", ARCHIVE_URL)
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)

    if not extract_dir.exists():
        print("Extracting", archive_path)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)

    if not combined_dir.exists():
        print("Extracting", combined_zip)
        with zipfile.ZipFile(combined_zip) as zf:
            zf.extractall(root)

    n = len(list(dataset_dir.glob("*.json")))
    print()
    print("BASIL data directory:")
    print(dataset_dir.resolve())
    print(f"JSON article files: {n}")
    print()
    print("Add to your shell or .zshrc:")
    print(f'  export BASIL_DATA_DIR="{dataset_dir.resolve()}"')


if __name__ == "__main__":
    main()
