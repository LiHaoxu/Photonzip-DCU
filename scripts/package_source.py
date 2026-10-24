#!/usr/bin/env python3
"""Create a source-only PhotonZip ZIP beside the repository."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "PhotonZip_pfpl_rel_lc_hdf5.zip"
EXCLUDED_DIRS = {".git", "build", "__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return not (
        any(part in EXCLUDED_DIRS for part in relative.parts)
        or path.suffix in EXCLUDED_SUFFIXES
        or path.name.endswith(".h5")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.unlink(missing_ok=True)
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity test failed at {bad_member}")

    print(output)
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")
    print(f"files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
