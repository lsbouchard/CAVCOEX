#!/usr/bin/env python3
"""Write or verify SHA256SUMS for a publication or numerical-code release."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS"
TOP_LEVEL = (
    "manuscript.tex",
    "supplement.tex",
    "continuum_methods.tex",
    "cover_letter_PRE.tex",
    "Makefile",
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "README_OVERLEAF.txt",
    "REPRODUCIBILITY.md",
    "pytest.ini",
    "PRE_PHASE_SWITCH_VALIDATION.md",
    "environment.yml",
    "environment-audit.yml",
    "environment-lock.yml",
    "environment-osx-arm64.lock",
)
CODE_TOP_LEVEL = (
    "Makefile", "README.md", "NOTICE.md", "LICENSE", "environment.yml", "environment-audit.yml",
    "environment-lock.yml", "environment-osx-arm64.lock", "REPRODUCIBILITY.md", "pytest.ini",
)
OPTIONAL_PUBLICATION_FILES = ("AUDIT_20260929.md", "REAUDIT_SOL_20260929.md")
OPTIONAL_SHARED_FILES = (
    "NUMERICAL_AUDIT_20260929.md", "REAUDIT_STATISTICS_SOL_20260929.md",
    "CONTINUUM_SWITCHING_20260929.md",
    "SINGLE_WALKER_CONTINUUM_PLAN_20260929.md",
)
DIRECTORIES = (
    "scripts",
    "tests",
    "data",
    "data_raw",
    "data_processed",
    "generated",
    "figures",
)
FINAL_PDFS = (
    "output/pdf/extensive_response_to_a_local_exclusion_cavity_at_first_order_coexistence_PRE.pdf",
    "output/pdf/extensive_response_to_a_local_exclusion_cavity_at_first_order_coexistence_supplement.pdf",
    "output/pdf/cover_letter_PRE.pdf",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mplconfig", ".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
LEAN_CONTROL_METADATA = (
    "data_raw/controls/single_walker_capped_control_20260929.json",
)


def retained_control_sources(root: Path) -> set[Path]:
    """Resolve only exact, integrity-verified source dependencies of lean controls."""
    import json
    result = set()
    resolved_root = root.resolve()
    snapshot_root = (resolved_root / "data_raw" / "source_snapshots").resolve()
    if not snapshot_root.is_relative_to(resolved_root):
        raise ValueError("Source snapshot directory must remain inside the project")
    for name in LEAN_CONTROL_METADATA:
        sidecar = root / name
        if not sidecar.exists():
            continue
        record = json.loads(sidecar.read_text())
        raw = sidecar.with_suffix(".npz")
        if sha256_file(raw) != record.get("raw_sha256"):
            raise ValueError("Retained control raw digest mismatch: " + name)
        snapshots = record.get("source_snapshots", {})
        if not {"driver", "kernel", "graph_validation", "independent_control"} <= snapshots.keys():
            raise ValueError("Retained control must identify its executed source dependencies")
        for relative, digest in snapshots.values():
            relative_path = Path(relative)
            source = (resolved_root / relative_path).resolve()
            if (relative_path.is_absolute() or not source.is_relative_to(resolved_root)
                    or not source.is_relative_to(snapshot_root)
                    or sha256_file(source) != digest):
                raise ValueError("Retained control source path/digest is invalid: " + relative)
            result.add(source)
    return result


def excluded_from_code(path: Path, root: Path, control_sources: set[Path] | None = None) -> bool:
    """Keep failed exploratory histories in the full audit, not lean Git data."""
    name = path.relative_to(root).as_posix()
    if name == 'generated/revision_locations.tex':
        return True  # Journal-response labels are not numerical release inputs.
    if name.startswith(("data_raw/continuum", "data_raw/controls/continuum")):
        return True
    if name.startswith(("data_raw/source_snapshots/continuum",
                        "data_raw/source_snapshots/test_single_walker_")):
        retained = retained_control_sources(root) if control_sources is None else control_sources
        return path.resolve() not in retained
    if name.startswith(("data_processed/continuum", "figures/continuum",
                        "generated/continuum")):
        return name not in {
            "data_processed/continuum_diagnostic_summary.json",
            "generated/continuum_diagnostic_summary_metadata.json",
        }
    return False


def sha256_file(path: Path) -> str:
    """Hash large trajectory archives without materializing them in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path = ROOT, profile: str = "publication") -> list[Path]:
    required = (*TOP_LEVEL, *FINAL_PDFS) if profile == "publication" else CODE_TOP_LEVEL
    missing = [name for name in (*required, *DIRECTORIES) if not (root / name).exists()]
    if missing:
        raise FileNotFoundError("required release inputs missing: " + ", ".join(missing))
    paths = [root / name for name in required]
    paths.extend(root / name for name in OPTIONAL_SHARED_FILES if (root / name).is_file())
    if profile == "publication":
        paths.extend(root / name for name in OPTIONAL_PUBLICATION_FILES if (root / name).is_file())
    for directory in DIRECTORIES:
        paths.extend(path for path in (root / directory).rglob("*") if path.is_file())
    control_sources = retained_control_sources(root) if profile == "code" else set()
    return sorted(
        path
        for path in paths
        if not EXCLUDED_PARTS.intersection(path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and not (profile == "code" and excluded_from_code(path, root, control_sources))
    )


def manifest_text(root: Path = ROOT, profile: str = "publication") -> str:
    lines = []
    for path in included_files(root, profile):
        digest = sha256_file(path)
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("publication", "code"), default="publication")
    parser.add_argument("--check", action="store_true", help="verify without changing files")
    args = parser.parse_args()
    expected = manifest_text(profile=args.profile)
    if args.check:
        if not OUTPUT.exists():
            raise SystemExit("SHA256SUMS is missing; freeze the release before verification")
        actual = OUTPUT.read_text(encoding="utf-8")
        if actual != expected:
            old = {line.split("  ", 1)[1]: line for line in actual.splitlines()}
            new = {line.split("  ", 1)[1]: line for line in expected.splitlines()}
            changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
            raise SystemExit("release integrity mismatch:\n" + "\n".join(changed))
        print(f"verified {len(expected.splitlines())} release files")
    else:
        OUTPUT.write_text(expected, encoding="utf-8")
        print(f"wrote {len(expected.splitlines())} checksums to {OUTPUT.name}")


if __name__ == "__main__":
    main()
