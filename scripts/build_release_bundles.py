#!/usr/bin/env python3
"""Package the current frozen release without modifying historical archives."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import zipfile

from make_release_manifest import FINAL_PDFS, ROOT, included_files, manifest_text, sha256_file


def tex_dependencies(path: Path, seen: set[Path] | None = None) -> set[Path]:
    """Resolve local TeX inputs and exactly the figures they reference."""
    seen = set() if seen is None else seen
    if path in seen:
        return seen
    if not path.is_file():
        raise FileNotFoundError(path)
    seen.add(path)
    text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))
    for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
        child = ROOT / name
        if not child.suffix:
            child = child.with_suffix(".tex")
        tex_dependencies(child, seen)
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)
        seen.add(ROOT / name)
    return seen


def write_bundle(output: Path, members: dict[str, Path]) -> None:
    """Create a new archive with an internal manifest and CRC verification."""
    checksums = "".join(
        f"{sha256_file(path)}  {name}\n"
        for name, path in sorted(members.items())
    )
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr("SHA256SUMS", checksums)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"CRC verification failed: {output}")
    print(f"created {output.name}: {len(members)} files plus SHA256SUMS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="release tag, e.g. audit_20260929")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag may contain only letters, numbers, underscores and hyphens")
    if (ROOT / "SHA256SUMS").read_text(encoding="utf-8") != manifest_text():
        raise SystemExit("current release does not match SHA256SUMS; freeze and verify first")
    outputs = [ROOT / f"CAVCOEX_PRE_{kind}_{args.tag}.zip" for kind in ("Overleaf", "reproducibility")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("a tagged archive already exists; use a new tag to preserve history")
    compact = {ROOT / "README_OVERLEAF.txt"}
    for name in ("manuscript.tex", "supplement.tex", "cover_letter_PRE.tex"):
        compact.update(tex_dependencies(ROOT / name))
    compact_members = {path.relative_to(ROOT).as_posix(): path for path in compact}
    compact_members.update({Path(name).name: ROOT / name for name in FINAL_PDFS})
    full_members = {path.relative_to(ROOT).as_posix(): path for path in included_files()}
    write_bundle(outputs[0], compact_members)
    write_bundle(outputs[1], full_members)


if __name__ == "__main__":
    main()
