"""Conformers from CREST, for drawing real (not idealized) geometries.

CREST runs in its own Pixi environment (``pixi run -e crest crest``), because it
needs an older libgfortran than SciPy (see ``pixi.toml``).
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from automol import Geometry, geom, rdkit_smiles

CONFORMERS_FILE = "crest_conformers.xyz"
"""The file in which CREST writes its conformers, lowest energy first."""

LOG_TAIL = 20
"""Number of lines of CREST's log to show when it fails."""


def conformers(
    smiles: str,
    path: str | Path,
    *,
    threads: int = 4,
    args: Sequence[str] = ("--gfn2",),
) -> list[Geometry]:
    """Get the conformers of a molecule, running CREST unless they are cached.

    The first call embeds a geometry for `smiles` with RDKit, runs a CREST
    conformer search on it in a temporary directory, and copies CREST's
    conformers to `path`. Later calls read them from `path`; delete it to rerun
    CREST.

    Args:
        smiles: SMILES string.
        path: Multi-frame xyz file to cache the conformers in, lowest energy
            first, with each energy (in Hartree) on its comment line.
        threads: Number of threads for CREST.
        args: Further command-line arguments for CREST, e.g. the method.

    Returns:
        The conformers, lowest energy first.

    Raises:
        FileNotFoundError: If CREST can't be found.
        RuntimeError: If CREST fails. The error shows the end of its log.
    """
    path = Path(path)
    geo = rdkit_smiles.geometry_fn(smiles)
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix="crest-") as tmp:
            work = Path(tmp)
            geo.xyz_file(path=work / "input.xyz")
            cmd = [*_crest_command(), "input.xyz", "-T", str(threads), *args]
            log = work / "crest.log"
            with log.open("w") as stream:
                proc = subprocess.run(  # noqa: S603
                    cmd, cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if proc.returncode or not (work / CONFORMERS_FILE).exists():
                tail = "\n".join(log.read_text().splitlines()[-LOG_TAIL:])
                msg = f"CREST failed (exit code {proc.returncode}):\n{tail}"
                raise RuntimeError(msg)
            shutil.copyfile(work / CONFORMERS_FILE, path)
    return geom.from_xyz_trajectory_file(path, charge=geo.charge, spin=geo.spin)


def energies(path: str | Path) -> list[float]:
    """Read the conformer energies that CREST writes on the comment lines.

    Args:
        path: Conformer file, as written by `conformers`.

    Returns:
        The energies, in Hartree, in the order of the conformers.
    """
    return [float(c) for c in geom.xyz_trajectory_comments(Path(path).read_text())]


def _crest_command() -> list[str]:
    """Get the command that runs CREST: from PATH, or from the Pixi environment."""
    if shutil.which("crest"):
        return ["crest"]
    root = os.environ.get("PIXI_PROJECT_ROOT")
    if root and shutil.which("pixi"):
        manifest = str(Path(root) / "pixi.toml")
        return ["pixi", "run", "--manifest-path", manifest, "-e", "crest", "crest"]
    msg = (
        "CREST was not found. Run this inside the project's Pixi environment, "
        "which runs CREST with `pixi run -e crest crest`."
    )
    raise FileNotFoundError(msg)
