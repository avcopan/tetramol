"""4-Methylbicyclo[2.2.2]octan-1-amine, from a CREST conformer search."""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import crest, orient, render

NAME = "4-methylbicyclo[2.2.2]octan-1-amine"
SMILES = "CC12CCC(N)(CC1)CC2"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Conformers: the first run runs CREST; delete the -conformers.xyz file to rerun it
path = HERE / f"{NAME}-conformers.xyz"
geos = crest.conformers(SMILES, path)
energies = crest.energies(path)
for i, energy in enumerate(energies):
    print(f"{i}: {(energy - energies[0]) * 627.5095:6.2f} kcal/mol")  # noqa: T201

# %% Choose: look at others with `render(geos[i])` or `geom.view(geos[i])`
geo = geos[0]

# %% Orient: to change it, run `orient(geo)` and paste the printed rotation here
rot = Rotation.from_quat([-0.612569, -0.622363, -0.362399, -0.325715])
geo = geom.rotate(geo, rot)

# %% Write
geo.xyz_file(path=HERE / f"{NAME}.xyz")
render(geo, HERE / f"{NAME}.svg")
