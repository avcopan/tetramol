"""4-Methylbicyclo[2.2.2]octan-1-amine, from a CREST conformer search."""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import crest, orient, render, render_spins

NAME = "4-methylbicyclo[2.2.2]octan-1-amine"
METHOD = "crest"
SMILES = "CC12CCC(N)(CC1)CC2"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Conformers: the first run runs CREST; delete the -conformers.xyz file to rerun it
path = HERE / f"{METHOD}-conformers.xyz"
geos = crest.conformers(SMILES, path)
energies = crest.energies(path)
for i, energy in enumerate(energies):
    print(f"{i}: {(energy - energies[0]) * 627.5095:6.2f} kcal/mol")  # noqa: T201

# %% Choose: look at others with `render(geos[i])` or `geom.view(geos[i])`
geo = geos[0]

# %% Orient: methyl C and N on the y axis (N on top), then spin about y
c, n = 0, 5
geo = geom.translate(geo, -geo.coordinates[c])
geo = geom.rotate(geo, geom.axis_alignment_rotation(geo, (c, n), axis=(0, 1, 0)))
geo = geom.translate(geo, -(geo.coordinates[n] + geo.coordinates[c]) / 2)
rot = Rotation.from_euler("y", 60.0, degrees=True)
geo = geom.rotate(geo, rot)

# %% Write
geo.xyz_file(path=HERE / f"{METHOD}.xyz")
render(geo, HERE / f"{METHOD}.svg")
