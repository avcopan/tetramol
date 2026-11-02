"""THAM, tris(hydroxymethyl)aminomethane, from a CREST conformer search."""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import crest, orient, orient_multi, render, render_spins

NAME = "tham"
METHOD = "crest"
SMILES = "NC(CO)(CO)CO"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Conformers: the first run runs CREST; delete the -conformers.xyz file to rerun it
path = HERE / f"{METHOD}-conformers.xyz"
geos = crest.conformers(SMILES, path)
energies = crest.energies(path)
for i, energy in enumerate(energies):
    print(f"{i}: {(energy - energies[0]) * 627.5095:6.2f} kcal/mol")  # noqa: T201

# %% Choose: to flip through the conformers (and rotate them) in the browser, run
# `orient_multi(geos, labels=labels)` and paste the printed `geo = geos[i]` here
geo = geos[13]
rot = Rotation.from_quat([0.082113, 0.690324, 0.054467, -0.716759])

# %% Write
geo.xyz_file(path=HERE / f"{METHOD}.xyz")
render(geo, HERE / f"{METHOD}.svg")
