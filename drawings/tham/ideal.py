"""THAM, tris(hydroxymethyl)aminomethane, on the ideal tetrahedral grid, N up."""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import Builder, orient, render

NAME = "tham"
METHOD = "ideal"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Build: C-N up, and three CH2OH arms that spiral the same way around the C-N
# axis (C3 symmetric, ignoring NH2); use -b.tangent(...) to spiral the other way
b = Builder(up=(0, 1, 0), left=(-1, 0, 0))
c = b.add("C")
n = b.add("N", c, 0)
for slot in b.open_slots(c):
    ch2 = b.add("C", c, slot)
    o = b.add("O", ch2, toward=b.tangent(ch2, c))
    b.add("H", o, toward=b.tangent(o, c))
b.add_hydrogens([n], avoid=(0, 0, -1))  # NH2 lone pair to the back
b.add_hydrogens()
geo = b.geometry()

# %% Orient: to change it, run `orient(geo)` and paste the printed rotation here
rot = Rotation.from_euler("yx", [-65, 30], degrees=True)
geo = geom.rotate(geo, rot)

# %% Write
geo.xyz_file(path=HERE / f"{METHOD}.xyz")
render(geo, HERE / f"{METHOD}.svg")
