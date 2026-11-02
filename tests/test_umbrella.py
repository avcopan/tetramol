"""Umbrella-image tests: agreement with the builder, and edge-on mirror planes."""

import numpy as np
import pytest
from automol import Geometry, geom
from scipy.spatial.transform import Rotation

from tetramol import Builder, umbrella_edge_on, umbrella_image


def _tham() -> tuple[Builder, int]:
    """THAM on the grid, with the NH2 lone pair to the back."""
    b = Builder(up=(0, 1, 0), left=(-1, 0, 0))
    c = b.add("C")
    n = b.add("N", c, 0)
    for slot in b.open_slots(c):
        ch2 = b.add("C", c, slot)
        o = b.add("O", ch2, toward=b.tangent(ch2, c))
        b.add("H", o, toward=b.tangent(o, c))
    b.add_hydrogens([n], avoid=(0, 0, -1))
    b.add_hydrogens()
    return b, n


def _nh_hydrogens(geo: Geometry, n: int) -> list[int]:
    amat = geom.adjacency_matrix(geo)
    return [int(k) for k in np.flatnonzero(amat[n]) if geo.symbols[k] == "H"]


def test__umbrella_image_matches_builder() -> None:
    """Test that the geometry version agrees with `Builder.umbrella_image`."""
    b, n = _tham()
    found, keys = umbrella_image(b.geometry(), n)
    expected, expected_keys = b.umbrella_image(n)
    assert keys == expected_keys
    assert found.symbols == expected.symbols
    assert np.allclose(found.coordinates, expected.coordinates)


@pytest.mark.parametrize("seed", range(5))
def test__umbrella_edge_on(seed: int) -> None:
    """Test that after the twist, each image H is at the same depth as its H."""
    b, n = _tham()
    geo = geom.rotate(b.geometry(), Rotation.random(random_state=seed))
    twisted = umbrella_edge_on(geo, n)
    hs = _nh_hydrogens(twisted, n)

    # Only the NH2 moves, rigidly about the C-N bond: lengths and angles are kept
    others = [k for k in range(geo.atom_count) if k not in hs]
    assert np.allclose(twisted.coordinates[others], geo.coordinates[others])
    for h in hs:
        assert np.isclose(geom.distance(twisted, [n, h]), geom.distance(geo, [n, h]))
        assert np.isclose(geom.angle(twisted, [0, n, h]), geom.angle(geo, [0, n, h]))

    # Edge-on: the image of each H has the same z, i.e. the plane contains z
    umbrella_geo, image = umbrella_image(twisted, n)
    assert np.allclose(
        np.sort(umbrella_geo.coordinates[image, 2]),
        np.sort(umbrella_geo.coordinates[hs, 2]),
    )
    # Already edge-on: twisting again does nothing
    assert np.allclose(umbrella_edge_on(twisted, n).coordinates, twisted.coordinates)


def test__umbrella_edge_on_bond_at_viewer_raises() -> None:
    """Test that a bond pointing at the viewer has no edge-on mirror plane."""
    b, n = _tham()
    geo = geom.rotate(b.geometry(), Rotation.from_euler("x", 90, degrees=True))
    with pytest.raises(ValueError, match="points at the viewer"):
        umbrella_edge_on(geo, n)


def test__umbrella_image_parent_required() -> None:
    """Test that a center with several heavy neighbors needs an explicit parent."""
    b, n = _tham()
    with pytest.raises(ValueError, match="give the `parent`"):
        umbrella_image(b.geometry(), 0)
    # With a parent given, it works (here mirroring the rest of the molecule)
    _, keys = umbrella_image(b.geometry(), 0, parent=n)
    assert len(keys) == b.atom_count - 4  # all but this C, the N, and its two H's
