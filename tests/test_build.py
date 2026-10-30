"""Builder tests: ideal bonds and angles, the diamond lattice, and slot choice."""

import itertools

import numpy as np
import pytest
from automol import Geometry, geom
from scipy.spatial.transform import Rotation

from tetramol import TETRAHEDRAL_ANGLE, XH_LENGTH, XY_LENGTH, Builder


def _neopentane_amine(**kwargs: object) -> Builder:
    """C(N)(C)(C)C with hydrogens: the user's 'tripod under a C-N bond'."""
    b = Builder(**kwargs)  # ty: ignore[invalid-argument-type]
    c = b.add("C")
    b.add("N", c, 0)  # along `up`
    for slot in b.open_slots(c):
        b.add("C", c, slot)
    b.add_hydrogens()
    return b


def _bonded_pairs(b: Builder) -> list[tuple[int, int]]:
    return [(i, j) for i, nbrs in enumerate(b.bonds) for j in nbrs.values() if i < j]


def _assert_ideal(b: Builder) -> None:
    """Check every bond length and every angle at every atom."""
    geo = b.geometry()
    for i, j in _bonded_pairs(b):
        has_h = "H" in (b.symbols[i], b.symbols[j])
        expected = XH_LENGTH if has_h else XY_LENGTH
        assert np.isclose(geom.distance(geo, [i, j]), expected)
    for j, nbrs in enumerate(b.bonds):
        for i, k in itertools.combinations(nbrs.values(), 2):
            assert np.isclose(geom.angle(geo, [i, j, k]), TETRAHEDRAL_ANGLE)


def test__ideal_bonds_and_angles() -> None:
    """Test that all bonds and angles are ideal."""
    b = _neopentane_amine()
    assert b.geometry().symbols.count("H") == 9 + 2
    _assert_ideal(b)


def test__first_bonds_follow_up_and_left() -> None:
    """Test the C-N bond points up and a C sits in-plane to the left."""
    b = Builder(up=(0, 0, 1), left=(-1, 0, 0))
    c = b.add("C")
    n = b.add("N", c, toward=(0, 0, 1))
    c2 = b.add("C", c, toward=(-1, 0, 0))
    geo = b.geometry()
    assert np.allclose(geo.coordinates[n], [0, 0, XY_LENGTH])
    assert np.isclose(geo.coordinates[c2][1], 0.0)  # in the xz plane
    assert geo.coordinates[c2][0] < 0  # to the left
    assert np.isclose(geom.angle(geo, [n, c, c2]), TETRAHEDRAL_ANGLE)


def test__staggered() -> None:
    """Test that ethane comes out staggered (H-C-C-H dihedrals of 60 or 180)."""
    b = Builder()
    c1 = b.add("C")
    c2 = b.add("C", c1)
    b.add_hydrogens()
    geo = b.geometry()
    h1s = [k for k in b.bonds[c1].values() if k != c2]
    h2s = [k for k in b.bonds[c2].values() if k != c1]
    for h1, h2 in itertools.product(h1s, h2s):
        phi = abs(geom.dihedral(geo, [h1, c1, c2, h2], degrees=True))
        assert np.isclose(phi, 60.0) or np.isclose(phi, 180.0)


def test__diamond_lattice() -> None:
    """Test that filling in carbons gives the diamond lattice, without clashes."""
    b = Builder()
    b.add("C")
    shell = [0]
    for _ in range(3):
        new_shell = []
        for key, slot in [(k, s) for k in shell for s in b.open_slots(k)]:
            try:
                new_shell.append(b.add("C", key, slot))
            except ValueError as err:  # a 6-ring closes onto an existing atom
                assert "occupied" in str(err)  # noqa: PT017
        shell = new_shell
    # Diamond has 4, 12, and 24 atoms at 1, 2, and 3 bonds from any atom
    assert b.atom_count == 1 + 4 + 12 + 24
    dmat = geom.distance_matrix(b.geometry())
    dists = np.unique(dmat[np.triu_indices(b.atom_count, k=1)].round(6))
    # Nearest and next-nearest neighbor distances of diamond
    assert np.isclose(dists[0], XY_LENGTH)
    assert np.isclose(dists[1], XY_LENGTH * np.sqrt(8 / 3))
    _assert_ideal(b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test__grid_rotates_with_basis(seed: int) -> None:
    """Test that rotating `up` and `left` rotates the whole molecule."""
    rot = Rotation.random(random_state=seed)
    ref = _neopentane_amine().geometry()
    up, left = rot.apply([[0, 0, 1], [-1, 0, 0]])
    rotated = _neopentane_amine(up=up, left=left).geometry()
    assert np.allclose(geom.rotate(ref, rot).coordinates, rotated.coordinates)
    assert geom.alignment_rotation(rotated, ref).approx_equal(rot, atol=1e-8)


def test__ring_closure_site_is_occupied() -> None:
    """Test that adding onto an occupied site raises (a 6-ring closes there)."""
    b = Builder()
    ring = [b.add("C")]
    # Bonds along slots 0, 1, 2, 0, 1, 2 go around a chair cyclohexane ring
    # (bond vectors +v0, -v1, +v2, -v0, +v1, -v2 sum to zero)
    for slot in [0, 1, 2, 0, 1]:
        ring.append(b.add("C", ring[-1], slot))
    with pytest.raises(ValueError, match="occupied by atom 0"):
        b.add("C", ring[-1], 2)
    geo = b.geometry()
    assert np.isclose(geom.distance(geo, [ring[0], ring[-1]]), XY_LENGTH)


@pytest.mark.parametrize(("symbol", "nh"), [("C", 4), ("N", 3), ("O", 2), ("F", 1)])
def test__add_hydrogens_counts(symbol: str, nh: int) -> None:
    """Test that atoms are filled up to their valence."""
    b = Builder()
    b.add(symbol)
    b.add_hydrogens()
    assert b.geometry().symbols.count("H") == nh


def test__add_hydrogens_avoid() -> None:
    """Test that the slot nearest `avoid` is left empty (e.g. a lone pair)."""
    b = Builder()
    n = b.add("N")
    b.add_hydrogens(avoid=(0, 0, 1))
    assert b.open_slots(n) == [0]  # slot 0 points up


def test__geometry_spin() -> None:
    """Test default spin for open- and closed-shell molecules."""
    b = Builder()
    c = b.add("C")
    for _ in range(3):
        b.add("H", c)
    assert b.geometry().spin == 1
    b.add("H", c)
    assert b.geometry().spin == 0
    assert isinstance(b.geometry(), Geometry)


def test__add_errors() -> None:
    """Test invalid additions."""
    b = Builder()
    c = b.add("C")
    with pytest.raises(ValueError, match="first atom"):
        b.add("C")
    with pytest.raises(ValueError, match="No atom"):
        b.add("C", 5)
    with pytest.raises(ValueError, match="not both"):
        b.add("C", c, 0, toward=(0, 0, 1))
    b.add("C", c, 0)
    with pytest.raises(ValueError, match="not empty"):
        b.add("C", c, 0)
    b.add_hydrogens([c])
    with pytest.raises(ValueError, match="no empty slots"):
        b.add("C", c)


def _methylamine() -> tuple[Builder, int, int]:
    """CH3-NH2, with the N lone pair in slot 3."""
    b = Builder()
    c = b.add("C")
    n = b.add("N", c, 0)
    b.add("H", n, 1)
    b.add("H", n, 2)
    b.add_hydrogens([c])
    return b, c, n


def test__umbrella_image() -> None:
    """Test that the image is the inverted amine, and that its image is the original."""
    b, c, n = _methylamine()
    hs = [b.bonds[n][1], b.bonds[n][2]]
    geo, images = b.umbrella_image(n)
    assert geo.atom_count == b.atom_count + 2
    assert [geo.symbols[k] for k in images] == ["H", "H"]
    # The image bonds are ideal, and the images are new positions
    for h in images:
        assert np.isclose(geom.distance(geo, [n, h]), XH_LENGTH)
        assert np.isclose(geom.angle(geo, [c, n, h]), TETRAHEDRAL_ANGLE)
        assert min(geom.distance(geo, [h, k]) for k in hs) > 0.5
    assert np.isclose(geom.angle(geo, [*images[:1], n, images[1]]), TETRAHEDRAL_ANGLE)
    # Mirroring the images again gives back the original H's
    geo2, images2 = b.umbrella_image(n, images, geo=geo)
    assert np.allclose(geo2.coordinates[images2], geo.coordinates[hs])


def test__umbrella_image_default_branch() -> None:
    """Test that the whole branch beyond the center is mirrored by default."""
    b = Builder()
    c = b.add("C")
    n = b.add("N", c, 0)
    c2 = b.add("C", n, 1)
    b.add("H", n, 2)
    b.add_hydrogens([c, c2])
    _, images = b.umbrella_image(n)
    assert len(images) == 1 + 1 + 3  # the methyl C, the N-H, and the methyl H's


def test__umbrella_image_errors() -> None:
    """Test that a root atom or an ambiguous lone pair is rejected."""
    b = Builder()
    c = b.add("C")
    n = b.add("N", c, 0)
    with pytest.raises(ValueError, match="no parent"):
        b.umbrella_image(c)
    with pytest.raises(ValueError, match="exactly one"):
        b.umbrella_image(n)
    _, images = b.umbrella_image(n, [], normal=(1, 0, 0))
    assert images == []
