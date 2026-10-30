"""Grid tests."""

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tetramol import TETRAHEDRAL_ANGLE, tetrahedral_basis


@pytest.mark.parametrize("seed", [0, 1, 2])
def test__tetrahedral_basis(seed: int) -> None:
    """Test unit length, tetrahedral angles, and the roles of `up` and `left`."""
    rng = np.random.default_rng(seed)
    up, left = rng.normal(size=3), rng.normal(size=3)
    basis = tetrahedral_basis(up, left)
    assert np.allclose(np.linalg.norm(basis, axis=1), 1.0)
    for v1, v2 in itertools.combinations(basis, 2):
        assert np.isclose(np.arccos(np.dot(v1, v2)), TETRAHEDRAL_ANGLE)
    assert np.allclose(basis[0], up / np.linalg.norm(up))
    # Direction 1 is in the up-left plane, on the side of `left`
    normal = np.cross(up, left)
    assert np.isclose(np.dot(basis[1], normal), 0.0)
    assert np.dot(basis[1], left - np.dot(left, basis[0]) * basis[0]) > 0
    # Directions 2 and 3 are direction 1 rotated by +/-120 degrees about `up`
    rot = Rotation.from_rotvec(2 * np.pi / 3 * basis[0])
    assert np.allclose(rot.apply(basis[1]), basis[2])
    assert np.allclose(rot.apply(basis[2]), basis[3])
    assert np.allclose(basis.sum(axis=0), 0.0)


@pytest.mark.parametrize(
    ("up", "left"), [((0, 0, 0), (1, 0, 0)), ((0, 0, 1), (0, 0, 2))]
)
def test__tetrahedral_basis_degenerate_raises(up: tuple, left: tuple) -> None:
    """Test that a zero `up` or a `left` parallel to it is rejected."""
    with pytest.raises(ValueError, match="non-zero"):
        tetrahedral_basis(up, left)
