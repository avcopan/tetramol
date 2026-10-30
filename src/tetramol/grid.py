"""The tetrahedral grid: ideal bond directions, angles, and lengths."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

TETRAHEDRAL_ANGLE = float(np.arccos(-1 / 3))
"""The ideal tetrahedral angle, arccos(-1/3) (about 109.47 degrees), in radians."""

XH_LENGTH = 1.1
"""Ideal bond length to hydrogen, in Angstroms."""

XY_LENGTH = 1.5
"""Ideal bond length between heavy atoms, in Angstroms."""


def tetrahedral_basis(
    up: ArrayLike = (0.0, 0.0, 1.0), left: ArrayLike = (-1.0, 0.0, 0.0)
) -> np.ndarray:
    """Get the four bond directions of a tetrahedral grid.

    Direction 0 points along `up`. Direction 1 lies in the plane of `up` and
    `left`, on the side of `left`. Directions 2 and 3 follow by rotating
    direction 1 about `up` by 120 and 240 degrees (counterclockwise, looking
    down from the tip of `up`).

    Args:
        up: Direction 0.
        left: Direction in the plane of directions 0 and 1. Only its component
            perpendicular to `up` matters.

    Returns:
        Unit vectors, one per row, shape ``(4, 3)``. Each pair makes the
        tetrahedral angle.

    Raises:
        ValueError: If `up` is zero or `left` is parallel to `up`.

    Example:
        >>> basis = tetrahedral_basis()
        >>> basis.round(6).tolist()[:2]
        [[0.0, 0.0, 1.0], [-0.942809, 0.0, -0.333333]]
    """
    up = _unit(up, "up")
    left = np.asarray(left, dtype=float)
    side = _unit(left - np.dot(left, up) * up, "left perpendicular to up")
    v1 = -up / 3 + 2 * np.sqrt(2) / 3 * side
    rot = Rotation.from_rotvec(2 * np.pi / 3 * up)
    return np.array([up, v1, rot.apply(v1), rot.inv().apply(v1)])


def _unit(vec: ArrayLike, name: str) -> np.ndarray:
    """Normalize a vector, raising `ValueError` if it is zero."""
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        msg = f"Vector {name} must be non-zero, got {vec}."
        raise ValueError(msg)
    return vec / norm
