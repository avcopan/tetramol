"""Motion-arrow tests: the page fit, exact projections, heads, and round trips."""

import re
from pathlib import Path

import numpy as np
import pytest
from automol import Geometry, geom
from scipy.spatial.transform import Rotation

from tetramol import (
    ARROW_STYLE,
    Builder,
    ExternalRotation,
    Torsion,
    Umbrella,
    add_arrows,
    render,
    render_png,
    torsions,
    umbrella_edge_on,
    umbrella_image,
)
from tetramol.arrows import _Circle, _Drawing  # internals, to test the geometry

ARROW = re.compile(r'<path\b[^>]*id="(?:torsion|umbrella|rotation)[^"]*"[^>]*/>')


def _tham() -> tuple[Geometry, int, int, list[int]]:
    """THAM drawn with its NH2 umbrella images, as in `drawings/tham/ideal.py`."""
    b = Builder(up=(0, 1, 0), left=(-1, 0, 0))
    c = b.add("C")
    n = b.add("N", c, 0)
    for slot in b.open_slots(c):
        ch2 = b.add("C", c, slot)
        o = b.add("O", ch2, toward=b.tangent(ch2, c))
        b.add("H", o, toward=b.tangent(o, c))
    b.add_hydrogens([n], avoid=(0, 0, -1))
    b.add_hydrogens()
    geo = geom.rotate(b.geometry(), Rotation.from_euler("yx", [-65, 30], degrees=True))
    geo = umbrella_edge_on(geo, n)
    umbrella_geo, images = umbrella_image(geo, n)
    return umbrella_geo, c, n, images


def _svg(geo: Geometry, a_mat: np.ndarray, shift: np.ndarray) -> str:
    """A minimal drawing: each atom a circle and each bond a line, back to front."""
    page = geo.coordinates[:, :2] @ a_mat.T + shift
    scale = np.sqrt(abs(np.linalg.det(a_mat)))
    radius = {"H": 0.23, "C": 0.43, "N": 0.40, "O": 0.39}
    amat = geom.adjacency_matrix(geo)
    parts = []
    for k in np.argsort(geo.coordinates[:, 2]):
        x, y = page[k]
        r = radius[geo.symbols[k]] * scale
        parts.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{r:.4f}" fill="#ffffff"/>')
        for m in np.flatnonzero(amat[k]):
            if geo.coordinates[m, 2] < geo.coordinates[k, 2]:
                (x1, y1), (x2, y2) = page[k], page[m]
                parts.append(
                    f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" '
                    f'stroke="#000000" stroke-width="{0.2 * scale:.4f}"/>'
                )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-500 -500 1000 1000">\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )


def _similarity(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-0.3, 0.3)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rng.uniform(5, 200) * rot @ np.diag([1.0, -1.0]), rng.uniform(-50, 50, 2)


@pytest.mark.parametrize("seed", range(3))
def test__fit_recovers_the_page_map(seed: int) -> None:
    """Test that the 3D -> page map is recovered from the circles alone."""
    geo, *_ = _tham()
    a_mat, shift = _similarity(seed)
    drawing = _Drawing(_svg(geo, a_mat, shift), geo, ARROW_STYLE)
    assert np.allclose(drawing.a_mat, a_mat, atol=1e-6 * np.abs(a_mat).max())
    assert np.allclose(drawing.shift, shift, atol=1e-3)


def test__wrong_geometry_raises() -> None:
    """Test that a geometry that wasn't drawn is refused."""
    geo, *_ = _tham()
    svg = _svg(geo, 50 * np.diag([1.0, -1.0]), np.zeros(2))
    turned = geom.rotate(geo, Rotation.from_euler("y", 40, degrees=True))
    with pytest.raises(ValueError, match="doesn't match"):
        add_arrows(svg, turned, torsions(turned))


@pytest.mark.parametrize("seed", range(5))
def test__circle_projects_to_its_ellipse(seed: int) -> None:
    """Test that a circle normal to u projects to semi-axes r and r |u_z|.

    The minor axis is along the projected u.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    circle = _Circle.about(rng.normal(size=3), u, 0.7)
    rel = circle.at(np.linspace(0, 2 * np.pi, 20001))[:, :2] - circle.center[:2]
    dist = np.linalg.norm(rel, axis=1)
    assert np.isclose(dist.max(), 0.7, atol=1e-6)
    assert np.isclose(dist.min(), 0.7 * abs(u[2]), atol=1e-6)
    minor = rel[np.argmin(dist)]
    assert (
        abs(minor[0] * u[1] - minor[1] * u[0]) < 1e-3
    )  # parallel to the projected axis


def test__add_arrows_only_inserts() -> None:
    """Test that removing the arrows gives back the SVG.

    The namespace and viewBox changes are confined to the root.
    """
    geo, c, n, images = _tham()
    svg = str(render(geo))
    out = str(
        add_arrows(
            svg, geo, [*torsions(geo), Umbrella(n, images), ExternalRotation(c, n)]
        )
    )
    strip = lambda text: re.sub(r"<svg\b[^>]*>", "<svg>", ARROW.sub("", text))  # noqa: E731
    assert strip(out) == strip(svg)
    assert str(add_arrows(svg, geo, [])) == svg


def test__heads_sit_on_their_arcs() -> None:
    """Test that each head's tip is the arc's end and its base is a head length back."""
    geo, *_ = _tham()
    drawing = _Drawing(
        _svg(geo, 40 * np.diag([1.0, -1.0]), np.zeros(2)), geo, ARROW_STYLE
    )
    circle = _Circle.about(geo.coordinates[0], np.array([0.0, 0.6, 0.8]), 0.4)
    tip, base, *_ = drawing._head_points(circle, 0.0, 2.0)  # noqa: SLF001
    assert np.allclose(tip, drawing.page(circle.at(2.0)))
    assert np.isclose(
        np.linalg.norm(tip - base), ARROW_STYLE.head.length * drawing.scale
    )
    # The base is on the same ellipse
    phi = drawing._back_along(circle, 0.0, 2.0, ARROW_STYLE.head.length * drawing.scale)  # noqa: SLF001
    assert np.allclose(base, drawing.page(circle.at(phi)))


def test__arrows_follow_the_page_map() -> None:
    """Test that arrows on a moved and scaled copy are moved and scaled the same."""
    geo, c, n, images = _tham()
    arrows = [*torsions(geo), Umbrella(n, images), ExternalRotation(c, n)]
    a1, t1 = 40 * np.diag([1.0, -1.0]), np.array([3.0, -2.0])
    a2, t2 = 2.5 * a1, np.array([-40.0, 17.0])
    heads = []
    for a_mat, shift in ((a1, t1), (a2, t2)):
        out = str(add_arrows(_svg(geo, a_mat, shift), geo, arrows))
        tips = [
            re.findall(r'd="M ([-\d.]+),([-\d.]+)', t)[0]
            for t in ARROW.findall(out)
            if "-head" in t
        ]
        heads.append(np.array(tips, dtype=float))
    # page2 = A2 A1^-1 (page1 - t1) + t2
    mapped = (heads[0] - t1) @ np.linalg.inv(a1).T @ a2.T + t2
    assert np.allclose(mapped, heads[1], atol=1e-4)


def test__umbrella_arrow() -> None:
    """Test that the umbrella arrow pairs each image with its original.

    It draws one arrow, with a head at each end.
    """
    geo, _, n, images = _tham()
    drawing = _Drawing(
        _svg(geo, 40 * np.diag([1.0, -1.0]), np.zeros(2)), geo, ARROW_STYLE
    )
    pairs = drawing._umbrella_pairs(n, list(images))  # noqa: SLF001
    assert sorted(i for _, i in pairs) == sorted(images)
    for h, image in pairs:  # mirror images: same distance from N, same depth
        assert np.isclose(
            np.linalg.norm(geo.coordinates[h] - geo.coordinates[n]),
            np.linalg.norm(geo.coordinates[image] - geo.coordinates[n]),
        )
    inserts, _ = drawing.umbrella(Umbrella(n, images))
    heads = [
        p
        for _, text in inserts
        for p in re.findall(r'id="umbrella[^"]*-(?:head|tail)" d="([^"]*)"', text)
    ]
    assert len(heads) == 2


def test__external_rotation_clears_whats_below() -> None:
    """Test that the rotation axis starts past everything drawn along it."""
    geo, c, n, images = _tham()
    out = str(
        add_arrows(
            _svg(geo, 40 * np.diag([1.0, -1.0]), np.zeros(2)),
            geo,
            [Umbrella(n, images), ExternalRotation(c, n)],
        )
    )
    (axis,) = re.findall(
        r'id="rotation-[^"]*-axis" d="M ([-\d.]+),([-\d.]+) L ([-\d.]+),([-\d.]+)"', out
    )
    _, y_near, _, y_far = (float(v) for v in axis)
    assert y_far < y_near  # the axis runs up the page, away from N
    tails = re.findall(r'id="umbrella-[^"]*-(?:head|tail)" d="([^"]*)"', out)
    top = min(float(y) for d in tails for y in re.findall(r",([-\d.]+)", d))
    assert y_near < top  # it starts above the umbrella arrow


def test__torsions_finds_every_rotor() -> None:
    """Test that THAM's rotors are its 3 C-O, 3 C-C, and 1 C-N bonds."""
    geo, c, n, _ = _tham()
    found = {(t.i, t.j) for t in torsions(geo)}
    assert len(found) == 7
    assert (c, n) in found
    assert all(geo.symbols[j] in "CNO" for _, j in found)
    assert all(Torsion(i, j).placement == "bond-front" for i, j in found)


def test__render_png(tmp_path: Path) -> None:
    """Test rasterizing a drawing with arrows, whole and cropped."""
    geo, c, n, _ = _tham()
    svg = add_arrows(render(geo), geo, [*torsions(geo), ExternalRotation(c, n)])
    png = render_png(svg, tmp_path / "tham.svg", width=200)
    assert png.startswith(b"\x89PNG")
    assert (tmp_path / "tham.png").read_bytes() == png
    assert render_png(svg, crop=(0, 0, 100, 100), width=50).startswith(b"\x89PNG")
