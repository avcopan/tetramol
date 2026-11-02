"""Drawing tests: rendering and the browser orientation round trip."""

import json
import threading
import urllib.request
import webbrowser
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tetramol import Builder, draw, orient, render, render_spins


def _methylamine() -> Builder:
    b = Builder()
    c = b.add("C")
    b.add("N", c, toward=(0, 0, 1))
    b.add_hydrogens()
    return b


def test__render(tmp_path: Path) -> None:
    """Test rendering an SVG file."""
    render(_methylamine().geometry(), tmp_path / "methylamine.xyz")
    svg = (tmp_path / "methylamine.svg").read_text()
    assert svg.lstrip().startswith(("<svg", "<?xml"))


def test__orient_page() -> None:
    """Test that the page wires up the py3Dmol viewer it embeds."""
    page = draw.orient_page(_methylamine().geometry())
    viewer = page.split("const viewer = ")[1].split(";")[0]
    assert f"var {viewer} = null;" in page
    assert "setViewChangeCallback" in page
    assert "/done" in page


def test__serve_orient_page_round_trip() -> None:
    """Test that the served page posts the view quaternion back to Python."""
    server, result = draw.serve_orient_page(_methylamine().geometry())
    url = f"http://localhost:{server.server_address[1]}"
    with urllib.request.urlopen(url) as resp:
        assert b"Done" in resp.read()

    quat = Rotation.random(random_state=0).as_quat().tolist()
    req = urllib.request.Request(f"{url}/done", data=json.dumps(quat).encode())  # noqa: S310
    with urllib.request.urlopen(req):  # noqa: S310
        pass
    server.server_close()
    assert np.allclose(result[0], quat)


def test__orient_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `orient`, with a stand-in for the browser that clicks "Done"."""
    rot = Rotation.random(random_state=1)

    def browser(url: str) -> None:
        with urllib.request.urlopen(url):  # noqa: S310
            pass
        data = json.dumps(rot.as_quat().tolist()).encode()
        req = urllib.request.Request(f"{url}/done", data=data)  # noqa: S310
        threading.Thread(target=urllib.request.urlopen, args=(req,)).start()

    monkeypatch.setattr(webbrowser, "open", browser)
    found = orient(_methylamine().geometry())
    assert found.approx_equal(rot)


def test__orient_page_spin() -> None:
    """Test that the spin page has a slider for the (normalized) axis."""
    page = draw.orient_page(_methylamine().geometry(), axis=(0, 2, 0))
    assert 'id="angle"' in page
    assert "const axis = [0.0, 1.0, 0.0];" in page


@pytest.mark.parametrize(
    ("axis", "posted", "angle"),
    [
        ((0, 1, 0), Rotation.from_euler("y", 40, degrees=True), 40),
        ((0, 1, 0), Rotation.from_euler("y", -30, degrees=True), 330),
        ((1, 1, 1), Rotation.from_rotvec(np.radians(75) * np.ones(3) / np.sqrt(3)), 75),
    ],
)
def test__orient_spin_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    axis: tuple[float, float, float],
    posted: Rotation,
    angle: float,
) -> None:
    """Test that `orient` with an axis returns the spin about that axis."""

    def browser(url: str) -> None:
        data = json.dumps(posted.as_quat().tolist()).encode()
        req = urllib.request.Request(f"{url}/done", data=data)  # noqa: S310
        threading.Thread(target=urllib.request.urlopen, args=(req,)).start()

    monkeypatch.setattr(webbrowser, "open", browser)
    found = orient(_methylamine().geometry(), axis=axis)
    unit = np.array(axis) / np.linalg.norm(axis)
    assert found.approx_equal(Rotation.from_rotvec(np.radians(angle) * unit))


def test__render_spins(tmp_path: Path) -> None:
    """Test that the contact sheet has one labeled panel per angle."""
    angles = [0, 30, 60, 90, 120]
    render_spins(_methylamine().geometry(), tmp_path / "spins", angles=angles)
    svg = (tmp_path / "spins.svg").read_text()
    assert svg.count("<svg") == len(angles) + 1
    for angle in angles:
        assert f">{angle}°</text>" in svg
