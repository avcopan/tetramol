"""tetramol tests."""

import tetramol


def test_stub() -> None:
    """Stub test to ensure the test suite runs."""
    print(tetramol.__version__)  # noqa: T201


def test__greet() -> None:
    """Test the greet function."""
    assert tetramol.greet("World") == "Hello, World!"


def test__greet_jim() -> None:
    """Test the greet_jim function."""
    assert tetramol.greet_jim() == "Hello, Jim!"
