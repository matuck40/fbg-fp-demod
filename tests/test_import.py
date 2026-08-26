"""Smoke test: the package is importable and carries a version."""

import fbgfp


def test_package_exposes_a_version():
    assert fbgfp.__version__
