"""The demo's headline numbers, asserted.

The README quotes the RMS figures the demo prints; this test runs the same
seeded scenario and fails if they regress, so the quoted numbers stay
backed by the suite rather than by a claim.
"""

import importlib.util
import pathlib


def _load_demo():
    path = pathlib.Path(__file__).resolve().parent.parent / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_headline_numbers_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # demo.png lands here, not in the repo
    demo = _load_demo()
    rms_pressure_bar, rms_temperature_c = demo.main()
    assert rms_pressure_bar < 0.10  # README: "under 0.1 bar RMS"
    assert rms_temperature_c < 0.60  # README: "~0.5 degC"
    assert (tmp_path / "demo.png").exists()
