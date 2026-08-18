"""Tests for paper figure reproduction scripts.

These tests run each figure script against the sample dataset with a
non-interactive matplotlib backend.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

FIGURE_SCRIPTS = Path(__file__).parent.parent / "paper" / "figures" / "scripts"


@pytest.mark.parametrize(
    "script",
    [
        "fig_restitution_map.py",
        "fig_profiles_approach_spread.py",
        "fig_energy_budget.py",
        "fig_time_series.py",
        "fig_central_gap_scaling.py",
        "fig_exponent_summary.py",
        "fig_gap_velocity_field.py",
    ],
)
def test_figure_script_runs(script: str) -> None:
    """Run a figure script with MPLBACKEND=Agg and ensure it exits cleanly."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, str(FIGURE_SCRIPTS / script)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
