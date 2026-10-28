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
        "scaling_neck_power_laws.py",
        "scaling_neck_balances.py",
        "scaling_central_gap_sealing.py",
        "scaling_backflow_drainage.py",
        "scaling_dissipation_neck_velocity.py",
        "scaling_profile_collapse.py",
        "scaling_retraction.py",
        "scaling_adhesion.py",
    ],
)
def test_figure_script_runs(script: str) -> None:
    """Run a figure script with MPLBACKEND=Agg and ensure it exits cleanly.

    The scripts are pinned to the committed sample dataset so the suite is
    fast and hermetic: style.resolve_h5_path would otherwise pick up the
    full ~300 MB dataset from data/ when it exists locally.
    """
    sample_h5 = (
        Path(__file__).parent.parent / "data" / "sample" / "parametric_sample.h5"
    )
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["GPU_EHL_DATA"] = str(sample_h5)
    result = subprocess.run(
        [sys.executable, str(FIGURE_SCRIPTS / script)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
