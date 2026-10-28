"""Tests for the version-2 dataset pipeline (merge + per-case grid override).

Builds small synthetic HDF5 files in the ``save_results_h5`` layout and
exercises ``scripts/merge_fine_subset.py`` and the readers that must honor
group-level grid overrides.
"""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_SNAPSHOTS = 10
CASE_DATASETS = (
    "Force",
    "Central_h",
    "Min_h",
    "V",
    "w_snapshots",
    "p_snapshots",
    "D_snapshots",
    "E_elastic",
)

# Distinct per-dataset offsets so the merged file can be checked against
# its true source (coarse or fine) case by case.
DATASET_OFFSETS = {name: float(i) for i, name in enumerate(CASE_DATASETS)}


def _load_module(relative_path: str):
    spec = importlib.util.spec_from_file_location(
        Path(relative_path).stem, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def merge_mod():
    return _load_module("scripts/merge_fine_subset.py")


@pytest.fixture(scope="module")
def convergence_mod():
    return _load_module("scripts/check_fine_coarse_convergence.py")


def write_sweep_h5(
    path: Path,
    nr: int,
    nt: int,
    delta_t: float,
    stokes_values: list[float],
    exponents: list[float],
    seed: int = 0,
    random_data: bool = False,
) -> None:
    """Write a synthetic sweep file in the ``save_results_h5`` layout.

    With ``random_data=False`` every dataset is constant and equal to its
    case/dataset marker, so the merged file can be checked verbatim against
    its source. With ``random_data=True`` unit-scale noise is added.
    """
    rng = np.random.default_rng(seed)
    r = np.linspace(0.0, 6.0, nr)
    time_grid = np.linspace(0.0, nt * delta_t, nt + 1)
    snapshot_idx = np.linspace(0, nt - 1, NUM_SNAPSHOTS, dtype=int)

    with h5py.File(path, "w") as f:
        f.create_dataset("r_grid", data=r)
        f.create_dataset("time_grid", data=time_grid)
        f.create_dataset("snapshot_indices", data=snapshot_idx)
        for stokes in stokes_values:
            for exponent in exponents:
                grp = f.create_group(f"Stokes_{stokes:.1f}/Shape_{exponent:.1f}")
                for name in CASE_DATASETS:
                    marker = stokes * 1000.0 + exponent + DATASET_OFFSETS[name]
                    if name == "D_snapshots":
                        shape = (NUM_SNAPSHOTS,)
                    elif name.endswith("_snapshots"):
                        shape = (NUM_SNAPSHOTS, nr)
                    else:
                        shape = (NUM_SNAPSHOTS,)
                    data = np.full(shape, marker)
                    if random_data:
                        data = data + rng.random(shape)
                    grp.create_dataset(name, data=data)


def _expected_marker(
    stokes: float, exponent: float, dataset: str, source_nr: int
) -> np.ndarray:
    marker = stokes * 1000.0 + exponent + DATASET_OFFSETS[dataset]
    if dataset == "D_snapshots":
        shape = (NUM_SNAPSHOTS,)
    elif dataset.endswith("_snapshots"):
        shape = (NUM_SNAPSHOTS, source_nr)
    else:
        shape = (NUM_SNAPSHOTS,)
    return np.full(shape, marker)


def test_merge_replaces_subset_and_keeps_rest(merge_mod, tmp_path):
    """Fine groups are replaced (with own grids); coarse groups are verbatim."""
    coarse_stokes = [100.0, 1000.0, 3162.3, 10000.0]
    exponents = [2.0, 6.0]
    coarse_path = tmp_path / "coarse.h5"
    fine_path = tmp_path / "fine.h5"
    out_path = tmp_path / "merged.h5"

    write_sweep_h5(
        coarse_path,
        nr=32,
        nt=100,
        delta_t=5.0e-4,
        stokes_values=coarse_stokes,
        exponents=exponents,
        seed=1,
    )
    write_sweep_h5(
        fine_path,
        nr=64,
        nt=200,
        delta_t=2.5e-4,
        stokes_values=[3162.3, 10000.0],
        exponents=exponents,
        seed=2,
    )
    merged = merge_mod.merge_datasets(coarse_path, fine_path, out_path)
    assert merged == sorted(
        f"Stokes_{stokes:.1f}/Shape_{exponent:.1f}"
        for stokes in (3162.3, 10000.0)
        for exponent in exponents
    )

    with h5py.File(out_path, "r") as f:
        # Root-level grids describe the coarse reference resolution.
        assert len(f["r_grid"]) == 32
        assert f.attrs["version"] == 2
        assert f.attrs["coarse_nr"] == 32
        assert f.attrs["fine_nr"] == 64
        assert np.isclose(f.attrs["coarse_delta_t"], 5.0e-4)
        assert np.isclose(f.attrs["fine_delta_t"], 2.5e-4)

        for stokes in (100.0, 1000.0):  # coarse-only cases
            for exponent in exponents:
                grp = f[f"Stokes_{stokes:.1f}/Shape_{exponent:.1f}"]
                assert grp.attrs["resolution"] == "coarse"
                assert grp.attrs["nr"] == 32
                assert "r_grid" not in grp
                for name in CASE_DATASETS:
                    expected = _expected_marker(stokes, exponent, name, 32)
                    np.testing.assert_allclose(grp[name][...], expected)

        for stokes in (3162.3, 10000.0):  # fine cases
            for exponent in exponents:
                grp = f[f"Stokes_{stokes:.1f}/Shape_{exponent:.1f}"]
                assert grp.attrs["resolution"] == "fine"
                assert grp.attrs["nr"] == 64
                assert np.isclose(grp.attrs["delta_t"], 2.5e-4)
                assert len(grp["r_grid"]) == 64
                assert len(grp["time_grid"]) == 201
                for name in CASE_DATASETS:
                    expected = _expected_marker(stokes, exponent, name, 64)
                    np.testing.assert_allclose(grp[name][...], expected)


def test_merge_refuses_to_overwrite(merge_mod, tmp_path):
    out_path = tmp_path / "merged.h5"
    write_sweep_h5(tmp_path / "coarse.h5", 32, 100, 5.0e-4, [100.0], [2.0])
    write_sweep_h5(tmp_path / "fine.h5", 64, 200, 2.5e-4, [100.0], [2.0])
    out_path.write_text("")
    with pytest.raises(FileExistsError):
        merge_mod.merge_datasets(tmp_path / "coarse.h5", tmp_path / "fine.h5", out_path)


def test_merge_verifies_output(merge_mod, tmp_path):
    """verify_merged_file passes on a valid merge and detects tampering."""
    coarse_path = tmp_path / "coarse.h5"
    fine_path = tmp_path / "fine.h5"
    out_path = tmp_path / "merged.h5"
    write_sweep_h5(coarse_path, 32, 100, 5.0e-4, [100.0, 3162.3], [2.0])
    write_sweep_h5(fine_path, 64, 200, 2.5e-4, [3162.3], [2.0])

    merged = merge_mod.merge_datasets(coarse_path, fine_path, out_path)
    merge_mod.verify_merged_file(out_path, expected_fine=len(merged))

    with h5py.File(out_path, "r+") as f:
        # Tamper: fine group loses its grid override.
        del f["Stokes_3162.3/Shape_2.0/r_grid"]
    with pytest.raises(AssertionError):
        merge_mod.verify_merged_file(out_path, expected_fine=len(merged))


def test_load_results_h5_exposes_per_case_grids(tmp_path):
    """``load_results_h5`` surfaces group-level grids only for fine cases."""
    from gpu_ehl.io import load_results_h5

    coarse_path = tmp_path / "coarse.h5"
    fine_path = tmp_path / "fine.h5"
    out_path = tmp_path / "merged.h5"
    write_sweep_h5(coarse_path, 32, 100, 5.0e-4, [100.0, 3162.3], [2.0])
    write_sweep_h5(fine_path, 64, 200, 2.5e-4, [3162.3], [2.0])

    merge_spec = importlib.util.spec_from_file_location(
        "merge_fine_subset", REPO_ROOT / "scripts" / "merge_fine_subset.py"
    )
    merge_mod = importlib.util.module_from_spec(merge_spec)
    merge_spec.loader.exec_module(merge_mod)
    merge_mod.merge_datasets(coarse_path, fine_path, out_path)

    data = load_results_h5(out_path)
    assert len(data["r_grid"]) == 32

    coarse_case = data["cases"]["Stokes_100.0"]["Shape_2.0"]
    assert "r_grid" not in coarse_case

    fine_case = data["cases"]["Stokes_3162.3"]["Shape_2.0"]
    assert "r_grid" in fine_case
    assert len(fine_case["r_grid"]) == 64
    assert fine_case["w_snapshots"].shape == (NUM_SNAPSHOTS, 64)


def test_convergence_check_runs_on_synthetic_v2(convergence_mod, tmp_path):
    """The convergence pipeline runs end-to-end on synthetic coarse+fine."""
    coarse_path = tmp_path / "coarse.h5"
    fine_path = tmp_path / "fine.h5"
    csv_path = tmp_path / "convergence.csv"

    write_sweep_h5(
        coarse_path, 64, 400, 5.0e-4, [3162.3, 10000.0], [2.0], seed=3, random_data=True
    )
    write_sweep_h5(
        fine_path, 128, 800, 2.5e-4, [3162.3, 10000.0], [2.0], seed=4, random_data=True
    )

    coarse = convergence_mod.collect_metrics(coarse_path)
    fine = convergence_mod.collect_metrics(fine_path)
    rows = convergence_mod.compare(coarse, fine, csv_path)

    assert len(rows) == 2
    assert csv_path.exists()
    convergence_mod.print_summary(rows)

    # Synthetic data differ between files, so relative diffs are positive
    # and finite for every metric.
    for row in rows:
        for key in ("e", "t_seal", "h_min", "r_n", "p_n", "lambda"):
            assert np.isfinite(row[f"{key}_rel_diff"])
