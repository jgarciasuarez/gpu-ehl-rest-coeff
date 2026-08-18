#!/usr/bin/env python3
"""Extract a small sample from parametric_study_final.h5 for CI and demos.

Selection rule:
- For exponent n = 2.0, keep Stokes = 10, 100, 1000, 10000.
- For all other shape exponents, keep only Stokes = 100.0.

The sample is written to ``data/sample/parametric_sample.h5``.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import h5py

# Allow running this script before the package is installed.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from gpu_ehl.sweep import make_parameter_grid  # noqa: E402

logger = logging.getLogger(__name__)

# Sample Stokes numbers retained for the reference shape (n = 2.0).
SAMPLE_STOKES = [10.0, 100.0, 1000.0, 10000.0]

# Shape labels as they appear in the HDF5 file.  The first physical exponent is
# n = 1.05, but the stored group is ``Shape_1.0``.
SHAPE_LABELS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def extract_sample(
    source: Path,
    destination: Path,
    reference_shape: float = 2.0,
    reference_stokes: float = 100.0,
) -> None:
    """Extract selected cases from the full dataset.

    Parameters
    ----------
    source : Path
        Path to the full ``parametric_study_final.h5`` file.
    destination : Path
        Path for the output sample file.
    reference_shape : float, optional
        Shape exponent for which the sample Stokes values are retained
        (default: 2.0).
    reference_stokes : float, optional
        Stokes value retained for all non-reference shapes (default: 100.0).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    S_flat, _ = make_parameter_grid()
    S_flat = sorted({float(s) for s in S_flat})

    # Sanity-check that the requested sample Stokes exist in the full grid.
    missing_stokes = [
        s for s in SAMPLE_STOKES
        if not any(abs(s - ss) < 1e-6 for ss in S_flat)
    ]
    if missing_stokes:
        logger.warning("Requested sample Stokes not in full grid: %s", missing_stokes)

    selected_cases = []
    for shape_label in SHAPE_LABELS:
        if abs(shape_label - reference_shape) < 1e-8:
            for s_val in SAMPLE_STOKES:
                selected_cases.append((s_val, shape_label))
        else:
            selected_cases.append((reference_stokes, shape_label))

    logger.info(
        "Extracting %d cases from %s to %s",
        len(selected_cases),
        source,
        destination,
    )

    with h5py.File(source, "r") as src, h5py.File(destination, "w") as dst:
        for key in ("r_grid", "time_grid", "snapshot_indices"):
            if key in src:
                dst.create_dataset(key, data=src[key][:])

        for s_val, e_val in selected_cases:
            group_name = f"Stokes_{s_val:.1f}/Shape_{e_val:.1f}"
            if group_name not in src:
                logger.warning("Group %s not found in source; skipping", group_name)
                continue

            src_grp = src[group_name]
            dst_grp = dst.create_group(group_name)
            for dset_name in src_grp:
                dst_grp.create_dataset(
                    dset_name,
                    data=src_grp[dset_name][:],
                    compression="gzip",
                    compression_opts=4,
                )
            logger.debug("Copied %s", group_name)

    logger.info("Sample extraction complete")


def _default_source_path() -> Path:
    """Return the default full-dataset path.

    Resolution order:
    1. ``GPU_EHL_DATA`` environment variable.
    2. ``data/parametric_study_final.h5`` relative to the repository root.
    """
    env_path = os.environ.get("GPU_EHL_DATA")
    if env_path:
        return Path(env_path)

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "parametric_study_final.h5"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a small sample from the full parametric dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_default_source_path(),
        help="Path to parametric_study_final.h5",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).parent / "sample" / "parametric_sample.h5",
        help="Output sample path",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    extract_sample(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
