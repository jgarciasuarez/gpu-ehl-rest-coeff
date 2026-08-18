#!/usr/bin/env python3
"""Download the full parametric dataset from Zenodo.

The default record corresponds to the dataset archived for
"Restitution of power-law elastic indenters in fluid-mediated impact":
    https://doi.org/10.5281/zenodo.21995894
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve

logger = logging.getLogger(__name__)

ZENODO_RECORD_ID: str = "21995894"
DATASET_FILENAME: str = "parametric_study_final.h5"


def _format_size(num_bytes: float) -> str:
    """Return a human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _resolve_download_url(record_id: str, filename: str) -> tuple[str, int]:
    """Query the Zenodo API and return the download URL and file size."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    logger.info("Fetching record metadata from %s", api_url)

    try:
        with urlopen(api_url, timeout=30) as response:
            metadata = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Zenodo API returned {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach Zenodo API: {exc.reason}") from exc

    for file_info in metadata.get("files", []):
        if file_info.get("key") == filename:
            links = file_info.get("links", {})
            download_url = links.get("download") or links.get("self")
            if not download_url:
                raise RuntimeError(
                    f"Zenodo file entry for {filename} has no download link"
                )
            return download_url, int(file_info.get("size", 0))

    available = [f.get("key") for f in metadata.get("files", [])]
    raise RuntimeError(
        f"File {filename!r} not found in Zenodo record {record_id}. "
        f"Available files: {available}"
    )


def download_dataset(destination: Path, record_id: str | None = None) -> None:
    """Download the full parametric dataset.

    Parameters
    ----------
    destination : Path
        Output path for the HDF5 file.
    record_id : str or None, optional
        Zenodo record ID. If ``None``, the global default is used.
    """
    record_id = record_id or ZENODO_RECORD_ID
    download_url, expected_size = _resolve_download_url(record_id, DATASET_FILENAME)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading %s (%s) from Zenodo record %s to %s",
        DATASET_FILENAME,
        _format_size(expected_size) if expected_size else "unknown size",
        record_id,
        destination,
    )

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100.0, 100.0 * downloaded / total_size)
        sys.stdout.write(
            f"\r  progress: {percent:5.1f}% ({_format_size(downloaded)} / "
            f"{_format_size(total_size)})"
        )
        sys.stdout.flush()

    try:
        urlretrieve(download_url, destination, reporthook=_reporthook)
    except HTTPError as exc:
        raise RuntimeError(f"Download failed with {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Download failed: {exc.reason}") from exc
    finally:
        sys.stdout.write("\n")

    if expected_size and destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded file size mismatch: expected {expected_size}, got "
            f"{destination.stat().st_size}"
        )

    logger.info("Dataset saved to %s (%s)", destination, _format_size(destination.stat().st_size))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the gpu-ehl parametric dataset from Zenodo."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).parent / DATASET_FILENAME,
        help="Output HDF5 path",
    )
    parser.add_argument(
        "--record-id",
        type=str,
        default=None,
        help="Zenodo record ID",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_dataset(args.destination, args.record_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
