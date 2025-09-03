"""Utility download function."""

import warnings
from pathlib import Path

from pooch import retrieve


def download_file(url: str, fname: str, force_download: bool = True) -> None:
    """
    Download a file with progress bar.

    Args:
        url (str): URL to download.
        fname (str): File name.
        chunk_size (str): Chunk size.
        force_download (bool): Flag to force download even if file exists.

    Source: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    """
    destination_path = Path(fname)
    if destination_path.exists() and not force_download:
        warnings.warn("File exists. Skipping download.", stacklevel=1)
        return

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if force_download:
        destination_path.unlink(missing_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        retrieve(
            url,
            fname=destination_path.name,
            path=destination_path.parent,
            progressbar=True,
            known_hash=None,
        )
