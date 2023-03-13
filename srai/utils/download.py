"""Utility functions for loaders examples."""
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, filename: Path, chunk_size: int = 1024) -> None:
    """
    Download a file with progress bar.

    Args:
        url (str): URL to download.
        filename (Path): File to save data to.
        chunk_size (str): Chunk size.

    Source: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("wb") as file, tqdm(
        desc=filename.name,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
