"""Utility functions for loaders examples."""
import requests
from tqdm import tqdm


def download(url: str, fname: str, chunk_size: int = 1024) -> None:
    """
    Download a file with progress bar.

    Args:
        url (str): URL to download.
        fname (str): File name.
        chunk_size (str): Chunk size.

    Source: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname.split("/")[-1],
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
