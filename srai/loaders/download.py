"""Utility download function."""

import warnings
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm

from srai.constants import FORCE_TERMINAL


def download_file(
    url: str, fname: Union[str, Path], chunk_size: int = 1024, force_download: bool = True
) -> None:
    """
    Download a file with progress bar.

    Args:
        url (str): URL to download.
        fname (Union[str, Path]): File name.
        chunk_size (int): Chunk size.
        force_download (bool): Flag to force download even if file exists.

    Source: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    """
    if Path(fname).exists() and not force_download:
        warnings.warn("File exists. Skipping download.", stacklevel=1)
        return

    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        url,
        headers={"User-Agent": "SRAI Python package (https://github.com/kraina-ai/srai)"},
        stream=True,
    )
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as file,
        tqdm(
            desc=Path(fname).name,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            disable=FORCE_TERMINAL,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
