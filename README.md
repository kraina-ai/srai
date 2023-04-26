<p align="center">
  <img width="300" src="https://raw.githubusercontent.com/srai-lab/srai/main/docs/assets/logos/srai-logo-transparent.png">
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/srai-lab/srai?logo=apache&logoColor=%23fff">
    <img src="https://img.shields.io/github/checks-status/srai-lab/srai/main?logo=GitHubActions&logoColor=%23fff" alt="Checks">
    <a href="https://github.com/srai-lab/srai/actions/workflows/ci-dev.yml" target="_blank">
        <img alt="GitHub Workflow Status - DEV" src="https://img.shields.io/github/actions/workflow/status/srai-lab/srai/ci-dev.yml?label=build-dev&logo=GitHubActions&logoColor=%23fff">
    </a>
    <a href="https://github.com/srai-lab/srai/actions/workflows/ci-prod.yml" target="_blank">
        <img alt="GitHub Workflow Status - PROD" src="https://img.shields.io/github/actions/workflow/status/srai-lab/srai/ci-prod.yml?label=build-prod&logo=GitHubActions&logoColor=%23fff">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/srai-lab/srai/main" target="_blank">
        <img src="https://results.pre-commit.ci/badge/github/srai-lab/srai/main.svg" alt="pre-commit.ci status">
    </a>
    <a href="https://www.codefactor.io/repository/github/srai-lab/srai"><img alt="CodeFactor Grade" src="https://img.shields.io/codefactor/grade/github/srai-lab/srai?logo=codefactor&logoColor=%23fff"></a>
    <a href="https://app.codecov.io/gh/srai-lab/srai/tree/main"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/srai-lab/srai?logo=codecov&token=PRS4E02ZX0&logoColor=%23fff"></a>
    <a href="https://pypi.org/project/srai" target="_blank">
        <img src="https://img.shields.io/pypi/v/srai?color=%2334D058&label=pypi%20package&logo=pypi&logoColor=%23fff" alt="Package version">
    </a>
    <a href="https://pypi.org/project/srai" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/srai.svg?color=%2334D058&logo=python&logoColor=%23fff" alt="Supported Python versions">
    </a>
    <a href="https://pypi.org/project/srai" target="_blank">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/srai">
    </a>
</p>

# Spatial Representations for Artificial Intelligence

Project **Spatial Representations for Artificial Intelligence** (`srai`) aims to provide simple and efficient solutions to geospatial problems that are accessible to everybody and reusable in various contexts where geospatial data can be used. It is a Python module integrating many geo-related algorithms in a single package with unified API. Please see getting starded for installation and quick srart instructions.


## Use cases

In the current state, `srai` provides the following functionalities:

* **OSM data download** - downloading OpenStreetMap data for a given area using different sources
* **OSM data processing** - processing OSM data to extract useful information (e.g. road network, buildings, POIs, etc.)
* **GTFS processing** - extracting features from GTFS data
* **Regionization** - splitting a given area into smaller regions using different algorithms (e.g. Uber's H3[1], Voronoi, etc.)
* **Embedding** - embedding regions into a vector space based on different spatial features, and using different algorithms (eg. hex2vec[2], etc.)
* Utilities for spatial data visualization and processing

For future releases, we plan to add more functionalities, such as:

* **Pre-computed embeddings** - pre-computed embeddings for different regions and different embedding algorithms
* **Full pipelines** - full pipelines for different embedding approaches, pre-configured from `srai` components
* **Image data download and processing** - downloading and processing image data (eg. OSM tiles, etc.)


## Installation

To install `srai` simply run:

```bash
pip install srai
```

This will install the `srai` package and dependencies required by most of the use cases. There are several optional dependencies that can be installed to enable additional functionality. These are listed in the [optional dependencies](#optional-dependencies) section.

### Optional dependencies

The following optional dependencies can be installed to enable additional functionality:

* `srai[all]` - all optional dependencies
* `srai[osm]` - dependencies required to download OpenStreetMap data
* `srai[voronoi]` - dependencies to use Voronoi-based regionization method
* `srai[gtfs]` - dependencies to process GTFS data
* `srai[plotting]` - dependencies to plot graphs and maps
* `srai[torch]` - dependencies to use torch-based embedders

## Usage

### Downloading OSM data

To download OSM data for a given area, using a set of tags use one of `OSMLoader` classes:

* `OSMOnlineLoader` - downloads data from OpenStreetMap API - this is faster for smaller areas or tags counts
* `OSMPbfLoader` - loads data from automatically downloaded PBF file - this is faster for larger areas or tags counts

Example with `OSMPbfLoader`:

```python
from srai.loaders import OSMOnlineLoader
from srai.utils import geocode_to_region_gdf

filter = {"leisure": "park"}
area = geocode_to_region_gdf("Wrocław, Poland")
loader = OSMOnlineLoader()

parks_gdf = loader.load(area, filter)
```

### Downloading road network

Road network downloading is a special case of OSM data downloading. To download road network for a given area, use `OSMWayLoader` class:

```python
from srai.loaders import OSMWayLoader
from srai.loaders.osm_way_loader import NetworkType
from srai.utils import geocode_to_region_gdf

area = geocode_to_region_gdf("Wrocław, Poland")
loader = OSMWayLoader(NetworkType.BIKE)

nodes, edges = loader.load(area)
```

### Downloading GTFS data

To extract features from GTFS use `GTFSLoader`. It will extract trip count and available directions for each stop in 1h time windows.

```python
from pathlib import Path

from srai.loaders import GTFSLoader

gtfs_path = Path("path/to/gtfs.zip")
loader = GTFSLoader()

features = loader.load(gtfs_path)
```

### Regionization

Regionization is a process of dividing a given area into smaller regions. This can be done in a variety of ways:

* `H3Regionizer` - regionization using [Uber's H3 library](https://github.com/uber/h3)
* `S2Regionizer` - regionization using [Google's S2 library](https://github.com/google/s2geometry)
* `VoronoiRegionizer` - regionization using Voronoi diagram
* `AdministativeBoundaryRegionizer` - regionization using administrative boundaries

Example:

```python
from srai.regionizers import H3Regionizer
from srai.utils import geocode_to_region_gdf

area = geocode_to_region_gdf("Wrocław, Poland")
regionizer = H3Regionizer(resolution=8)

regions = regionizer.transform(area)
```

### Embedding

Embedding is a process of mapping regions into a vector space. This can be done in a variety of ways:

* `Hex2VecEmbedder` - embedding using hex2vec[1] algorithm
* `GTFS2VecEmbedder` - embedding using GTFS2Vec[2] algorithm
* `CountEmbedder` - embedding using count of features
* `ContextualCountEmbedder` - embedding using count of features in a given context (proposed in [3])
* `Highway2VecEmbedder` - embedding using Highway2Vec[4] algorithm

All of those methods share the same API. All of them require results from `Loader` (load features), `Regionizer` (split area into regions) and `Joiner` (join features to regions) to work. An example using `CountEmbedder`:

```python
from srai.embedders import CountEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader
from srai.regionizers import H3Regionizer
from srai.utils import geocode_to_region_gdf

loader = OSMPbfLoader()
regionizer = H3Regionizer(resolution=9)
joiner = IntersectionJoiner()

area = geocode_to_region_gdf("Wrocław, Poland")
features = loader.load(area)
regions = regionizer.transform(area)
joint = joiner.transform(regions, features)

embedder = CountEmbedder()
embeddings = embedder.transform(regions, features, joint)
```

`CountEmbedder` is a simple method, which does not require fitting. Other methods, such as `Hex2VecEmbedder` or `GTFS2VecEmbedder` require fitting and can be used in a similar way to `scikit-learn` estimators:

```python
from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader
from srai.regionizers import H3Regionizer

loader = OSMPbfLoader()
regionizer = H3Regionizer(resolution=9)
joiner = IntersectionJoiner()

area = geocode_to_region_gdf("Wrocław, Poland")
features = loader.load(area)
regions = regionizer.transform(area)
joint = joiner.transform(regions, features)

embedder = Hex2VecEmbedder()

# Option 1: fit and transform
embedder.fit(regions, features, joint)
embeddings = embedder.transform(regions, features, joint)

# Option 2: fit_transform
embeddings = embedder.fit_transform(regions, features, joint)
```

### Plotting, utilities and more

We also provide utilities for different spatial operations and plotting functions adopted to data formats used in `srai` For a full list of available methods, please refer to the [documentation](https://srai-lab.github.io/srai).

## Publications

Some of the methods implemented in `srai` have been published in scientific journals and conferences.

1. Szymon Woźniak and Piotr Szymański. 2021. Hex2vec: Context-Aware Embedding H3 Hexagons with OpenStreetMap Tags. In Proceedings of the 4th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GEOAI '21). Association for Computing Machinery, New York, NY, USA, 61–71. [paper](https://doi.org/10.1145/3486635.3491076), [arXiv](https://arxiv.org/abs/2111.00970)
2. Piotr Gramacki, Szymon Woźniak, and Piotr Szymański. 2021. Gtfs2vec: Learning GTFS Embeddings for comparing Public Transport Offer in Microregions. In Proceedings of the 1st ACM SIGSPATIAL International Workshop on Searching and Mining Large Collections of Geospatial Data (GeoSearch'21). Association for Computing Machinery, New York, NY, USA, 5–12. [paper](https://doi.org/10.1145/3486640.3491392), [arXiv](https://arxiv.org/abs/2111.00960)
3. Kamil Raczycki and Piotr Szymański. 2021. Transfer learning approach to bicycle-sharing systems' station location planning using OpenStreetMap data. In Proceedings of the 4th ACM SIGSPATIAL International Workshop on Advances in Resilient and Intelligent Cities (ARIC '21). Association for Computing Machinery, New York, NY, USA, 1–12. [paper](https://doi.org/10.1145/3486626.3493434), [arXiv](https://arxiv.org/abs/2111.00990)
4. Kacper Leśniara and Piotr Szymański. 2022. Highway2vec: representing OpenStreetMap microregions with respect to their road network characteristics. In Proceedings of the 5th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GeoAI '22). Association for Computing Machinery, New York, NY, USA, 18–29. [paper](https://doi.org/10.1145/3557918.3565865)


## Citation
TBD
