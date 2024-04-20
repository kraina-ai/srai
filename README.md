<p align="center">
  <img width="300" src="https://raw.githubusercontent.com/kraina-ai/srai/main/docs/assets/logos/srai-logo-transparent.png">
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/kraina-ai/srai?logo=apache&logoColor=%23fff">
    <img src="https://img.shields.io/github/checks-status/kraina-ai/srai/main?logo=GitHubActions&logoColor=%23fff" alt="Checks">
    <a href="https://github.com/kraina-ai/srai/actions/workflows/ci-dev.yml" target="_blank">
        <img alt="GitHub Workflow Status - DEV" src="https://img.shields.io/github/actions/workflow/status/kraina-ai/srai/ci-dev.yml?label=build-dev&logo=GitHubActions&logoColor=%23fff">
    </a>
    <a href="https://github.com/kraina-ai/srai/actions/workflows/ci-prod.yml" target="_blank">
        <img alt="GitHub Workflow Status - PROD" src="https://img.shields.io/github/actions/workflow/status/kraina-ai/srai/ci-prod.yml?label=build-prod&logo=GitHubActions&logoColor=%23fff">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/kraina-ai/srai/main" target="_blank">
        <img src="https://results.pre-commit.ci/badge/github/kraina-ai/srai/main.svg" alt="pre-commit.ci status">
    </a>
    <a href="https://www.codefactor.io/repository/github/kraina-ai/srai"><img alt="CodeFactor Grade" src="https://img.shields.io/codefactor/grade/github/kraina-ai/srai?logo=codefactor&logoColor=%23fff"></a>
    <a href="https://app.codecov.io/gh/kraina-ai/srai/tree/main"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/kraina-ai/srai?logo=codecov&token=PRS4E02ZX0&logoColor=%23fff"></a>
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

<p align="center">⚠️🚧 This library is under HEAVY development. Expect breaking changes between <i>minor</i> versions 🚧⚠️</p>

<p align="center">💬 Feel free to open an issue if you find anything confusing or not working 💬</p>

Project **Spatial Representations for Artificial Intelligence** (`srai`) aims to provide simple and efficient solutions to geospatial problems that are accessible to everybody and reusable in various contexts where geospatial data can be used. It is a Python module integrating many geo-related algorithms in a single package with unified API. Please see getting started for installation and quick start instructions.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1iyajkX81PLrel-Xmz1lQpYQvePnLoO1U"  style="max-width:600px;width:100%"/>
</p>

## Use cases

In the current state, `srai` provides the following functionalities:

* **OSM data download** - downloading OpenStreetMap data for a given area using different sources
* **OSM data processing** - processing OSM data to extract useful information (e.g. road network, buildings, POIs, etc.)
* **GTFS processing** - extracting features from GTFS data
* **Regionalization** - splitting a given area into smaller regions using different algorithms (e.g. Uber's H3[1], Voronoi, etc.)
* **Embedding** - embedding regions into a vector space based on different spatial features, and using different algorithms (eg. hex2vec[2], etc.)
* Utilities for spatial data visualization and processing

For future releases, we plan to add more functionalities, such as:

* **Pre-computed embeddings** - pre-computed embeddings for different regions and different embedding algorithms
* **Full pipelines** - full pipelines for different embedding approaches, pre-configured from `srai` components
* **Image data download and processing** - downloading and processing image data (eg. OSM tiles, etc.)

### End-to-end examples

Right now, `srai` provides a toolset for data download and processing sufficient to solve downstream tasks. Please see [this project](https://kraina-ai.github.io/Transfer-learning-approach-to-bicycle-sharing-systems-station-location-planning-using-OpenStreetMap/) by [@RaczeQ](https://github.com/RaczeQ), which predicts Bike Sharing System (BSS) stations' locations for a wide range of cities worldwide.

For `srai` integration into full [kedro](https://kedro.org/) pipeline, see [this project](https://github.com/Calychas/highway2vec_remaster/) by [@Calychas](https://github.com/Calychas).

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
* `srai[voronoi]` - dependencies to use Voronoi-based regionalization method
* `srai[gtfs]` - dependencies to process GTFS data
* `srai[plotting]` - dependencies to plot graphs and maps
* `srai[torch]` - dependencies to use torch-based embedders

## Tutorial

For a full tutorial on `srai` and geospatial data in general visit the [srai-tutorial](https://github.com/kraina-ai/srai-tutorial) repository. It contains easy to follow jupyter notebooks concentrating on every part of the library. Additionally, there is [a recording](https://www.youtube.com/watch?v=JlyPh_AdQ8E) available from the EuroScipy 2023 conference covering that material.

## Usage

If you prefer an interactive notebook, examples of `srai` usage are available in this [Colab Notebook](https://colab.research.google.com/drive/17z2OYZG82FZNRK86Kt-eSgbJn9m7meSH?usp=sharing)

### Downloading OSM data

To download OSM data for a given area, using a set of tags use one of `OSMLoader` classes:

* `OSMOnlineLoader` - downloads data from OpenStreetMap API using [osmnx](https://github.com/gboeing/osmnx) - this is faster for smaller areas or tags counts
* `OSMPbfLoader` - loads data from automatically downloaded PBF file from [protomaps](https://protomaps.com/) - this is faster for larger areas or tags counts

Example with `OSMOnlineLoader`:

```python
from srai.loaders import OSMOnlineLoader
from srai.plotting import plot_regions
from srai.regionalizers import geocode_to_region_gdf

query = {"leisure": "park"}
area = geocode_to_region_gdf("Wrocław, Poland")
loader = OSMOnlineLoader()

parks_gdf = loader.load(area, query)
folium_map = plot_regions(area, colormap=["rgba(0,0,0,0)"], tiles_style="CartoDB positron")
parks_gdf.explore(m=folium_map, color="forestgreen")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kraina-ai/srai/main/docs/assets/images/downloading_osm_data.jpg"  style="max-width:600px;width:100%"/>
</p>

### Downloading road network

Road network downloading is a special case of OSM data downloading. To download road network for a given area, use `OSMWayLoader` class:

```python
from srai.loaders import OSMNetworkType, OSMWayLoader
from srai.plotting import plot_regions
from srai.regionalizers import geocode_to_region_gdf

area = geocode_to_region_gdf("Utrecht, Netherlands")
loader = OSMWayLoader(OSMNetworkType.BIKE)

nodes, edges = loader.load(area)

folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
edges[["geometry"]].explore(m=folium_map, color="seagreen")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kraina-ai/srai/main/docs/assets/images/downloading_road_network_data.jpg" style="max-width:600px;width:100%"/>
</p>

### Downloading GTFS data

To extract features from GTFS use `GTFSLoader`. It will extract trip count and available directions for each stop in 1h time windows.

```python
from pathlib import Path

from srai.loaders import GTFSLoader, download_file
from srai.plotting import plot_regions
from srai.regionalizers import geocode_to_region_gdf

area = geocode_to_region_gdf("Vienna, Austria")
gtfs_file = Path("vienna_gtfs.zip")
download_file("https://transitfeeds.com/p/stadt-wien/888/latest/download", gtfs_file.as_posix())
loader = GTFSLoader()

features = loader.load(gtfs_file)

folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
features[["trips_at_8", "geometry"]].explore("trips_at_8", m=folium_map)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kraina-ai/srai/main/docs/assets/images/downloading_gtfs_data.jpg" style="max-width:600px;width:100%"/>
</p>

### Regionalization

Regionalization is a process of dividing a given area into smaller regions. This can be done in a variety of ways:

* `H3Regionalizer` - regionalization using [Uber's H3 library](https://github.com/uber/h3)
* `S2Regionalizer` - regionalization using [Google's S2 library](https://github.com/google/s2geometry)
* `VoronoiRegionalizer` - regionalization using Voronoi diagram
* `AdministativeBoundaryRegionalizer` - regionalization using administrative boundaries

Example:

```python
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

area = geocode_to_region_gdf("Berlin, Germany")
regionalizer = H3Regionalizer(resolution=7)

regions = regionalizer.transform(area)

folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
plot_regions(regions_gdf=regions, map=folium_map)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kraina-ai/srai/main/docs/assets/images/regionalization.jpg" style="max-width:600px;width:100%"/>
</p>

### Embedding

Embedding is a process of mapping regions into a vector space. This can be done in a variety of ways:

* `Hex2VecEmbedder` - embedding using hex2vec[1] algorithm
* `GTFS2VecEmbedder` - embedding using GTFS2Vec[2] algorithm
* `CountEmbedder` - embedding based on features counts
* `ContextualCountEmbedder` - embedding based on features counts with neighbourhood context (proposed in [3])
* `Highway2VecEmbedder` - embedding using Highway2Vec[4] algorithm

All of those methods share the same API. All of them require results from `Loader` (load features), `Regionalizer` (split area into regions) and `Joiner` (join features to regions) to work. An example using `CountEmbedder`:

```python
from srai.embedders import CountEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMOnlineLoader
from srai.plotting import plot_regions, plot_numeric_data
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

loader = OSMOnlineLoader()
regionalizer = H3Regionalizer(resolution=9)
joiner = IntersectionJoiner()

query = {"amenity": "bicycle_parking"}
area = geocode_to_region_gdf("Malmö, Sweden")
features = loader.load(area, query)
regions = regionalizer.transform(area)
joint = joiner.transform(regions, features)

embedder = CountEmbedder()
embeddings = embedder.transform(regions, features, joint)

folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
plot_numeric_data(regions, "amenity_bicycle_parking", embeddings, map=folium_map)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kraina-ai/srai/main/docs/assets/images/embedding_count_embedder.jpg" style="max-width:600px;width:100%"/>
</p>

`CountEmbedder` is a simple method, which does not require fitting. Other methods, such as `Hex2VecEmbedder` or `GTFS2VecEmbedder` require fitting and can be used in a similar way to `scikit-learn` estimators:

```python
from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from srai.plotting import plot_regions, plot_numeric_data

loader = OSMPbfLoader()
regionalizer = H3Regionalizer(resolution=11)
joiner = IntersectionJoiner()

area = geocode_to_region_gdf("City of London")
features = loader.load(area, HEX2VEC_FILTER)
regions = regionalizer.transform(area)
joint = joiner.transform(regions, features)

embedder = Hex2VecEmbedder()
neighbourhood = H3Neighbourhood(regions_gdf=regions)

embedder = Hex2VecEmbedder([15, 10, 3])

# Option 1: fit and transform
# embedder.fit(regions, features, joint, neighbourhood, batch_size=128)
# embeddings = embedder.transform(regions, features, joint)

# Option 2: fit_transform
embeddings = embedder.fit_transform(regions, features, joint, neighbourhood, batch_size=128)

folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
plot_numeric_data(regions, 0, embeddings, map=folium_map)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kraina-ai/srai/main/docs/assets/images/embedding_hex2vec_embedder.jpg" style="max-width:600px;width:100%"/>
</p>

### Pre-trained models usage

We provide pre-trained models for some of the embedding methods. To use them, simply download them from [here](https://drive.google.com/drive/folders/14sH33-kNxA0q1O1abPWTpuix8raR_XbD?usp=drive_link) and load them using `load` method:

```python
from srai.embedders import Hex2VecEmbedder

model_path = "path/to/model"
embedder = Hex2VecEmbedder.load(model_path)
```

### Plotting, utilities and more

We also provide utilities for different spatial operations and plotting functions adopted to data formats used in `srai` For a full list of available methods, please refer to the [documentation](https://kraina-ai.github.io/srai).

## Contributing

If you are willing to contribute to `srai`, feel free to do so! Visit [our contributing guide](./CONTRIBUTING.md) for more details.

## Publications

Some of the methods implemented in `srai` have been published in scientific journals and conferences.

1. Szymon Woźniak and Piotr Szymański. 2021. Hex2vec: Context-Aware Embedding H3 Hexagons with OpenStreetMap Tags. In Proceedings of the 4th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GEOAI '21). Association for Computing Machinery, New York, NY, USA, 61–71. [paper](https://doi.org/10.1145/3486635.3491076), [arXiv](https://arxiv.org/abs/2111.00970)
2. Piotr Gramacki, Szymon Woźniak, and Piotr Szymański. 2021. Gtfs2vec: Learning GTFS Embeddings for comparing Public Transport Offer in Microregions. In Proceedings of the 1st ACM SIGSPATIAL International Workshop on Searching and Mining Large Collections of Geospatial Data (GeoSearch'21). Association for Computing Machinery, New York, NY, USA, 5–12. [paper](https://doi.org/10.1145/3486640.3491392), [arXiv](https://arxiv.org/abs/2111.00960)
3. Kamil Raczycki and Piotr Szymański. 2021. Transfer learning approach to bicycle-sharing systems' station location planning using OpenStreetMap data. In Proceedings of the 4th ACM SIGSPATIAL International Workshop on Advances in Resilient and Intelligent Cities (ARIC '21). Association for Computing Machinery, New York, NY, USA, 1–12. [paper](https://doi.org/10.1145/3486626.3493434), [arXiv](https://arxiv.org/abs/2111.00990)
4. Kacper Leśniara and Piotr Szymański. 2022. Highway2vec: representing OpenStreetMap microregions with respect to their road network characteristics. In Proceedings of the 5th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GeoAI '22). Association for Computing Machinery, New York, NY, USA, 18–29. [paper](https://doi.org/10.1145/3557918.3565865), [arXiv](https://arxiv.org/abs/2304.13865)
5. Daniele Donghi and Anne Morvan. 2023. GeoVeX: Geospatial Vectors with Hexagonal Convolutional Autoencoders. In Proceedings of the 6th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GeoAI '23). Association for Computing Machinery, New York, NY, USA, 3–13. [paper](https://doi.org/10.1145/3615886.3627750)

## Acknowledgements

We would like to thank Piotr Szymański PhD \([@niedakh](https://twitter.com/niedakh)\) for his invaluable guidance and support in the development of this library. His expertise and mentorship have been instrumental in shaping the library's design and functionality, and we are very grateful for his input.

## Citation

If you wish to cite the SRAI library, please use our [paper](https://arxiv.org/abs/2310.13098)

```bibtex
@inproceedings{
  Gramacki_SRAI_Towards_Standardization_2023,
  author = {
    Gramacki, Piotr and
    Leśniara, Kacper and
    Raczycki, Kamil and
    Woźniak, Szymon and
    Przymus, Marcin and
    Szymański, Piotr
  },
  booktitle = {Proceedings of the 6th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery},
  month = nov,
  publisher = {Association for Computing Machinery},
  title = {{SRAI: Towards Standardization of Geospatial AI}},
  url = {https://dl.acm.org/doi/10.1145/3615886.3627740},
  year = {2023}
}
```

## License

This library is licensed under the [Apache License 2.0](https://github.com/kraina-ai/srai/blob/main/LICENSE.md).

The free [OpenStreetMap](https://www.openstreetmap.org/) data, which is used for the development of SRAI, is licensed under the [Open Data Commons Open Database License](https://opendatacommons.org/licenses/odbl/) (ODbL) by the [OpenStreetMap Foundation](https://osmfoundation.org/) (OSMF).
