# Spatial Representations for Artificial Intelligence

<p align="center">
  <img width="300" src="assets/logos/srai-logo-transparent.png">
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

Project *Spatial Representations for Artificial Intelligence* (`srai`) aims to provide simple and efficient solutions to geospatial problems that are accessible to everybody and reusable in various contexts where geospatial data can be used. It is a Python module integrating many geo-related algorithms in a single package with unified API. Please see [getting starded](getting_started) for installation and quick srart instructions.


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

## Publications

Some of the methods implemented in `srai` have been published in scientific journals and conferences.

1. Szymon Woźniak and Piotr Szymański. 2021. Hex2vec: Context-Aware Embedding H3 Hexagons with OpenStreetMap Tags. In Proceedings of the 4th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GEOAI '21). Association for Computing Machinery, New York, NY, USA, 61–71. [paper](https://doi.org/10.1145/3486635.3491076), [arXiv](https://arxiv.org/abs/2111.00970)
2. Piotr Gramacki, Szymon Woźniak, and Piotr Szymański. 2021. Gtfs2vec: Learning GTFS Embeddings for comparing Public Transport Offer in Microregions. In Proceedings of the 1st ACM SIGSPATIAL International Workshop on Searching and Mining Large Collections of Geospatial Data (GeoSearch'21). Association for Computing Machinery, New York, NY, USA, 5–12. [paper](https://doi.org/10.1145/3486640.3491392), [arXiv](https://arxiv.org/abs/2111.00960)
3. Kamil Raczycki and Piotr Szymański. 2021. Transfer learning approach to bicycle-sharing systems' station location planning using OpenStreetMap data. In Proceedings of the 4th ACM SIGSPATIAL International Workshop on Advances in Resilient and Intelligent Cities (ARIC '21). Association for Computing Machinery, New York, NY, USA, 1–12. [paper](https://doi.org/10.1145/3486626.3493434), [arXiv](https://arxiv.org/abs/2111.00990)
4. Kacper Leśniara and Piotr Szymański. 2022. Highway2vec: representing OpenStreetMap microregions with respect to their road network characteristics. In Proceedings of the 5th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GeoAI '22). Association for Computing Machinery, New York, NY, USA, 18–29. [paper](https://doi.org/10.1145/3557918.3565865)

## References

1. https://h3geo.org/
2. https://doi.org/10.1145/3486635.3491076
