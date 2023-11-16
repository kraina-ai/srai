# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.3] - 2023-11-16

## [0.6.2] - 2023-11-16

### Added

- CI release via GitHub Actions

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.6.1] - 2023-11-12

### Added

- Start writing examples in docstrings.

### Changed

- Change documentation rendering style.

### Fixed

- Expose functions in `h3` module.

## [0.6.0] - 2023-11-02

### Changed

- Update code to use Python 3.9 syntax.

### Removed

- Support for Python 3.8.

## [0.5.2] - 2023-10-29

### Added

- Geofabrik and OSM Fr index caching and reading from cache to avoid reloading.
- Tests for Windows OS.

### Changed

- `PbfFileClipper` temporary files operations moved to working directory.

## [0.5.1] - 2023-10-27

### Added

- Option for `CountEmbedder` and every other descendant embedder to use `OsmTagsFilter` and `GroupedOsmTagsFilter` as input to `expected_output_features` parameter.

### Changed

- Modified `GEOFABRIK_LAYERS` definition to make more precise `building` tag values list by applying all accepted and popular tag values from `tagsinfo`.

## [0.5.0] - 2023-10-26

### Added

- `GeoVex` model as a `GeoVexEmbedder` implemented by [@mschrader15](https://github.com/mschrader15), proposed by [@ddonghi](https://github.com/ddonghi) & [@annemorvan](https://github.com/annemorvan)
- Buffer geometries using `H3` cells by [@mschrader15](https://github.com/mschrader15)
- Option for `OSMLoaders` to parse any Shapely geometry, not only `GeoDataFrames`.

## [0.4.1] - 2023-10-23

### Changed

- Added more verbosity to the `AdministrativeBoundaryRegionalizer`.

### Fixed

- Added automatic features count clearing in `PbfFileHandler` after operation.
- Added directory creation before saving OSM extracts index file.

## [0.4.0] - 2023-10-21

### Added

- `PbfFileClipper` for cutting bigger `osm.pbf` files into smaller sizes for faster operations. Included clipping inside `PbfFileDownloader` for new bigger extracts sources. [#369](https://github.com/kraina-ai/srai/issues/369)

### Changed

- Bumped `h3ronpy` library to `0.18.0` with added support for MacOS. Removed override with check for H3 operations if system is `darwin`. Changed internal API to use `ContainmentMode`.
- Refactored `OSMLoader`'s `GroupedOsmTagsFilter` features grouping to be faster by refactoring pandas operations [#354](https://github.com/srai-lab/srai/issues/354)
- Sped up `VoronoiRegionalizer` by removing redundant intersection operations and vectorizing ecdf2geodetic calculations [#351](https://github.com/kraina-ai/srai/issues/351)
- Sped up `ContextualCountEmbedder` by removing iteration over dataframe rows and vectorizing operations to work at a whole `numpy` array at once [#359](https://github.com/kraina-ai/srai/issues/359)
- Added Geofabrik and OpenStreetMap.fr PBF extracts download services. Added automatic switch from default `protomaps` download service to `geofabrik` on error. [#158](https://github.com/kraina-ai/srai/issues/158) [#366](https://github.com/kraina-ai/srai/issues/366)

## [0.3.3] - 2023-08-13

### Changed

- Modified `OSMPbfLoader` intersection logic.
- Changed default tiles style for `plotting.plot_numeric_data` function.

## [0.3.2] - 2023-08-12

### Changed

- Migrated the repository ownership from `srai-lab` to `kraina-ai`.
- Improved speed of `OSMPbfLoader` by moving intersection step to the end.
- Changed API and improved `plotting.plot_numeric_data` function.
- Changed `AdministrativeBoundaryRegionalizer` loading speed.

### Fixed

- Added checks for `osmnx` `1.5.0` version with deprecated `geometry` module.

## [0.3.1] - 2023-08-09

### Fixed

- Repaired bug with `VoronoiRegionalizer` and wrong polygon orientation.

## [0.3.0] - 2023-08-08

### Added

- `include_center` parameter to Neighbourhoods [#288](https://github.com/srai-lab/srai/issues/288)
- `__version__` entry to library API. [#305](https://github.com/srai-lab/srai/issues/305)
- `srai.h3` module with functions for translating list of h3 cells into shapely polygons and calculating local ij coordinates.

### Changed

- Refactored H3Regionalizer to be faster using [h3ronpy](https://github.com/nmandery/h3ronpy) library [#311](https://github.com/srai-lab/srai/issues/311)
- BREAKING! Renamed NetworkType to OSMNetworkType and made it importable directly from `srai.loaders` [#227](https://github.com/srai-lab/srai/issues/227)
- BREAKING! Renamed osm\_filter\_type and grouped\_osm\_filter\_type into OsmTagsFilter and GroupedOsmTagsFilter [#261](https://github.com/srai-lab/srai/issues/261)
- Removed osmnx dependency version cap [#303](https://github.com/srai-lab/srai/issues/303)
- BREAKING! Removed `utils` module [#128](https://github.com/srai-lab/srai/issues/128)
  - `srai.utils._optional` moved to `srai._optional`
  - `srai.utils._pytorch_stubs` moved to `srai.embedders._pytorch_stubs`
  - `srai.utils.download` moved to `srai.loaders.download` (and can be imported with `from srai.loaders import download_file`)
  - `srai.utils.geocode` moved to `srai.regionalizers.geocode` (and can be imported with `from srai.regionalizers import geocode_to_region_gdf`)
  - `srai.utils.geometry` and `srai.utils.merge` moved to `srai.geometry`
  - `srai.utils.typing` moved to `srai._typing`

### Fixed

- Improved simplification and buffering of polygons for Protomaps extracts [#309](https://github.com/srai-lab/srai/issues/309)
- Eliminated some occasional errors in large scale executions of VoronoiRegionalizer [#313](https://github.com/srai-lab/srai/issues/313)

## [0.2.0] - 2023-07-05

### Added

- Loading and saving fittable embedders

### Changed

- BREAKING: renamed Regionizer to Regionalizer [#282](https://github.com/srai-lab/srai/issues/282)

### Fixed

- Freeze osmnx version to <=1.4.0, as 1.5.0 is not compatible with our code [#303](https://github.com/srai-lab/srai/issues/303)

## [0.1.1] - 2023-04-27

### Added

- SlippyMapRegionizer
- OSMTileLoader
- GTFS Loader from gtfs2vec paper
- GTFS2Vec Model from gtfs2vec paper
- GTFS2Vec Embedder using gtfs2vec model
- Hex2Vec Model from hex2vec paper
- Hex2Vec Embedder using hex2vec model
- Highway2Vec Model from highway2vec paper
- Highway2Vec Embedder using highway2vec model
- OSMOnlineLoader
- OSMPbfLoader
- OSMWayLoader
- Neighbourhood
- H3Neighbourhood
- AdjacencyNeighbourhood
- CountEmbedder
- ContextualCountEmbedder
- (CI) Changelog Enforcer
- Utility plotting module based on Folium and Plotly
- Project README
- Documentation for srai library
- Citation information

### Changed

- Change embedders and joiners interface to have `.transform` method
- Change linter to Ruff and removed flake8, isort, pydocstyle
- Change default value inside `transform` function of IntersectionJoiner to not return geometry.
- Make torch and pytorch-lightning as optional dependencies ([#210](https://github.com/srai-lab/srai/issues/210))

### Fixed

- IntersectionJoiner incorrectly returned feature columns when `return_geom=False` ([#208](https://github.com/srai-lab/srai/issues/208))
- Tests for pandas >=2

## [0.0.1] - 2022-11-23

### Added

- PDM as a dependency management tool
- black, flake8, isort, mypy, pytest-cov
- pre-commit configuration
- Apache 2.0 license
- mkdocs for documentation
- GitHub pages to host documentation
- initial requirements
- H3 Regionizer
- Voronoi Regionizer
- Administrative Boundary Regionizer
- Intersection Joiner
- Geoparquet Loader

[Unreleased]: https://github.com/kraina-ai/srai-sandbox/compare/0.6.3...HEAD

[0.6.3]: https://github.com/kraina-ai/srai-sandbox/compare/0.6.2...0.6.3

[0.6.2]: https://github.com/kraina-ai/srai-sandbox/compare/0.6.1...0.6.2

[0.6.1]: https://github.com/kraina-ai/srai-sandbox/compare/0.6.0...0.6.1

[0.6.0]: https://github.com/kraina-ai/srai-sandbox/compare/0.5.2...0.6.0

[0.5.2]: https://github.com/kraina-ai/srai-sandbox/compare/0.5.1...0.5.2

[0.5.1]: https://github.com/kraina-ai/srai-sandbox/compare/0.5.0...0.5.1

[0.5.0]: https://github.com/kraina-ai/srai-sandbox/compare/0.4.1...0.5.0

[0.4.1]: https://github.com/kraina-ai/srai-sandbox/compare/0.4.0...0.4.1

[0.4.0]: https://github.com/kraina-ai/srai-sandbox/compare/0.3.3...0.4.0

[0.3.3]: https://github.com/kraina-ai/srai-sandbox/compare/0.3.2...0.3.3

[0.3.2]: https://github.com/kraina-ai/srai-sandbox/compare/0.3.1...0.3.2

[0.3.1]: https://github.com/kraina-ai/srai-sandbox/compare/0.3.0...0.3.1

[0.3.0]: https://github.com/kraina-ai/srai-sandbox/compare/0.2.0...0.3.0

[0.2.0]: https://github.com/kraina-ai/srai-sandbox/compare/0.1.1...0.2.0

[0.1.1]: https://github.com/kraina-ai/srai-sandbox/compare/0.0.1...0.1.1

[0.0.1]: https://github.com/kraina-ai/srai-sandbox/releases/tag/0.0.1
