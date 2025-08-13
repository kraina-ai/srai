# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `rq-geo-toolkit`, `geoarrow-rust-core` and `pooch` libraries to the dependencies

### Changed

- Bumped `QuackOSM` version to `0.14.0`
- Bumped `OvertureMaestro` version to `0.5.0`
- Refactored download function to use `pooch` library instead of `requests`

### Fixed

- Enabled `CountEmbedder` to parse dataframes with indexes other than string type [#537](https://github.com/kraina-ai/srai/issues/537)

## [0.9.7] - 2025-05-21

### Added

- Option to pass `trainer_kwargs` to `Gtfs2VecEmbedder`'s `fit` and `fit_transform` functions

### Fixed

- Change CRS setting for GeoDataFrame in `OSMPbfLoader`

## [0.9.6] - 2025-04-06

### Added

- Functions `convert_to_regions_gdf` and `convert_to_features_gdf` to transform any existing geo input to an object with the correct index name

## [0.9.5] - 2025-02-23

### Fixed

- Catched OSMnx `InsufficientResponseError` in `OSMOnlineLoader`

## [0.9.4] - 2025-02-16

### Fixed

- Saving expected target features in `GeoVexEmbedder`

## [0.9.3] - 2025-02-10

### Added

- Option to change default aggregation in `ContextualCountEmbedder` from `average` to `median`, `sum`, `min` or `max`

## [0.9.2] - 2025-02-03

### Added

- Option to change inherited parameter `count_subcategories` in `Hex2VecEmbedder` and `GeoVexEmbedder`

### Changed

- `OvertureMapsLoader` docstring with regards to `count_subcategories` parameter
- Reduced memory footprint in `CountEmbedder` by using Arrow's zero-copy protocol

## [0.9.1] - 2025-01-27

### Changed

- Refactored `ContextualCountEmbedder` progress bar and multiprocessing implementation

## [0.9.0] - 2025-01-26

### Added

- `OvertureMapsLoader` for loading features from Overture Maps datasets

### Changed

- Replaced all `union_all` calls with `unary_union()` in GeoPandas context
- Transform logic in `CountEmbedder` to work with new boolean based features dataframes from Overture Maps Loader
- Changed `IntersetionJoiner` logic to use `STRTree` index instead of `sjoin` function
- Refactored `CountEmbedder` to work on the lazy execution engine from the `polars` library

## [0.8.4] - 2025-01-04

### Added

- Option to pass `verbosity_mode` to `OsmPbfLoader`

### Changed

- Default OSM download source from Geofabrik to `any`
- Replaced deprecated function calls from the `QuackOSM` library

## [0.8.3] - 2025-01-01

### Fixed

- Removed GTFS feed validation for `gtfs-kit` versions >= `10.0.0`

## [0.8.2] - 2024-12-30

### Fixed

- Changed polygon creation in spherical voronoi algorithm to avoid rounding error on vertices

## [0.8.1] - 2024-12-30

### Fixed

- Made code compliant with `osmnx`'s new `2.0.0` release in `OsmOnlineLoader`

## [0.8.0] - 2024-12-28

> \[!WARNING]
> This release was yanked. Please use the [0.8.1](https://github.com/kraina-ai/srai/releases/tag/0.8.1) release instead.

### Fixed

- Made code compliant with `osmnx`'s new `2.0.0` release [#473](https://github.com/kraina-ai/srai/issues/473)
- Made code compliant with `h3ronpy`'s new `0.22.0` release [#471](https://github.com/kraina-ai/srai/issues/471)

## [0.7.8] - 2024-12-28

### Changed

- Set max `h3ronpy` version to `<0.22.0` (implemented by [@bouzaghrane](https://github.com/bouzaghrane))
- Set max `osmnx` version to `<2.0.0`

### Fixed

- Removed a list comprehension in geometry related operations (implemented by [@ebonnal](https://github.com/ebonnal))

## [0.7.7] - 2024-09-25

### Changed

- Bumped minimal QuackOSM version to `0.11.0`

## [0.7.6] - 2024-08-29

### Fixed

- Changed a typo in the `BASE_OSM_GROUPS_FILTER` (implemented by [@starsep](https://github.com/starsep))

### Added

- Added methods for saving and loading fittable `GeoVexEmbedder` (implemented by [@sabman](https://github.com/sabman) and [@mhassanch](https://github.com/mhassanch)) [#457](https://github.com/kraina-ai/srai/issues/457)

## [0.7.5] - 2024-06-04

### Fixed

- Changed H3 regionalization logic from `IntersectsBoundary` to `Covers` to properly assign H3 regions to geometries smaller than an H3 cell

## [0.7.4] - 2024-05-05

### Added

- Support for Python 3.12 after upgrading QuackOSM to `0.7.0` and DuckDB to `0.10.2`

### Changed

- Refactored `ContextualCountEmbedder` by adding multiprocessing for faster transformations

## [0.7.3] - 2024-04-21

### Changed

- Make geofabrik the default download source for OSMPbfLoader

## [0.7.2] - 2024-04-20

### Changed

- Update tooling
- Remove Black in favour of Ruff format

### Added

- Conform to PEP 561

## [0.7.1] - 2024-04-17

### Fixed

- Issue caused by the change in the `h3-py` library 4.0.0b3 [#431](https://github.com/kraina-ai/srai/issues/431)

## [0.7.0] - 2024-02-02

### Added

- Support for `BBBike` download service.
- Option to merge a list of OSM tags filters into a single one.

### Changed

- Refactored `PbfFileHandler` to use `QuackOSM` library instead of `osmium` and `GDAL` [#405](https://github.com/kraina-ai/srai/pull/405)
- Changed the default pbf download source from `protomaps` download service to `any`.

### Removed

- `PbfFileLoader` and moved its logic to `QuackOSM` library.
- Support for `protomaps` download service.
- `PbfFileClipper` (unnecessary after geospatial filtering has been incorporated into refactored `PbfFileHandler`) [#405](https://github.com/kraina-ai/srai/pull/405).
- `PbfFileHandler` (unnecessary after moving away from `osmium` implementation).

### Fixed

- Bugs in GTFS Loader: missing index name and NaN handling (implemented by [@zackAemmer](https://github.com/zackAemmer)).

## [0.6.2] - 2023-12-28

### Added

- CI release via GitHub Actions
- Set up docs style for having code examples from docstrings.
- Automatic warnings filtering inside `VoronoiRegionalizer`.

### Changed

- Added option to parse `MultiPolygon` geometries in `srai.geometry.remove_interiors` function.

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

[Unreleased]: https://github.com/kraina-ai/srai/compare/0.9.7...HEAD

[0.9.7]: https://github.com/kraina-ai/srai/compare/0.9.6...0.9.7

[0.9.6]: https://github.com/kraina-ai/srai/compare/0.9.5...0.9.6

[0.9.5]: https://github.com/kraina-ai/srai/compare/0.9.4...0.9.5

[0.9.4]: https://github.com/kraina-ai/srai/compare/0.9.3...0.9.4

[0.9.3]: https://github.com/kraina-ai/srai/compare/0.9.2...0.9.3

[0.9.2]: https://github.com/kraina-ai/srai/compare/0.9.1...0.9.2

[0.9.1]: https://github.com/kraina-ai/srai/compare/0.9.0...0.9.1

[0.9.0]: https://github.com/kraina-ai/srai/compare/0.8.4...0.9.0

[0.8.4]: https://github.com/kraina-ai/srai/compare/0.8.3...0.8.4

[0.8.3]: https://github.com/kraina-ai/srai/compare/0.8.2...0.8.3

[0.8.2]: https://github.com/kraina-ai/srai/compare/0.8.1...0.8.2

[0.8.1]: https://github.com/kraina-ai/srai/compare/0.8.0...0.8.1

[0.8.0]: https://github.com/kraina-ai/srai/compare/0.7.8...0.8.0

[0.7.8]: https://github.com/kraina-ai/srai/compare/0.7.7...0.7.8

[0.7.7]: https://github.com/kraina-ai/srai/compare/0.7.6...0.7.7

[0.7.6]: https://github.com/kraina-ai/srai/compare/0.7.5...0.7.6

[0.7.5]: https://github.com/kraina-ai/srai/compare/0.7.4...0.7.5

[0.7.4]: https://github.com/kraina-ai/srai/compare/0.7.3...0.7.4

[0.7.3]: https://github.com/kraina-ai/srai/compare/0.7.2...0.7.3

[0.7.2]: https://github.com/kraina-ai/srai/compare/0.7.1...0.7.2

[0.7.1]: https://github.com/kraina-ai/srai/compare/0.7.0...0.7.1

[0.7.0]: https://github.com/kraina-ai/srai/compare/0.6.2...0.7.0

[0.6.2]: https://github.com/kraina-ai/srai/compare/0.6.1...0.6.2

[0.6.1]: https://github.com/kraina-ai/srai/compare/0.6.0...0.6.1

[0.6.0]: https://github.com/kraina-ai/srai/compare/0.5.2...0.6.0

[0.5.2]: https://github.com/kraina-ai/srai/compare/0.5.1...0.5.2

[0.5.1]: https://github.com/kraina-ai/srai/compare/0.5.0...0.5.1

[0.5.0]: https://github.com/kraina-ai/srai/compare/0.4.1...0.5.0

[0.4.1]: https://github.com/kraina-ai/srai/compare/0.4.0...0.4.1

[0.4.0]: https://github.com/kraina-ai/srai/compare/0.3.3...0.4.0

[0.3.3]: https://github.com/kraina-ai/srai/compare/0.3.2...0.3.3

[0.3.2]: https://github.com/kraina-ai/srai/compare/0.3.1...0.3.2

[0.3.1]: https://github.com/kraina-ai/srai/compare/0.3.0...0.3.1

[0.3.0]: https://github.com/kraina-ai/srai/compare/0.2.0...0.3.0

[0.2.0]: https://github.com/kraina-ai/srai/compare/0.1.1...0.2.0

[0.1.1]: https://github.com/kraina-ai/srai/compare/0.0.1...0.1.1

[0.0.1]: https://github.com/kraina-ai/srai/releases/tag/0.0.1
