# Getting started guide

## Installation

To install `srai` simply run:

```bash
pip install srai
```

This will install the `srai` package and dependencies required by most of the use cases. There are several optional dependencies that can be installed to enable additional functionality. These are listed in the [optional dependencies](#optional-dependencies) section.

## Optional dependencies

The following optional dependencies can be installed to enable additional functionality:

* `srai[all]` - all optional dependencies
* `srai[osm]` - dependencies required to download OpenStreetMap data
* `srai[voronoi]` - dependencies to use Voronoi-based regionalization method
* `srai[gtfs]` - dependencies to process GTFS data
* `srai[plotting]` - dependencies to plot graphs and maps
* `srai[torch]` - dependencies to use torch-based embedders

## Usage

Please see the [examples](../examples) directory for examples of how to use `srai`.

## Use cases

Please see the [Use Cases](use_cases) for detailed description of `srai` use-cases.
It also covers features planned for future releases as a roadmap of what this project's
goals are.

## Full pipeline example

We also provide a full pipeline implemented in `kedro` to showcase real-world `srai` usage. Pipeline is available [here](https://github.com/Calychas/highway2vec_remaster/).
