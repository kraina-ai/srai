---
title: OBSR Datasets
weight: 100
---


# OBSR Datasets

Examples showcasing the available datasets from the Open Benchmark for Spatial Representations (OBSR).

- [Airbnb Multicity](airbnb_multicity.ipynb)
- [Chicago Crime](chicago_crime.ipynb)
- [Geolife](geolife.ipynb)
- [House Sales in King County](house_sales_in_king_county.ipynb)
- [Philadelphia Crime](philadelphia_crime.ipynb)
- [Porto Taxi](porto_taxi.ipynb)
- [San Francisco Police Department's (SFPD) Incident Report Datatset](police_department_incidents.ipynb)

## Benchmark and datasets paper

ArXiv: TODO

## Available prediction tasks

Region-based tasks:
- Price prediction
- Activity prediction

Trajectory-based tasks:
- Human Mobility Prediction (HMC)
- Travel Time Estimation (TTE)

|  | Price prediction | Activity prediction | Human Mobility Prediction (HMC) | Travel Time Estimation (TTE) |
|---|:---:|:---:|:---:|:---:|
| [Airbnb Multicity](airbnb_multicity.ipynb) | ✅ |  |  |  |
| [Chicago Crime](chicago_crime.ipynb) |  | ✅ |  |  |
| [Geolife](geolife.ipynb) |  |  | ✅ | ✅ |
| [House Sales in King County](house_sales_in_king_county.ipynb) | ✅ |  |  |  |
| [Philadelphia Crime](philadelphia_crime.ipynb) |  | ✅ |  |  |
| [Porto Taxi](porto_taxi.ipynb) |  |  | ✅ | ✅ |
| [SFPD Incident Report Datatset](police_department_incidents.ipynb) |  | ✅ |  |  |

## Contributing new datasets

Datasets codebase in SRAI is based mainly on HuggingFace backend. The library offers preprocessed datasets with predefined train and test splits, as well as "`raw`" version for users to split manually.

Although most of the datasets are based on data from [`Kraina`](https://huggingface.co/kraina) organisation, we are not limited to it. Each dataset class can point to any public dataset on HuggingFace.

We want to collect and distribute vector based datasets for evaluating geospatial embeddings in a variety of tasks.

Requirements for a new spatial dataset:
- Publicly available
- Has a spatial component (points with a target, collection of points that will be agrgegated, trajectories, regions for classification)
- Doesn't violate any licenses (can be Non-Commercial) - for example, `GeoLife` and `Philadelphia Crime` datasets are downloaded from the original sources, and only splits with IDs are saved on HuggingFace, because their licenses don't allow for redistribution.
