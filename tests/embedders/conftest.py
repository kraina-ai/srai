"""Conftest for Embedders."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely import geometry


@pytest.fixture  # type: ignore
def gdf_empty() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame(geometry=[])


@pytest.fixture  # type: ignore
def gdf_regions() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with 3 hexagonal regions."""
    regions_gdf = gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.02710946531851, 51.110065389823305),
                    (17.029634931698617, 51.1092989279356),
                    (17.03212452567607, 51.11021450606774),
                    (17.032088692873092, 51.11189657169522),
                    (17.029563145936592, 51.11266305206119),
                    (17.02707351236059, 51.11174744831988),
                    (17.02710946531851, 51.110065389823305),
                ],
            ),
            geometry.Polygon(
                shell=[
                    (17.03212452567607, 51.11021450606774),
                    (17.034649970341516, 51.109447934020366),
                    (17.037139662738255, 51.11036340911803),
                    (17.037103950094387, 51.11204548186887),
                    (17.03457842489355, 51.11281207240022),
                    (17.032088692873092, 51.11189657169522),
                    (17.03212452567607, 51.11021450606774),
                ],
            ),
            geometry.Polygon(
                shell=[
                    (17.02952725046974, 51.114345051613405),
                    (17.029563145936592, 51.11266305206119),
                    (17.032088692873092, 51.11189657169522),
                    (17.03457842489355, 51.11281207240022),
                    (17.03454264959235, 51.11449407907883),
                    (17.03201702210393, 51.115260577927586),
                    (17.02952725046974, 51.114345051613405),
                ],
            ),
        ],
        index=pd.Index(
            data=["891e2040897ffff", "891e2040d4bffff", "891e2040d5bffff"], name="region_id"
        ),
        crs="epsg:4326",
    )
    return regions_gdf


@pytest.fixture  # type: ignore
def gdf_features() -> gpd.GeoDataFrame:
    """Get GeoDataFrame with example OSM-like features."""
    features_gdf = gpd.GeoDataFrame(
        {
            "leisure": ["playground", None, "adult_gaming_centre", None],
            "amenity": [None, "pub", "pub", None],
        },
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.0360858, 51.1103927),
                    (17.0358804, 51.1104389),
                    (17.0357855, 51.1105503),
                    (17.0359451, 51.1105907),
                    (17.0361589, 51.1105402),
                    (17.0360858, 51.1103927),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1114868),
                    (17.0320, 51.1114868),
                    (17.0320, 51.1117503),
                    (17.0317168, 51.1117503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1124868),
                    (17.0320, 51.1124868),
                    (17.0320, 51.1127503),
                    (17.0317168, 51.1127503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0307168, 51.1104868),
                    (17.0310, 51.1104868),
                    (17.0310, 51.1107503),
                    (17.0307168, 51.1107503),
                ]
            ),
        ],
        index=pd.Index(
            data=["way/312457804", "way/1533817161", "way/312457812", "way/312457834"],
            name="feature_id",
        ),
        crs="epsg:4326",
    )

    return features_gdf


@pytest.fixture  # type: ignore
def gdf_joint() -> gpd.GeoDataFrame:
    """Get joint GeoDataFrame for matching regions and features from this module."""
    joint_gdf = gpd.GeoDataFrame(
        geometry=[
            geometry.Polygon(
                shell=[
                    (17.0358804, 51.1104389),
                    (17.0357855, 51.1105503),
                    (17.0359451, 51.1105907),
                    (17.0361589, 51.1105402),
                    (17.0360858, 51.1103927),
                    (17.0358804, 51.1104389),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1117503),
                    (17.032, 51.1117503),
                    (17.032, 51.1114868),
                    (17.0317168, 51.1114868),
                    (17.0317168, 51.1117503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0307168, 51.1107503),
                    (17.031, 51.1107503),
                    (17.031, 51.1104868),
                    (17.0307168, 51.1104868),
                    (17.0307168, 51.1107503),
                ]
            ),
            geometry.Polygon(
                shell=[
                    (17.0317168, 51.1127503),
                    (17.032, 51.1127503),
                    (17.032, 51.1124868),
                    (17.0317168, 51.1124868),
                    (17.0317168, 51.1127503),
                ]
            ),
        ],
        index=pd.MultiIndex.from_arrays(
            arrays=[
                ["891e2040d4bffff", "891e2040897ffff", "891e2040897ffff", "891e2040d5bffff"],
                ["way/312457804", "way/1533817161", "way/312457834", "way/312457812"],
            ],
            names=["region_id", "feature_id"],
        ),
        crs="epsg:4326",
    )
    return joint_gdf
