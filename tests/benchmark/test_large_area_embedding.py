"""Test for checking memory usage of the whole pipeline."""

import random
from random import choice
from string import ascii_lowercase, digits

import geopandas as gpd
import numpy as np
import pytest
from shapely import box

from srai.constants import FEATURES_INDEX
from srai.embedders import ContextualCountEmbedder
from srai.h3 import ring_buffer_h3_regions_gdf
from srai.joiners import IntersectionJoiner
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, S2Regionalizer


@pytest.mark.parametrize("concatenate_vectors", [False, True])  # type: ignore
def test_large_area_embedding(concatenate_vectors: bool) -> None:
    """Test if large area embedding can be calculated on a single machine."""
    # TODO: increase later
    H3_RESOLUTION = 5
    S2_RESOLUTION = 9
    H3_DISTANCE = 10

    chars = ascii_lowercase + digits
    columns = ["".join(choice(chars) for _ in range(8)) for _ in range(100)]
    values = ["".join(choice(chars) for _ in range(8)) for _ in range(100)]

    area = gpd.GeoDataFrame(geometry=[box(5.818355, 46.037418, 24.363277, 52.769854)], crs=4326)

    h3_regions = H3Regionalizer(resolution=H3_RESOLUTION).transform(area)
    print(f"H3 regions: {len(h3_regions)}")
    buffered_h3_regions = ring_buffer_h3_regions_gdf(h3_regions, H3_DISTANCE)
    print(f"Buffered H3 regions: {len(buffered_h3_regions)}")

    s2_regions = S2Regionalizer(resolution=S2_RESOLUTION).transform(area)
    print(f"S2 regions: {len(s2_regions)}")

    data = np.full((len(s2_regions), len(columns)), None)
    for i in range(len(s2_regions)):
        data[i, random.randint(0, len(columns) - 1)] = random.choice(values)

    s2_regions[columns] = data
    s2_regions.index.rename(FEATURES_INDEX, inplace=True)

    joint = IntersectionJoiner().transform(buffered_h3_regions, s2_regions)
    print(f"Joint: {len(joint)}")

    embeddings = ContextualCountEmbedder(
        neighbourhood=H3Neighbourhood(),
        neighbourhood_distance=H3_DISTANCE,
        count_subcategories=True,
        concatenate_vectors=concatenate_vectors,
    ).transform(buffered_h3_regions, s2_regions, joint)

    assert len(embeddings) == len(buffered_h3_regions)
