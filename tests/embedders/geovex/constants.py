"""Constants for geovex tests."""

from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER

EMBEDDING_SIZE = 32
TRAINER_KWARGS = {"max_epochs": 1, "accelerator": "cpu", "deterministic": True}

PREDEFINED_TEST_CASES = [
    {
        "test_case_name": "wro_9",
        "root_region_index": "891e2040887ffff",
        "region_gen_radius": 2,
        "model_radius": 2,
        "seed": 42,
        "tags": {
            t: HEX2VEC_FILTER[t]
            for t in [
                "building",
                "amenity",
            ]
        },
        "convolutional_layer_size": 32,
        "num_layers": 2,
    },
    {
        "test_case_name": "AL_10",
        "root_region_index": "8a44ed664c27fff",
        "region_gen_radius": 3,
        "model_radius": 3,
        "seed": 5555,
        "tags": {
            t: HEX2VEC_FILTER[t]
            for t in [
                "building",
                "amenity",
            ]
        },
        "convolutional_layer_size": 32,
        "num_layers": 2,
    },
]
