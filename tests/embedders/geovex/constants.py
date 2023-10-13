"""Constants for hex2vec tests."""
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER

EMBEDDING_SIZE = 32
TRAINER_KWARGS = {"max_epochs": 1, "accelerator": "cpu"}

PREDEFINED_TEST_CASES = [
    {
        "test_case_name": "wro_9",
        "geocoding_name": "Wrocław, Poland",
        "root_region_index": "891e2040887ffff",
        "region_gen_radius": 12,
        "h3_res": 9,
        "model_radius": 3,
        "seed": 42,
        "tags": HEX2VEC_FILTER,
    },
]
