"""Constants for s2vec tests."""

from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER

PREDEFINED_TEST_CASES = [
    {
        "test_case_name": "wro_s2_14",
        "root_regions_tokens": ["470fc275", "470fc277"],
        "seed": 42,
        "tags": {
            t: HEX2VEC_FILTER[t]
            for t in [
                "building",
                "amenity",
            ]
        },
        "img_res": 14,
        "patch_res": 18,
        "num_heads": 2,
        "encoder_layers": 6,
        "decoder_layers": 2,
        "embedding_dim": 256,
        "decoder_dim": 128,
        "mask_ratio": 0.75,
        "dropout_prob": 0.2,
    },
]
