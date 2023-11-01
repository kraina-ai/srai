"""Constants for hex2vec tests."""

ENCODER_SIZES = [32, 16]
TRAINER_KWARGS = {"max_epochs": 1, "accelerator": "cpu", "deterministic": True}

PREDEFINED_TEST_CASES = [
    {
        "test_case_name": "wro_9",
        "geocoding_name": "Wrocław, Poland",
        "root_region_index": "891e2040887ffff",
        "h3_res": 9,
        "radius": 7,
        "seed": 42,
    },
    {
        "test_case_name": "poz_8",
        "geocoding_name": "Poznań, Poland",
        "root_region_index": "881e24a125fffff",
        "h3_res": 8,
        "radius": 6,
        "seed": 5555,
    },
]
