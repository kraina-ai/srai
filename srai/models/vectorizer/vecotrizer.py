"""
Vectorizer module.

This module contains implementation of vectorizer. Various embeddings of geolocation can be selected
"""


class Vectorizer:
    """
    Vectorizer module.

    H3 regionalizer is used.
    Should return datasets.Dataset() generated from DF with vectors -> \
        (geo embedding + additional features) and target labels

    Hence, needed:
     - resolution
     - embedder type
     - not geo numerical column names -> to standarize
     - not geo non-numerical column names -> to encoding
    """

    def __init__(self) -> None:
        """
        Initialization of vectorizer.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def _get_embeddings(self) -> None:
        """Method to get embeddings from geolocation."""
        raise NotImplementedError

    def get_dataset(self) -> None:
        """
        Method to return datasets.Dataset.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
