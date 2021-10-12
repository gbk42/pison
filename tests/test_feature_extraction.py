"""Tests for feature_extraction functions and classes"""
import numpy as np
import pytest
import pandas as pd
from ..feature_extraction import FeatureExtractor, process_dataframe


@pytest.mark.parametrize(
    ["extractor", "data", "answer"],
    [
        (FeatureExtractor("mean", np.mean), np.arange(10), {"mean": 4.5}),
        (
            FeatureExtractor("max", np.max, segment_length=4, interval=3),
            np.arange(13),
            {"max.0.4": 3, "max.3.7": 6, "max.6.10": 9},
        ),
        (
            FeatureExtractor(
                "argmax", np.argmax, segment_length=4, interval=3, pad_remainder="zero"
            ),
            np.arange(7),
            {"argmax.0.4": 3, "argmax.3.7": 3, "argmax.6.7": 0},
        ),
    ],
)
def test_feature_extractors(extractor, data, answer):
    """Test the feature extractor wrapper class"""
    result = extractor(data)
    assert result == answer


def test_process_dataframe():
    multi_index = [
        np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]),
        np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
    ]
    test_frame = pd.DataFrame(
        data=np.arange(24).reshape(12, 2), columns=["a", "b"], index=multi_index
    )
    extractors = {
        "a": FeatureExtractor("mean", np.mean),
        "b": [FeatureExtractor("mean", np.mean), FeatureExtractor("max", np.max)],
    }
    result = process_dataframe(test_frame, extractors, normalize=False)
    assert result.to_dict() == {
        "a.mean": {
            (0.0, 1.0): 2.0,
            (0.0, 2.0): 8.0,
            (1.0, 1.0): 14.0,
            (1.0, 2.0): 20.0,
        },
        "b.mean": {
            (0.0, 1.0): 3.0,
            (0.0, 2.0): 9.0,
            (1.0, 1.0): 15.0,
            (1.0, 2.0): 21.0,
        },
        "b.max": {
            (0.0, 1.0): 5.0,
            (0.0, 2.0): 11.0,
            (1.0, 1.0): 17.0,
            (1.0, 2.0): 23.0,
        },
    }
