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
    test_frame = pd.DataFrame(data=np.arange(9).reshape(3, 3), columns=["a", "b", "c"])
    extractors = {
        "a": FeatureExtractor("mean", np.mean),
        "b": [FeatureExtractor("mean", np.mean), FeatureExtractor("max", np.max)],
        "c": FeatureExtractor("min", np.min),
    }
    result = process_dataframe(test_frame, extractors)
    assert result == {"a.mean": 3.0, "b.mean": 4.0, "b.max": 7, "c.min": 2}
