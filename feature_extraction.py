"""Module containing feature extraction framework functions"""

import attr
from numpy.lib.function_base import interp
import pandas as pd
import numpy as np
from typing import Literal, Optional, Callable, Iterator, List, Tuple, Dict, Union


@attr.s(auto_attribs=True)
class FeatureExtractor:

    annotation: str
    extractor: Callable[[np.ndarray], float]
    segment_length: Optional[int] = attr.ib(default=None)
    interval: Optional[int] = attr.ib(default=None)
    pad_remainder: Optional[Literal["zero", "mean"]] = attr.ib(default=None)

    def __attrs_post_init__(self):
        """If segment length is specified but no interval is specified, then set interval to the segment length"""
        if self.interval is None and self.segment_length is not None:
            self.interval = self.segment_length

    def __call__(self, data: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """Execute this extractor's extractor and format the data into a DataFrame with appropriate columns"""
        features = {}
        if len(prefix):
            full_annotation = f"{prefix}." + self.annotation
        else:
            full_annotation = self.annotation

        if self.segment_length is None:
            features[full_annotation] = self.extractor(data)
        else:
            for (start, stop), segment in self.segment_observations(data):
                features[f"{full_annotation}.{start}.{stop}"] = self.extractor(segment)
        return features

    def segment_observations(self, data: np.ndarray) -> Iterator[np.ndarray]:
        """Segment a time series vector into chunks of `segment_length` length taken at a frequency of `interval` samples

        If there are remaining samples not evenly divided in the segmentation, they can be ignored or included with a mean pad

        Args:
            data: The data to segment as a numpy array - this may be any type of sensor data
            segment_length: number of samples for each segment
            overlap: The number of samples between segments
            pad_remainder: Set to true, this pads the remainder samples with the mean of the whole vector

        Returns:
            processed data as a matrix of n_segments x segment_length
        """
        if self.segment_length is None:
            yield data
        else:
            start = 0
            while start < len(data):
                segment = data[start : start + self.segment_length]
                if start + self.segment_length >= len(data):
                    if self.pad_remainder:
                        n_pad = self.segment_length - len(segment)
                        segment = np.concatenate(
                            (
                                segment,
                                np.zeros(n_pad)
                                if self.pad_remainder == "zeros"
                                else np.ones(n_pad) * np.mean(data),
                            )
                        )
                    else:
                        break
                yield (start, min(len(data), start + self.segment_length)), segment
                start += self.interval


def process_dataframe(
    dataset: pd.DataFrame,
    column_extractors: Dict[str, Union[FeatureExtractor, List[FeatureExtractor]]],
):
    if not set(column_extractors.keys()).issubset(set(dataset.columns)):
        raise ValueError("Extractors must target columns of the dataframe provided")
    # Apply the extractors to each column of data in the dataframe
    all_features = {}
    for column, extractors in column_extractors.items():
        if not isinstance(extractors, list):
            extractors = [extractors]
        for extractor in extractors:
            all_features.update(extractor(dataset.loc[:, column], column))
    return all_features
