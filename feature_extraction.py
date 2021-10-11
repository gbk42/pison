"""Module containing feature extraction framework functions"""

import itertools
import math
from typing import (Callable, Dict, Iterator, List, Literal, Optional,
                    Sequence, Tuple, Union)

import attr
import numpy as np
import pandas as pd
import scipy.signal


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


def zero_crossings(data_segment: np.ndarray) -> float:
    """Count the number of zero crossings in a data segment

    Adapted from https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    """
    return len(np.where(np.diff(np.sign(data_segment)))[0])


def mean_crossings(data_segment: np.ndarray) -> float:
    """Count the number of mean crossings in the segment

    This feature captures transitions above and below a mean value, like rotations about an axis or changes in
    direction
    """
    return len(np.where(np.diff(np.sign(data_segment - data_segment.mean())))[0])


def spectral_entropy(nfft: int, noverlap: int) -> Callable:
    """Determine the spectral entropy, a feature that captures the total bandwidth of the signal

    While this function can be applied within segments, it defines the nfft and noverlap
    again to specify the length of the FFT (see scipy.signal.welch)

    Args:
        nfft: the number of points for the welch periodogram
        noverlap: the number of samples to overlap the FFT by

    Returns:
        The entropy of the power spectrum, calculated just as a probability distribution would be
    """

    def func(data_segment: np.ndarray) -> float:
        """Parametrized executor for spectral_energy function"""
        freq, pxx = scipy.signal.welch(data_segment, nperseg=nfft, noverlap=noverlap)
        pxx = pxx / np.sum(pxx)
        entropy = -np.sum(pxx * np.log(pxx))
        return entropy

    return func


def peak_frequency(data_segment: np.ndarray) -> float:
    """Get the bin of the strongest frequency in the fft of a given segment"""
    return np.argmax(np.abs(np.fft.fft(data_segment)[: len(data_segment) // 2 + 1]))


def peak_z_score(data_segment: np.ndarray) -> float:
    """Calculate the z-score of the largest value in a data segment"""
    return (np.max(data_segment) - np.median(data_segment)) / np.std(data_segment)


def process_dataframe(
    dataset: pd.DataFrame,
    column_extractors: Dict[str, Union[FeatureExtractor, List[FeatureExtractor]]],
    normalize: bool = True,
):
    # Apply the extractors to each column of data in the dataframe
    results = pd.DataFrame()
    for (body, rep), observation in dataset.groupby(level=(0, 1)):
        obs_features = {}
        for features, extractors in column_extractors.items():
            feature_prefix = (
                ".".join(features) if isinstance(features, list) else features
            )
            if not isinstance(extractors, list):
                extractors = [extractors]
            for extractor in extractors:
                obs_features.update(
                    extractor(observation.loc[:, features].values, feature_prefix)
                )
        results = results.append(
            {**{"body_label": body, "rep": rep}, **obs_features}, ignore_index=True
        )
    results = results.set_index(["body_label", "rep"]).sort_index()

    if normalize:
        results = (results - results.mean()) / results.std()
        results.dropna(axis=1, inplace=True)
    return results


def quaternion_to_euler_angle(q_data: pd.DataFrame) -> pd.DataFrame:
    """Convert a frame of quaternions to euler angles for visualization

    Adapted from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/

    Euler angles do not necessarily provide a perfect representation of quaternions, but are an easy way to
    plot motion over time in 3D

    Args:
        q_data: Quaternion dataframe with columns qw, qx, qy, qz

    Returns:
        Dataframe with columns roll, pitch, yaw (x, y, z rotation)
    """
    eulers = pd.DataFrame(columns=["roll", "pitch", "yaw"])
    for idx, sample in q_data.iterrows():
        eulers = eulers.append(
            _quaternion_sample_to_euler(sample.qw, sample.qx, sample.qy, sample.qz),
            ignore_index=True,
        )
    return eulers


def _quaternion_sample_to_euler(
    qw: float, qx: float, qy: float, qz: float
) -> Dict[str, float]:
    """Convert a quaternion into euler angles (roll, pitch, yaw)

    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)

    Args:
        qw: W component (real) of quaternion
        qx: X component (imaginary) of quaternion
        qy: Y component (imaginary) of quaternion
        qz: Z component (imaginary) of quaternion
    """
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_z = math.atan2(t3, t4)
    return {"roll": roll_x, "pitch": pitch_y, "yaw": yaw_z}
