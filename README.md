# Installation and Setup

While the `requirements.txt` file specifies all dependencies for a `python 3.9.1` virtual environment, the version of `pip`, `wheel`, and `setuptools` must be upgrade prior to installation or `kiwisolver` will fail during the build process.

To set up the environment, first run:

`python -m venv .venv`

Then upgrade packaging tools:

`pip install -U wheel setuptools pip`

Then install requirements:

`pip install -r requirements.txt`

Brief tests for data processing can be run with:

`pytest tests`


# Writeup and Notebooks

The writeup for the project is located in writeup.md and there are three notebooks:

1. `dynamics_analysis.ipynb` - Study done on only dynamics data for feature extraction and clustering
2. `emg_analysis.ipynb` - Study done on only EMG data for feature extraction and clustering
3. `full_sensor_analysis.ipynb` - Full feature clustering and classification



###  Thanks!