# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from os import environ
from pathlib import Path
from setuptools import setup, find_packages
import site
import sys
import time

# Workaround for enabling editable user pip installs
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Version
here = Path(__file__).absolute().parent
version_data = {}
with open(here.joinpath("recommenders", "__init__.py"), "r") as f:
    exec(f.read(), version_data)
version = version_data.get("__version__", "0.0")

# Get the long description from the README file
with open(here.joinpath("recommenders", "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

HASH = environ.get("HASH", None)
if HASH is not None:
    version += ".post" + str(int(time.time()))

install_requires = [
    "category-encoders>=2.6.0,<3",  # requires packaging
    "cornac>=1.15.2,<=2.2.2;python_version<='3.8'",
    "cornac>=2.3.0,<3;python_version>='3.9'",  # requires packaging, tqdm
    "hyperopt>=0.2.7,<1",
    "lightgbm>=4.0.0,<5",
    "locust>=2.12.2,<3",  # requires jinja2
    "memory-profiler>=0.61.0,<1",
    "nltk>=3.8.1,<4",  # requires tqdm
    "numpy>=1.26.4;python_version>='3.12'", # https://stackoverflow.com/a/77364602/4505998
    "notebook>=6.5.5,<8",  # requires ipykernel, jinja2, jupyter, nbconvert, nbformat, packaging, requests
    "numba>=0.57.0,<1",
    "pandas>2.0.0,<3.0.0",  # requires numpy
    "pandera[strategies]>=0.6.5,<0.18;python_version<='3.8'",  # For generating fake datasets
    "pandera[strategies]>=0.15.0;python_version>='3.9'",
    "retrying>=1.3.4,<2",
    "scikit-learn>=1.2.0,<2",  # requires scipy, and introduce breaking change affects feature_extraction.text.TfidfVectorizer.min_df
    "scikit-surprise>=1.1.3",
    "seaborn>=0.13.0,<1",  # requires matplotlib, packaging
    "statsmodels<=0.14.1;python_version<='3.8'",
    "statsmodels>=0.14.4;python_version>='3.9'",
    "transformers>=4.27.0,<5",  # requires packaging, pyyaml, requests, tqdm
]

# shared dependencies
extras_require = {
    "gpu": [
        "fastai>=2.7.11,<3",
        "numpy<1.25.0;python_version<='3.8'",
        "nvidia-ml-py>=11.525.84",
        "spacy<=3.7.5;python_version<='3.8'",
        "tensorflow>=2.8.4,!=2.9.0.*,!=2.9.1,!=2.9.2,!=2.10.0.*,<2.16; python_version<='3.8'",  # Fixed TF due to constant security problems and breaking changes #2073
        "tensorflow~=2.16; python_version>'3.8'",  # Version needed for python 3.12
        "tf-slim>=1.1.0",  # No python_requires in its setup.py
        # Use keras 2 instead of keras 3
        "tf-keras~=2.15; python_version<='3.8'", # Not compatible with py3.12
        "tf-keras~=2.16; python_version>'3.8'",
        "torch>=2.0.1,<3",
    ],
    "spark": [
        "pyarrow>=10.0.1",
        "pyspark>=3.3.0,<=4",
    ],
    "dev": [
        "black>=23.3.0",
        "pytest>=7.2.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.10.0",  # for access to mock fixtures in pytest
    ],
}
# For the brave of heart
extras_require["all"] = list(set(sum([*extras_require.values()], [])))

# The following dependencies need additional testing
extras_require["experimental"] = [
    # xlearn requires cmake to be pre-installed
    "xlearn==0.40a1",
    # VW C++ binary needs to be installed manually for some code to work
    "vowpalwabbit>=8.9.0,<9",
    # nni needs to be upgraded
    "nni==1.5",
    "pymanopt>=0.2.5",
    "lightfm>=1.17,<2",
]

# The following dependency can be installed as below, however PyPI does not allow direct URLs.
# Temporary fix for pymanopt, pypi version does not work with TF2
# pip install -r requirements-external.txt

setup(
    name="recommenders",
    version=version,
    description="Recommenders - Python utilities for building recommendation systems",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/recommenders-team/recommenders",
    project_urls={
        "Documentation": "https://recommenders-team.github.io/recommenders/intro.html",
        "Wiki": "https://github.com/recommenders-team/recommenders/wiki",
    },
    author="Recommenders contributors",
    author_email="recommenders-technical-discuss@lists.lfaidata.foundation",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
    ],
    extras_require=extras_require,
    keywords="recommendations recommendation recommenders recommender system engine "
    "machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={"recommenders": "recommenders"},
    python_requires=">=3.6",
    packages=find_packages(
        where=".",
        exclude=["contrib", "docs", "examples", "scenarios", "tests", "tools"],
    ),
    setup_requires=[
        "numpy>=1.15",
    ],
)
