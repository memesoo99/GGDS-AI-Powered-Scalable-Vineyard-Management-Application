#!/usr/bin/env python3
import os
from typing import List

import setuptools

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
    """Load requirements from a file.
    >>> _load_requirements(_PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        if ln.startswith("http") or "@http" in ln:
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grape-thinning",
    version="1.0.0",
    author="Jisoo Kim",
    author_email="genniferk1234@gmail.com",
    description="Wrapper for Grape thinning System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/memesoo99/grape-thinning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    setup_requires=['numpy>=1.22.1'],
    install_requires=_load_requirements(_PATH_ROOT),
    extras_require={"all": ["psutil"]},
)