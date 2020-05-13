# Copyright (c) 2020, NVIDIA CORPORATION.
from setuptools import find_packages, setup


setup(
    name='xbb_tools',
    version='0.1',
    author='RAPIDS',
    packages=[
        'xbb_tools',
        'xbb_tools/text_vectorizers'
    ],
)