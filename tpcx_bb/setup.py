# Copyright (c) 2020, NVIDIA CORPORATION.
from setuptools import find_packages, setup

qnums = [str(i).zfill(2) for i in range(1,31)]

setup(
    name='xbb_tools',
    version='0.2',
    author='RAPIDS',
    packages=[
        'benchmark_runner',
        'xbb_tools',
        'xbb_tools/text_vectorizers',
        'queries'
    ] + [
        f'queries/q{qnum}' for qnum in qnums
    ],
    package_data={'benchmark_runner': ['benchmark_config.yaml']},
    include_package_data=True,
)