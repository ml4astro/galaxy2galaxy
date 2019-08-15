"""Install galaxy2galaxy."""

from setuptools import find_packages
from setuptools import setup


setup(
    name='galaxy2galaxy',
    version='0.0.1rc',
    description='Galaxy2Galaxy',
    author='ML4Astro Contributors',
    url='http://github.com/ml4astro/galaxy2galaxy',
    license='MIT',
    packages=find_packages(),
    scripts=[
        'galaxy2galaxy/bin/g2g-trainer',
        'galaxy2galaxy/bin/g2g-datagen',
        'galaxy2galaxy/bin/g2g-exporter',
    ],
    install_requires=[
        'six',
        'scipy',
        'numpy',
        'astropy',
        'tensor2tensor',
        'tensorflow',
        'tensorflow-hub',
        'tensorflow-datasets',
        'tensorflow-probability',
        'tensorflow-gan',
        'fits2hdf',
        'galsim',
        'unagi'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.13.1'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.13.1'],
        'tensorflow-hub': ['tensorflow-hub>=0.1.1'],
        'tests': [
            'absl-py',
            'pytest>=3.8.0',
            'mock',
            'pylint',
            'jupyter',
            'gsutil',
            'matplotlib',
            # Need atari extras for Travis tests, but because gym is already in
            # install_requires, pip skips the atari extras, so we instead do an
            # explicit pip install gym[atari] for the tests.
            # 'gym[atari]',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    dependency_links=['git+https://github.com/EiffL/unagi.git@b581e84624c04de4346f944b822441835cd1880d#egg=<unagi>-0.1'],
    keywords='astronomy machine learning',
)
