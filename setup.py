"""Install galaxy2galaxy."""

from setuptools import find_packages
from setuptools import setup
from io import open

# read the contents of the README file
with open('README.md', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='galaxy2galaxy',
    description='Galaxy2Galaxy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ML4Astro Contributors',
    url='http://github.com/ml4astro/galaxy2galaxy',
    license='MIT',
    packages=find_packages(),
    package_data={'': ['*.sql', '*.fits'],},
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
        'dopamine_rl==3.0.1',
        'tensor2tensor==1.15.4',
        'tensorflow-hub',
        'tensorflow-probability<0.8',
        'tensorflow-gan',
        'fits2hdf',
        'unagi>=0.1.1'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.15.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.13.1'],
        'tensorflow-hub': ['tensorflow-hub>=0.1.1'],
        'galsim': ['galsim' 'galflow'],
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
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    keywords='astronomy machine learning',
    use_scm_version=True,
    include_package_data=True,
    setup_requires=['setuptools_scm'],
)
