#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'networkx',
    'tqdm'
]

setup_requirements = [
    'pytest-runner'
]

test_requirements = [
    'pytest',
    'pandas',
    'faker',
]

setup(
    name='disamby',
    version='0.2.1',
    description="Python package to carry out entity disambiguation based on "
                "string matching",
    long_description=readme + '\n\n' + history,
    author="Luca Verginer",
    author_email='luca@verginer.eu',
    url='https://github.com/verginer/disamby',
    packages=find_packages(include=['disamby']),
    entry_points={
        'console_scripts': [
            'disamby=disamby.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords=['disamby', 'disambiguation'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
