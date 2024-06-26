#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = ['pandas', 'numpy']

test_requirements = ['pytest>=3', ]

setup(
    author="Conor Walsh",
    author_email='conorwalsh206@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Toolkit for calibration of machine learning classifiers.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='calibra',
    name='calibra',
    packages=find_packages(include=['calibra', 'calibra.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/conorwalsh99/calibra',
    version='0.3.2',
    zip_safe=False,
)

# + '\n\n' + history