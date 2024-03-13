# License: Apache-2.0
# import os
import setuptools
import numpy
from Cython.Build import cythonize
from gators import __version__

with open("README.md", "r") as f:
    long_description = f.read()

extensions = [
    setuptools.Extension(
        "imputer",
        ["gators/imputers/imputer.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "encoder",
        ["gators/encoders/encoder.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "data_cleaning",
        ["gators/data_cleaning/data_cleaning.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "binning",
        ["gators/binning/binning.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "clipping",
        ["gators/clipping/clipping.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "scaler",
        ["gators/scalers/scaler.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "feature_gen",
        ["gators/feature_generation/feature_gen.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "feature_gen_dt",
        ["gators/feature_generation_dt/feature_gen_dt.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    setuptools.Extension(
        "feature_gen_str",
        ["gators/feature_generation_str/feature_gen_str.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
]

setuptools.setup(
    name="gators",
    version=__version__,
    url="https://paypal.github.io/gators/",
    author="The Gators Development Team",
    options={"bdist_wheel": {"universal": True}},
    long_description_content_type="text/markdown",
    long_description=long_description,
    description="Model building and model scoring library",
    maintainer="Charles Poli",
    packages=setuptools.find_packages(exclude=["examples", "doc"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="License :: OSI Approved :: Apache Software License v2.0",
    setup_requires=["numpy", "Cython"],
    install_requires=[
        "scikit-learn>=1.0.1",
        "pandas",
        "pyarrow",
        "lightgbm",
        "xgboost",
        # "treelite",
        # "treelite-runtime",
        "dill",
    ],
    extras_require={
        "dev": [
            "docutils",
            "pytest",
            "pytest-cov",
            "coverage",
            "pylama",
            "jupyter",
            "numpydoc",
            "sphinx",
            "nbsphinx",
            "pydata_sphinx_theme",
            "pandoc",
            "tox",
            "tox-wheel",
            "black",
            "isort",
            "pyspark",
            "distributed",
            "dask",
            "numpy<=1.24.6",
        ],
        "pyspark": [
            "pyspark",
        ],
        "dask": [
            "fsspec>=0.3.3",
            "distributed",
            "dask",
        ],
    },
    package_data={
        "gators": [
            "*.pyx",
        ]
    },
    include_package_data=True,
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)
