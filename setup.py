# License: Apache-2.0
# import os
import setuptools
import numpy
from Cython.Build import cythonize
from gators import __version__

with open('README.md', 'r') as f:
    long_description = f.read()

extensions = [
    setuptools.Extension(
        'imputer',
        ['gators/imputers/imputer.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'encoder',
        ['gators/encoders/encoder.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'data_cleaning',
        ['gators/data_cleaning/data_cleaning.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'binning',
        ['gators/binning/binning.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'clipping',
        ['gators/clipping/clipping.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'scaler',
        ['gators/scalers/scaler.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'feature_gen',
        ['gators/feature_generation/feature_gen.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'feature_gen_dt',
        ['gators/feature_generation_dt/feature_gen_dt.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'feature_gen_str',
        ['gators/feature_generation_str/feature_gen_str.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    )
]

setuptools.setup(
    name='gators',
    version=__version__,
    url='https://paypal.github.io/gators/',
    author='The Gators Development Team',
    options={'bdist_wheel': {'universal': True}},
    long_description=long_description,
    long_description_content_type="text_markdown",
    description='Model building and Model deployment library',
    maintainer='Charles Poli',
    packages=setuptools.find_packages(exclude=['examples', 'doc']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache-2 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='Apache-2.0 Software License',
    setup_requires=['numpy', 'Cython'],
    install_requires=[
        'scipy',
        'scikit-learn',
        'seaborn',
        'pandas',
        'numpy',  #<1.21.0',
        'lightgbm',
        'xgboost>=0.90',
        'treelite>=0.93',
        'treelite-runtime>=0.93',
        'dill',
        ],
    extras_require={
        'dev': [
            'docutils==0.16.0',
            'pytest-cov',
            'pylama>=7.6.5',
            'ipykernel',
            'jupyter',
            'numpydoc',
            'sphinx>=3.3.0',
            'nbsphinx>=0.8.0',
            'pydata_sphinx_theme',
            'pandoc',
            'tox',
            'tox-wheel',
            'black',
            'isort',
        ],
        'koalas':[
            'pyspark>=2.4.3',
            'koalas<3.2',
        ],
        'dask': [
            'distributed',
            'dask',
        ]
    },
    package_data={'gators': [
        '*.pyx',
    ]},
    include_package_data=True,
    ext_modules=cythonize(extensions, compiler_directives={
                          'language_level': "3"}),
    zip_safe=False,
)
