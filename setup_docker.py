# License: Apache-2.0
import setuptools
import numpy
from Cython.Build import cythonize
from gators import __version__
import pyximport
pyximport.install()


extensions = [
    setuptools.Extension(
        'imputer',
        ['/gators/gators/imputers/imputer.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'encoder',
        ['/gators/gators/encoders/encoder.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'data_cleaning',
        ['/gators/gators/data_cleaning/data_cleaning.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'binning',
        ['/gators/gators/binning/binning.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'clipping',
        ['/gators/gators/clipping/clipping.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'scaler',
        ['/gators/gators/scalers/scaler.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'feature_gen',
        ['/gators/gators/feature_generation/feature_gen.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'feature_gen_dt',
        ['/gators/gators/feature_generation_dt/feature_gen_dt.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    setuptools.Extension(
        'feature_gen_str',
        ['/gators/gators/feature_generation_str/feature_gen_str.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    )
]

setuptools.setup(
    name='gators',
    version=__version__,
    author='Simility Data Team',
    author_email='cpoli@paypal.com',
    options={'bdist_wheel': {'universal': True}},
    description='Model building and Model deployment library',
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
        'setuptools>=41.0.0',
        'numpy==1.19.5',
        'requests>=2.23.0',
        'tqdm>=4.43.0',
        'scipy>=1.5.2',
        'Cython>=0.29.21',
        'dill>=0.3.1.1',
        'scikit-learn==0.23.1',
        'seaborn>=0.11.0',
        'pandas>=0.25.3<1.2',
        'treelite>=0.93',
        'treelite-runtime>=0.93',
        'xgboost>=0.90',
        'pyDOE>=0.3.8',
        'scikit-optimize>=0.8.1',
        'emcee>=3.0.2',
        'pyspark>=2.4.3',
        'koalas',
        'hyperopt>=0.2.5',
        'Lightgbm',
        'pytest>=5.0.0',
        'pytest-cov>=2.6.0',
        'pylama>=7.6.5',
        'sphinx>=3.3.0',
        'nbsphinx>=0.8.0',
        'pydata_sphinx_theme',
        'ipykernel',
        'jupyter',
        'numpydoc',
        'tox',
        'tox-wheel',
    ],
    package_data={'gators': [
        '*.c',
    ]},
    include_package_data=True,
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}),
    zip_safe=False,
)
