
import glob
import pathlib

import setuptools
from setuptools import find_packages, setup
from Cython import Build


cythonize_files = glob.glob('src/**/*.pyx', recursive=True)

PKG_NAME = 'miniumap'
PWD = pathlib.Path(__file__).parent.resolve()

install_requires = []
with open('requirements.txt', 'r') as reader:
    for line in reader:
        if not line.startswith('#'):
            install_requires.append(line.strip())

version = '0.1.0'

with open('Readme.md', 'r') as reader:
    long_description = reader.read()

setup(
    name=PKG_NAME,
    version=version,
    author='YUAN Ruihong, FU Qichen',
    author_email='ruihong.19@intl.zju.edu.cn, qichen.19@intl.zju.edu.cn',
    description='miniumap -- a Python implementation of Uniform Manifold Approximation Projection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Programming Language :: C",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: R",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(
        where='src',
        include=['*']
    ),
    package_dir={"": 'src'},
    install_requires=install_requires,
    ext_modules=Build.cythonize(cythonize_files,
                                compiler_directives={'language_level' : "3"},
                                annotate=True)
)
