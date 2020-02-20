import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


def get_long_description():
    return open("README.md", "r").read()


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f:
            requirements.append(line.strip())
    return requirements


setup(
    name="gomoku-a3c",
    version="0.0.0",
    keywords=("python", "tensorflow", "deeplearning", "gomoku"),
    description="Gomoku AI with A3C",
    long_description=get_long_description(),
    license="MIT",

    url="https://github.com/SigureMo/gomoku-a3c",
    author="SigureMo",
    author_email="sigure_mo@163.com",

    platforms="any",
    install_requires=get_requirements(),

    scripts=[],
    ext_modules=cythonize(Extension(
        'game',
        sources=['game.pyx'],
        language='c',
        include_dirs=[np.get_include()],
        library_dirs=[],
        libraries=[],
        extra_compile_args=[],
        extra_link_args=[]
    )),
    cmdclass={'build_ext': build_ext}
)
