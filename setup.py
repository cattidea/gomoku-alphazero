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


extensions = [
    Extension(
        "board",
        sources=["board.pyx"],
        language="c",
        include_dirs=[np.get_include()],
    ),
    Extension(
        "mcts",
        sources=["mcts.pyx"],
        language="c",
        include_dirs=[np.get_include()],
    ),
]


setup(
    name="gomoku-alphazero",
    version="0.0.0",
    keywords=("python", "tensorflow", "deeplearning", "gomoku"),
    description="Gomoku AI with AlphaZero",
    long_description=get_long_description(),
    license="MIT",
    url="https://github.com/SigureMo/gomoku-alphazero",
    author="SigureMo",
    author_email="sigure_mo@163.com",
    platforms="any",
    install_requires=get_requirements(),
    scripts=[],
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": build_ext},
)
