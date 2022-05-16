from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
import numpy as np

extensions = [
    Extension(
        "cyutils.analysis",
        ["cyutils/analysis.pyx"],
        include_dirs=[np.get_include()],
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "cyutils.cyzers",
        ["cyutils/cyzers.pyx"],
        include_dirs=[np.get_include()],
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "cystratbase",
        sources=["cystratbase.pyx"],
    ),
    Extension(
        "features",
        ["features.pyx"],
        include_dirs=[np.get_include()],
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(extensions, annotate=True,),
    zip_safe=False,
)
