import numpy as np
import os
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension("gibbs_lda_cython",
                         sources=["gibbs_lda_cython.c"],
                         include_dirs=[np.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())



