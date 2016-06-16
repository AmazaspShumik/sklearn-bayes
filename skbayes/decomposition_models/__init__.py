import subprocess
subprocess.call(['python', 'setup.py', 'build_ext'])
from gibbs_lda_cython import GibbsLDA


__all__ = ['GibbsLDA']