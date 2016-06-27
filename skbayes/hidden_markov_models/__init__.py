'''

   Bayesian Hidden Markov Model (Beal 2003)
   ========================================
   Bernoulli emission probs: VBBernoulliHMM
   Gaussian emission probs: VBGaussianHMM
   Multinoulli emission probs: VBMultinoulliHMM 
'''

from .hmm import VBPoissonHMM ,VBGaussianHMM,VBBernoulliHMM

__all__ = ['VBBernoulliHMM','VBGaussianHMM','VBPoissonHMM']