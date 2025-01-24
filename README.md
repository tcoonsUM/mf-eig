# MF-EIG
A multi-fidelity estimator of the expected information gain (EIG) for Bayesian optimal experimental design (OED).

## Paper
The pre-print is available on arXiv [here](https://arxiv.org/abs/2501.10845).

## Description
The EIG is an information-theoretic design criterion popularly used in Bayesian OED. However, its computation for nonlinear models is not tractable, and sampling-based estimators such as the nested Monte Carlo (NMC) estimator must be used in many modeling scenarios. Since such estimators require numerous evaluations of a forward predictive model of the given experiment, NMC estimators are often prohibitively expensive for high-fidelity models. We aim to expedite EIG computation by enabling multi-fidelity estimation via approximate control variate (ACV) estimators. In order to do so, a novel reparameterization of the EIG is presented in our paper, changing its expectations to be independent of the data models, a requirement for compatibility with ACV. When well-correlated and cheap low-fidelity models are available, the MF-EIG estimator can produce multiple orders-of-magnitude error reduction with respect to the original single-fidelity NMC estimator.

## Authors
[Thomas E. Coons](https://sites.google.com/umich.edu/tcoons/home) and [Xun Huan](https://uq.engin.umich.edu/)

## Requirements
This project utilizes the MXMCPy framework from [this paper](https://www.sciencedirect.com/science/article/pii/S0021999121007774) and [Python package](https://github.com/nasa/MXMCPy). In addition to the usual numpy, scipy, and matplotlib installations, these scripts assume you have MXMCPy and its own requirements installed. To install MXMCPy, you can run:
```
pip install mxmcpy
```

## Tutorial/Usage
See our [Basic Tutorial](https://github.com/tcoonsUM/mf-eig/blob/main/Tutorials/basic_tutorial.ipynb) for the basic usage of the MF-EIG estimator for additive independent noise.

## Examples
See [Final Paper Figures](https://github.com/tcoonsUM/mf-eig/tree/main/Final%20Paper%20Figures) to reproduce the results of our paper. Most data that require many CPU-hours of compute time are simply loaded in their respective scripts.

Tutorials will also be provided to run your own mf-eig optimization and analyses for general ensembles of utility models, for example to optimize $\text{N}_{\text{in}}$ for each problem and run the estimator over many trials.
