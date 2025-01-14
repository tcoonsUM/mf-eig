# mf-eig
A multi-fidelity estimator of the expected information gain (EIG) for Bayesian optimal experimental design (OED).

## Abstract
In many engineering and scientific applications, physical experiments are vital but costly and time-demanding. Therefore, it
is important to find experimental design conditions that maximize the value of these experiments while minimizing the associated
costs. Bayesian optimal experimental design (OED) provides a rigorous statistical framework for seeking the ideal experimental
design by leveraging a mathematical model of the experiment. When the goal of a given experiment is model-parameter infer-
ence, this process generally requires the use of a nested Monte Carlo (NMC) estimator to approximate the Expected Information
Gain (EIG) of each experimental design. Using NMC estimators can be prohibitively expensive in complex physical systems in
which accurate models are computationally intensive. To accelerate the OED process, we propose a multi-fidelity EIG (MF-EIG)
estimator in which an ensemble of utility models of varying accuracy and cost are combined into a single EIG estimator via the
approximate control variate (ACV) method. To enable ACV, we first derive an alternate form of the EIG. We then analyze and
discuss two aspects of the inner loop of the estimator that can improve estimator performance. Finally, the MF-EIG estimator is
demonstrated on two OED problems: (1) a nonlinear model problem, where the estimator properties and performance improve-
ments over NMC are empirically investigated, and (2) a turbulent-flow experimental application that aims to infer the parameters
of the Reynolds-averaged Navier–Stokes (RANS) computational fluid dynamics (CFD) shear-stress transport (SST) turbulence
closure model. In these demonstrations, the MF-EIG estimator produces orders-of-magnitude reductions in estimator error relative
to NMC.

## Tutorial/Usage
Jupyter Notebook tutorial to be added soon.

## Examples
See ['Final Paper Figures'](https://github.com/tcoonsUM/mf-eig/tree/main/Final%20Paper%20Figures)(https://github.com/tcoonsUM/mf-eig/tree/main/Final%20Paper%20Figures) to reproduce the results of our paper. Most data that require many CPU-hours of compute time are simply loaded in their respective scripts.

To reproduce all data, scripts can be found under [link coming soon](https://github.com/tcoonsUM/mf-eig/), for example to optimize $\text{N}_{\text{in}}$ for each problem and run the estimator over many trials.
