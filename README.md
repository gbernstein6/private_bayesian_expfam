# private_bayesian_expfam

This is a code release as part of the paper Differentially Private Bayesian Inference for Exponential Families.[1] (https://arxiv.org/pdf/1809.02188.pdf)

The code is split into two modules, one for models with _bounded_ sufficient statistics and one for models with _unbounded_ sufficient statistics. Each module has a `driver.py` from which the code can be run and posteriors plotted. The population size, privacy level, and model are specified by the user in the driver.

Models are fully plug and play. To add your own model, follow the template in the corresponding `distributions.py`. 

* **Code Requirements**
  - Python 2 or 3
  - Numpy, Scipy, Matplotlib
  - Autograd (https://github.com/HIPS/autograd)


[1] Garrett Bernstein and Daniel Sheldon. Differentially Private Bayesian Inference for Exponential Families. NeurIPS 2018. [https://arxiv.org/pdf/1809.02188.pdf]
