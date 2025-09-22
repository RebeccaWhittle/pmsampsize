# pmsampsize

Sample size for development of a prediction model.

## Installation

```bash
$ pip install pmsampsize
```

## Usage

`pmsampsize` computes the minimum sample size required for the development of a new multivariable prediction model using the criteria proposed by [Riley et al. (2018)](https://onlinelibrary.wiley.com/doi/10.1002/sim.7992). `pmsampsize` can be used to calculate the minimum sample size for the development of models with continuous, binary or survival (time-to-event) outcomes. [Riley et al. (2018)](https://onlinelibrary.wiley.com/doi/10.1002/sim.7992) lay out a series of criteria the sample size should meet. These aim to minimise the overfitting and to ensure precise estimation of key parameters in the prediction model.

```python
from pmsampsize.pmsampsize import *

samplesize = pmsampsize(type="b", cstatistic=0.89, parameters=24, prevalence=0.174) # change options to meet your requirements
```

## License

`pmsampsize` was created by Rebecca Whittle & Joie Ensor. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`pmsampsize` is based on the original R package [pmsampsize](https://cran.r-project.org/web/packages/pmsampsize/index.html) developed by Joie Ensor. 

`pmsampsize` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

