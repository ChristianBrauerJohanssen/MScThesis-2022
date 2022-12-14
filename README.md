# Macroprudential Policy and the Housing Market - *Quantifying Unequal Policy Impact Across Age and Wealth*
This is the working repository for my Master's thesis in Economics at the [University of Copenhagen](https://www.economics.ku.dk/).

## Replicating the results
To replicate all figures and tables in the thesis containing model output, run the notebook [HAHModel.ipynb](HAHModel.ipynb).

**Parameterisation:**

When initialising the model, parameters and grids are set and allocated as dictated in the initialisation in [HAHModel.py](HAHModel.py). Alternative parameter values can be provided in the .py file or passed as a dictionary with parameter name as key when initialising the model class. 

**Tables:**

TBA
 <!--
The average MPCs in Table 3 and MPCs from Table 4 sensitivity analysis are also present in notebook. As default, the MPCs are cross-computed in the simulation part, but to create the non cross-computed MPCs a boolean for ``cross_compute=False`` can be set when initiating the model.
-->
**Figures:**

TBA
 <!-- 
Each notebook will plot the figures associated with the given model.
-->

## Dependencies
The code structure builds upon the framework developed by Jeppe Druedahl & Co. in the [NumEconCopenhagen Project](https://github.com/NumEconCopenhagen)

Packages required for running the notebooks are:
- [ConSav](https://pypi.org/project/ConSav/)
- [EconModel](https://pypi.org/project/EconModel/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)
- [numba](https://pypi.org/project/numba/)
- [pandas](https://pypi.org/project/pandas/)
<!-- - [quantecon](https://pypi.org/project/quantecon/)
-->
