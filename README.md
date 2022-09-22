# Macroprudential Policy and the Housing Market - *Quantifying Unequal Policy Impact Across Age and Wealth*
This is the working repository for my Master's thesis in Economics at the [University of Copenhagen](https://www.economics.ku.dk/).


## Replicating the results
To replicate the results of the thesis, the interested reader would have to review the two following notebooks representing the models in A.1 and section 2 in our paper:
- [One-Asset Model](one_asset/OneAssetModel.ipynb)
- [Two-Asset Model](two_asset/TwoAssetModel.ipynb)

Each model-specific notebook is placed in a separate folder along with the modules needed for solving the model. The notebooks each contain the code for replicating the figures and computation times. 

**Model specification: Table 1 and Table 5:**

Can be found in the model initialisation in [Two-Asset Model module](two_asset/TwoAssetModel.py). 

**MPCs:**

The average MPCs in Table 3 and MPCs from Table 4 sensitivity analysis are also present in notebook. As default, the MPCs are cross-computed in the simulation part, but to create the non cross-computed MPCs a boolean for ``cross_compute=False`` can be set when initiating the model.

**Figures:**

Each notebook will plot the figures associated with the given model.


## Dependencies
The code is built upon the framework developed by Jeppe Druedahl & Co. in the [NumEconCopenhagen Project](https://github.com/NumEconCopenhagen)

Packages required for running the notebooks are:
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [numba](https://pypi.org/project/numba/)
- [quantecon](https://pypi.org/project/quantecon/)
- [EconModel](https://pypi.org/project/EconModel/)
- [ConSav](https://pypi.org/project/ConSav/)
- [GEModelTools](https://github.com/NumEconCopenhagen/GEModelTools)
- [matplotlib](https://pypi.org/project/matplotlib/)
