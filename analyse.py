###########
# imports #
###########
import numpy as np

####################
#   analyse model  #
####################

def gini_lorenz(x):
    """
    Calculate the Gini coefficient of a numpy array. 
    based on bottom eq: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    args:
        x: a numpy array

    returns:
       gini: the Gini coefficient of x
       lorenz: the Lorenz curve of x
    """
    # a. flatten and sort x
    x_flat_sort = np.sort(x.flatten())

    # b. compute the Lorenz curve
    lorenz = np.cumsum(x_flat_sort)/x_flat_sort.sum()

    # c. compute the Gini coefficient
    gini = 1 - 2 * lorenz[1:].sum() / lorenz[-1]
    
    return gini, lorenz

def model_moments_targ(model):
    """
    Calculate various moments targeted in the model calibration

    args:
        model: a HAH model object

    returns:
        model_moments: a numpy array of the model moments
    """
    # a. unpack 
    par = model.par
    sim = model.sim

    # b. name the moments
    names = ['Aggregate net worth / (annual) labour income',
             'Annual fraction of houses sold',
             'Home ownership rate of <35 y.o.',
             'Mean NW at age 75 / mean NW at age 55/50',
             'Share of households leaving no bequest'
             ]


    # c. calculate the model moments
    model_moments = np.array([model.mean(), model.std(), model.max(), model.min(), model.sum()])
    
    return names, model_moments    

def model_moments_untarg(model):
    """
    Calculate various untargeted moments from a model simulation

    args:
        model: a HAH model object

    returns:
        model_moments: a numpy array of the model moments
    """
    # a. unpack 
    par = model.par
    sim = model.sim

    # b. name the moments
    names = ['Homeowner share',
             'Mean house size',
             'Average housing expenditure'
             'Mean mortgage size',
             'Share of homeowners with mortgage',
             'DA mortgage share',
             'Average LTV ratio',
             'Average DTI ratio',
             'Gini coefficient',
             'Mean CEV',
             ]

    # c. calculate the model moments
    
    
    model_moments = np.array([model.mean(), model.std(), model.max(), model.min(), model.sum()])
    


    return names, model_moments    


