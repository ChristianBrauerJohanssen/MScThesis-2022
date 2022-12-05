###########
# imports #
###########
import numpy as np
from numba import njit

####################
#   analyse model  #
####################
@njit
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
    N = len(x_flat_sort)
    x_mean = x_flat_sort.mean()
    frac = 2/(N**2*x_mean)
    summation = 0 
    for i in range(N):
        summation += (i+1)*(x_flat_sort[i]-x_mean)
    gini = frac*summation
    
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

def model_moments(model):
    """
    Calculate various moments from a model simulation

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
             'Average housing expenditure',
             'Mean mortgage size',
             'Share of homeowners with mortgage',
             'DA mortgage share',
             'Average LTV ratio',
             'Average DTI ratio',
             'Gini coefficient',
             #'Mean CEV',
             ]
    # c. prep calculations
    bool_rent = sim.discrete == 3

    rent_cost = bool_rent*par.q_r*sim.h_tilde
    own_cost = par.delta*par.q*sim.h + sim.prop_tax + sim.interest
    hexp  = rent_cost + own_cost  

    # c. calculate the model moments
    ho_share = 1 - np.sum(sim.discrete == 3)/(par.T*par.simN)
    mean_hsize = np.mean(sim.h[sim.h > 0])                                          # conditional on owning a house
    mean_hexp = np.mean(hexp)                                     
    mean_mortgage = np.mean(sim.d_prime[sim.d_prime > 0])                           # conditional on having a mortgage
    ho_mort_share = np.mean(sim.d_prime[sim.h_prime > 0] > 0)                       # share of homwowners with mortgage
    da_mort_share = np.mean(sim.Tda_prime[sim.d_prime > 0] > 0)                     # DA share of all mortgages
    mean_ltv = np.mean(sim.d_prime[sim.d_prime > 0]/sim.h_prime[sim.d_prime > 0]) 
    mean_dti = np.mean(sim.d_prime[sim.d_prime > 0]/sim.y[sim.d_prime > 0])         # conditional on having a mortgage
    net_wealth = sim.a + sim.h_prime - sim.d_prime
    gini,_ = gini_lorenz(net_wealth)                      
    
    model_moments = np.array([ho_share,
                            mean_hsize, 
                            mean_hexp,
                            mean_mortgage, 
                            ho_mort_share, 
                            da_mort_share, 
                            mean_ltv, 
                            mean_dti, 
                            gini])
    
    return names, model_moments    


