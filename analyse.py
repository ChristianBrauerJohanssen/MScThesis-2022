###########
# imports #
###########
# a. standard packages
import numpy as np
from numba import njit, prange

# b. NumEconCph packages
from EconModel import jit

# c. local modules
import utility

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

#@njit(fastmath=True)
def calc_utility_cev_adjusted(sim,par,guess,n):
    """ calculate utility for each individual taking into account the CEV guess """

    # initialise
    u = 0
    
    for t in range(par.T):
        # stayers
        if sim.discrete[t,n] == 0: 
            move = 0
            rent = 0
            u += par.beta**t*utility.func(guess*sim.c[t,n],sim.h_prime[t,n],move,rent,t,par)
        # refinancers 
        elif sim.discrete[t,n] == 1:
            move = 0
            rent = 0
            u += par.beta**t*utility.func(guess*sim.c[t,n],sim.h_prime[t,n],move,rent,t,par)
        # buyers
        elif sim.discrete[t,n] == 2:
            move = 1
            rent = 0
            u += par.beta**t*utility.func(guess*sim.c[t,n],sim.h_prime[t,n],move,rent,t,par)
        # renters
        elif sim.discrete[t,n] == 3:
            move = 0
            rent = 1
            u += par.beta**t*utility.func(guess*sim.c[t,n],sim.h_tilde[t,n],move,rent,t,par)
    return u

#@njit(parallel=True)
def cev(sim1,sim2,par1,par2,guess=1,N=100):
    """ compute ex ante consumption equivalent variation as in Sommer and Sullivan (2018). Note that
    the computation holds all variables and choices constant except for consumption. 
    
    args:
        model1: a HAH model object
        model2: a HAH model object
        guess: a guess for the relative consumption change
    """

    # a. unpack and prep
    utility1 = sim1.utility[0:N] # baseline sum of discounted utility of size (1,simN)
    #guess_vec = np.ones(par1.simN)*guess # a vector of size (1,simN) with the guess
    guess_vec = np.ones(N)*guess # a vector of size (1,simN) with the guess

    # b. compute the utility of the counterfactual model for each individual    
    #for n in prange(par1.simN):
    for n in range(N):
        utility_now = 0 
        iter = 0
        while np.abs(utility1[n]-utility_now) > par1.tol*10**8 and iter <= par1.max_iter_simulate*10:

            # compute CEV-adjusted utility for the alternative model
            utility_now = calc_utility_cev_adjusted(sim2,par2,guess_vec[n],n)

            # check if baseline and counterfactual utility are unequal
            if np.abs(utility1[n]-utility_now) > par1.tol*10**8:
                # update guess
                guess_vec[n] = (utility_now/utility1[n])
            
            if iter%5000 == 0:
                print('CEV computation did not converge for individual ',n,' after ',iter,' iterations')
                print('utility1[n] = ',utility1[n])
                print('utility_now = ',utility_now)
                print('guess_vec[n] = ',guess_vec[n])
            
            # update counter and reset utility_now
            iter += 1
            utility_now = 0

            
    
    ## check if convergence was achieved for all individuals
    #if np.any(np.abs(utility1-utility_new) > par1.tol*10**8):
    #    raise ValueError('CEV computation did not converge for all individuals')

    return guess_vec

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
             'Mean NW at age 75 / mean NW at age 50',
             'Share of households leaving no bequest',
             'Taxes to labour income'
             ]

    # c. prep calculations
    mean_nw = np.mean(sim.a + sim.h_prime - sim.d_prime)
    mean_nw_age = np.mean(sim.a + sim.h_prime - sim.d_prime,axis=1)
    bool_da_last = sim.Tda_prime[par.T-2,:] > 0
    bequest_mat = (1+par.r)*sim.a[par.T-1,:] + (1-par.delta)*par.q*sim.h_prime[par.T-1,:] - bool_da_last*(1+par.r_da)*sim.d_prime[par.T-1,:] - (1-bool_da_last)*(1+par.r_m)*sim.d_prime[par.T-1,:]
    tax_to_inc = np.sum(sim.inc_tax)/np.sum(sim.y)
    

    # c. calculate the model moments
    nw_to_y = mean_nw/np.mean(sim.y)
    h_share_sold = np.mean(sim.h_prime != sim.h) # adjust for sim.h > 0 ?
    ho_u35 = np.mean(sim.h_prime[0:10,:] > 0)
    nw_75_55 = mean_nw_age[75-par.Tmin]/mean_nw_age[50-par.Tmin]
    no_beq_share = np.mean(bequest_mat == 0)

    model_moments = np.array([nw_to_y,
                            h_share_sold,
                            ho_u35,
                            nw_75_55,
                            no_beq_share,
                            tax_to_inc])
    
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
             'Gini wealth coefficient',
             #'Mean CEV',
             ]
    # c. prep calculations
    bool_rent = sim.discrete == 3

    rent_cost = bool_rent*par.q_r*sim.h_tilde
    own_cost = par.delta*par.q*sim.h + sim.prop_tax + sim.interest
    hexp  = rent_cost + own_cost  

    # d. calculate the model moments
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

