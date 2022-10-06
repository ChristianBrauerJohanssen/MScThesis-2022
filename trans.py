####################
# transition rules #
####################

# imports
import numpy as np
from numba import njit

@njit(fastmath=True)
def p_plus_func(p,psi,par,t):
    if t<=par.Tr:
        p_plus = p*psi*par.G*par.L[t]
        p_plus = np.fmax(p_plus,par.p_min) # lower bound
        p_plus = np.fmin(p_plus,par.p_max) # upper bound
    else: 
        p_plus = p*par.G*par.L[t] # no shocks to permanent income after retirement
        p_plus = np.fmax(p_plus,par.p_min) # lower bound
        p_plus = np.fmin(p_plus,par.p_max) # upper bound
    return p_plus

@njit(fastmath=True)
def h_plus_func(d,par,z,move,rent):
    if rent: 
        h_plus = 0
    n_plus = (par.Rh+z)*(1-par.delta)*d 
    n_plus = np.fmin(n_plus,par.n_max) # upper bound
    return h_plus

@njit(fastmath=True)
def m_plus_func(a,p_plus,xi_plus,par,t):
    if t<=par.Tr:
        y_plus = p_plus*xi_plus
    else:
        y_plus = p_plus
    m_plus = par.R*a+y_plus
    return m_plus

@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,par):
    return m_plus + (1-par.tau)*n_plus