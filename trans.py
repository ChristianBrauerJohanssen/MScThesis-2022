####################
# transition rules #
####################

# imports
import numpy as np
from numba import njit
import mt

@njit(fastmath=True)
def p_plus_func(p,psi,par,t):
    if t<=par.Tr:
        p_plus = p*psi*par.G*par.L[t]
        p_plus = np.fmax(p_plus,par.p_min) # lower bound
        p_plus = np.fmin(p_plus,par.p_max) # upper bound
    else: 
        p_plus = p*par.G*par.L[t] # no shocks to permanent income after retirement
        p_plus = np.fmax(par.p_min,np.fmin(p_plus,par.p_max)) # bounds
    return p_plus

@njit(fastmath=True)
def h_plus_func(h,rent,par):
    if rent: 
        h_plus = 0
    else: 
        h_plus = np.fmax(par.h_min,np.fmin(h,par.h_max)) # bounds
    return h_plus

@njit(fastmath=True)
def m_plus_func(a,p_plus,xi_plus,d,Tda,par,t):
    if t<=par.Tr:
        w_plus = p_plus*xi_plus
    else:
        w_plus = p_plus
    m_plus = a + mt.income_tax(w_plus,a,d,Tda,par)
    return m_plus

#njit(fastmath=True)
#def m_to_x_func(m_plus,,par):
#   if rent:
#
#   return m_plus + (1-par.tau)*n_plus
#

@njit(fastmath=True)
def Tda_plus_func(Tda):
    return np.fmax(0,Tda-1)

@njit(fastmath=True)
def d_plus_func(q,h,d,w,move,ref,t,Td,Tda,par):
    if move or ref:
        d_plus = (1+par.r_m)*np.fmin(par.omega_ltv*q*h,par.omega_dti*w)
    elif Tda > 0:
        d_plus = d
    else: 
        mp,_,_,_ = mt.mpmt(d,t,Td,Tda,par)
        d_plus = d - mp
    
    return d_plus