####################
# transition rules #
####################

# 1. imports
import numpy as np
from numba import njit
import mt

# 2. income process
@njit(fastmath=True)
def p_to_y_func(i_y,p,p_lag,t,par):
    y = p*par.chi[t] 
    #if t < par.Tr:
    #    y = p*par.chi[t] 
    #else: 
    #    y = par.chi[t]
    
    return y

# 3. cash on hand
@njit(fastmath=True)
def m_plus_func(a,y_plus,d,Td,Tda,par,t):
    
    pmt = mt.mpmt(d,t,Td,Tda,par)

    m_plus = (1+par.r)*a - pmt + mt.income_aftertax(y_plus,d,Tda,par) 
    return m_plus

# 4. mortgage plan
@njit(fastmath=True)
def Tda_plus_func(Tda):
    return np.fmax(0,Tda-1)

@njit(fastmath=True)
def d_plus_func(d,t,Td,Tda,par):
    if Tda > 0:
        d_plus = d
    else: 
        mp = mt.mpmt(d,t,Td,Tda,par)
        d_plus = (1+par.r_m)*d - mp
    
    return d_plus

# 5. bequest
@njit(fastmath=True)
def ab_plus_func(a,d,Tda,h,par):
    if Tda > 0:
        r = par.r_da
    else: 
        r = par.r_m
    return (1+par.r)*a + (1-par.delta)*par.q*h - mt.property_tax(par.q,h,par) - (1+r)*d


#@njit(fastmath=True)
#def p_plus_func(p,psi,par,t):
#    if t<=par.Tr:
#        p_plus = p**(par.rho_p)*psi*par.chi[t]
#        p_plus = np.fmax(par.p_min,np.fmin(p_plus,par.p_max)) # bounds
#    else: 
#        p_plus = p**(par.rho_p)*par.chi[t] # no shocks to permanent income after retirement
#        p_plus = np.fmax(par.p_min,np.fmin(p_plus,par.p_max)) # bounds
#    return p_plus