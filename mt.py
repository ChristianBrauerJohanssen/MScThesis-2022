#####################################
# mortgage payment and tax schedule #
#####################################

# imports
import numpy as np
from numba import njit, prange

# income and property tax functions
@njit 
def income_tax(w,a_lag,d,Tda,par):
    # find relevant interest rate 
    if Tda > 0: 
        r = par.r_da
    else: 
        r = par.r_m
    
    y = w + r*a_lag
    deduct = par.tau_r*np.min(r*d,par.rd_bar)

    ytilde = (1-par.tau_y0)*(y-deduct)**(1-par.tau_y1) + deduct

    return ytilde

@njit
def property_tax(q,h,par):
    tax_h = par.tau_h0*np.min(q*h,par.qh_bar) + np.max(0,par.tau_h1*(q*h-par.qh_bar))
    return tax_h

# mortgage payment schedule
#@njit
#def mpmt()