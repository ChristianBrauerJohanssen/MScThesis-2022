#####################################
# mortgage payment and tax schedule #
#####################################

# imports
import numpy as np
from numba import jit, njit, prange

# income and property tax functions
@njit(fastmath=True)
def income_aftertax(w,d,Tda,par):
    # unpack input
    tau_y0 = par.tau_y0
    tau_y1 = par.tau_y1
    tau_r = par.tau_r 
    rd_bar = par.rd_bar
    
    # find relevant interest rate 
    if Tda > 0: 
        r = par.r_da
    else: 
        r = par.r_m
    
    # compute after tax income
    y = w
    ird = tau_r*np.fmin(r*d,rd_bar)
    ytilde = (1-tau_y0)*(y-ird)**(1-tau_y1) + ird

    return ytilde

@njit(fastmath=True)
def property_tax(q,h,par):
    # unpack input
    tau_h0 = par.tau_h0
    tau_h1 = par.tau_h1
    qh_bar = par.qh_bar
    
    # compute property tax
    tax_h = tau_h0*np.fmin(q*h,qh_bar) + np.fmax(0,tau_h1*(q*h-qh_bar))

    return tax_h


# mortgage payment schedule
@njit(fastmath=True)
def Td_func(t,par):
    if par.T > t + par.Td_bar:
        Td = t + par.Td_bar
    else: 
        Td = par.T-1
    return Td

@njit(fastmath=True)
def alfahage(n,r):
     return ((1+r)**n-1)/(r*(1+r)**n)

@njit(fastmath=True)
def mpmt(d,t,Td,Tda,par):
    """ 
    args:
        d (float)       - outstanding mortgage balance
        Td (int)        - time of last mortgage term
        Tda (int)       _ remaining periods of deferred amortisation
    
    returns:
        tot_pmt         - total mortgage payment in period t
        ir_pmt          - interest payment in period t
        pr_pmt          - reduction in principal in period t
        pr_rem          - remaining principal
    """
    #derive the mortgage schedule
    if t >= Td:
        # mortgage fully payed down
        tot_pmt = 0
    
    elif Tda > 0: 
        # unpack 
        r = par.r_da
        # compute
        tot_pmt = r*d
        
    else: 
        # unpack
        r = par.r_m
        # compute
        tot_pmt = d*(r/(1-(1+r)**(t-Td)))
        
    return tot_pmt