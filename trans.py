####################
# transition rules #
####################

# 1. imports
import numpy as np
from numba import njit
import mt

# 2. income process
@njit(fastmath=True)
def p_plus_func(p,psi,par,t):
    if t<=par.Tr:
        p_plus = p**(par.rho_p)*psi*par.chi[t]
        p_plus = np.fmax(par.p_min,np.fmin(p_plus,par.p_max)) # bounds
    else: 
        p_plus = p**(par.rho_p)*par.chi[t] # no shocks to permanent income after retirement
        p_plus = np.fmax(par.p_min,np.fmin(p_plus,par.p_max)) # bounds
    return p_plus

@njit(fastmath=True)
def w_plus_func(p_plus,xi,work,par):
    if work == 0: 
        w_plus = par.b
    else:
        w_plus = (xi*p_plus-par.pi*par.b)/(1-par.pi)
    return w_plus

# 3. cash on hand
@njit(fastmath=True)
def m_plus_func(a,w_plus,d,Td,Tda,par,t):
    
    pmt,_,_,_ = mt.mpmt(d,t,Td,Tda,par)

    m_plus = a - pmt + mt.income_aftertax(w_plus,a,d,Tda,par) 
    return m_plus

njit(fastmath=True)
def m_to_mnet_stay(m_plus,h,par):
    # unpack parameters
    delta = par.delta
    q = par.q

    # compute m_net
    m_net = m_plus - delta*q*h - mt.property_tax(q,h,par)
    return m_net

njit(fastmath=True)
def m_to_mnet_ref(m_plus,h,d,d_prime,par): 
    # a. take new loan?
    loan = 0
    if d_prime > 0:
        loan = 1

    # b. net cash-on-hand
    m_net = m_plus-d-par.delta*par.q*h-mt.property_tax(par.q,h,par)-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime 
    return m_net

njit(fastmath=True)
def m_to_mnet_buy(m_plus,h,d,d_prime,hbuy,par): 
    # a. take new loan?
    loan = 0
    if d_prime > 0:
        loan = 1

    # b. net cash-on-hand
    m_net = m_plus+d-(par.delta+par.C_sell)*par.q*h-mt.property_tax(par.q,h,par)-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime -(1+par.C_buy)*par.q*hbuy
    return m_net

@njit(fastmath=True)
def m_to_mnet_rent(m_plus,htilde,par):
    # unpack parameters
    q_r = par.q_r
    return m_plus - q_r*htilde

# 4. mortgage plan
@njit(fastmath=True)
def Tda_plus_func(Tda):
    return np.fmax(0,Tda-1)

@njit(fastmath=True)
def d_plus_func(q,h,d,w,t,Td,Tda,par):
    #if move or ref:
    #    d_plus = (1+par.r_m)*np.fmin(par.omega_ltv*q*h,par.omega_dti*w)
    if Tda > 0:
        d_plus = d
    else: 
        mp,_,_,_ = mt.mpmt(d,t,Td,Tda,par)
        d_plus = (1+par.r_m)*d - mp
    
    return d_plus

# 5. bequest
@njit(fastmath=True)
def ab_plus_func(a,d,Tda,h,par):
    if Tda > 0:
        r = par.r_da
    else: 
        r = par.r_m
    return (1+par.r)*a + (1-par.C_sell-par.delta)*par.q*h - mt.property_tax(par.q,h,par) - (1+r)*d