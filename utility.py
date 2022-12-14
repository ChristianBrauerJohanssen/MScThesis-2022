########################
# utility and bequests #
########################

# imports
from numba import njit

# felicity function
@njit(fastmath=True)
def func(c,h,move,rent,t,par):
    return func_nopar(c,h,move,rent,par.nu,par.rho,par.alpha,par.kappa,par.phi,par.n[t])

@njit(fastmath=True)
def func_nopar(c,h,move,rent,nu,rho,alpha,kappa,phi,n):
    s = (1-rent)*h/n + rent*phi*h/n
    u = n*((1-nu)/(1-rho)*(c/n)**(1-rho) + nu/(1-alpha)*s**(1-alpha) - kappa*move)
    return u

# marginal felicity function wrt. consumption
njit(fastmath=True)
def marg_func(c,par,t):
    return marg_func_nopar(c,par.nu,par.rho,par.n[t])

@njit(fastmath=True)
def marg_func_nopar(c,nu,rho,n):
    return (1-nu)*(c/n)**(-rho)

# bequest function
@njit(fastmath=True)
def bequest_func(ab,par):
    return bequest_func_nopar(ab,par.thetab,par.K,par.rho)

@njit(fastmath=True)
def bequest_func_nopar(ab,thetab,K,rho):
    return thetab/(1-rho)*((ab+K)**(1-rho))

@njit(fastmath=True)
def marg_bequest_func(ab,par):
    return marg_bequest_func_nopar(ab,par.thetab,par.K,par.rho)

@njit(fastmath=True)
def marg_bequest_func_nopar(ab,thetab,K,rho):
    return thetab*(ab+K)**(-rho)