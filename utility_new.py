########################
# utility and bequests #
########################

# imports
from numba import njit

# felicity function
@njit(fastmath=True)
def func(c,h,move,rent,t,par):
    return func_nopar(c,h,move,rent,par.rho,par.alpha,par.kappa,par.phi,par.n[t])

@njit(fastmath=True)
def func_nopar(c,h,move,rent,rho,alpha,kappa,phi,n):
    s = (1-rent)*h + rent*phi*h
    u = (n*((c**alpha)*(s**(1.0-alpha)))**(1.0-rho))/(1.0-rho) - n*kappa*move
    return u

# marginal felicity function wrt. consumption
njit(fastmath=True)
def marg_func(c,h,rent,t,par):
    return marg_func_nopar(c,h,rent,par.rho,par.alpha,par.phi,par.n[t])

@njit(fastmath=True)
def marg_func_nopar(c,h,rent,rho,alpha,phi,n):
    s = (1-rent)*h + rent*phi*h
    return alpha*n*(c**(alpha*(1.0-rho)-1.0))*(s**((1.0-alpha)*(1.0-rho)))

# bequest function
@njit(fastmath=True)
def bequest_func(ab,par):
    return bequest_func_nopar(ab,par.thetab,par.K,par.rho,par)

@njit(fastmath=True)
def bequest_func_nopar(ab,thetab,K,rho,n):
    return thetab/(1-rho)*((ab+K)**(1-rho))

@njit(fastmath=True)
def marg_bequest_func(ab,par):
    return marg_bequest_func_nopar(ab,par.thetab,par.K,par.rho)

@njit(fastmath=True)
def marg_bequest_func_nopar(ab,thetab,K,rho):
    return thetab*(ab+K)**(-rho)