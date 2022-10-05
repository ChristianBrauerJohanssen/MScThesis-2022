# imports
from numba import njit

# felicity function
@njit(fastmath=True)
def func(c,h,n,move,rent,par):
    return func_nopar(c,h,n,move,rent,par.nu,par.rho,par.alpha,par.kappa,par.phi)

@njit(fastmath=True)
def func_nopar(c,h,n,move,rent,nu,rho,alpha,kappa,phi):
    s = (1-rent)*h/n + rent*phi*h/n
    u = n*((1-nu)/(1-rho)*(c/n)**(1-rho) + nu/(1-alpha)*s**(1-alpha) - kappa*move)
    return u

# marginal felicity function wrt. consumption
njit(fastmath=True)
def marg_func(c,n,par):
    return marg_func_nopar(c,n,par.nu,par.rho)

@njit(fastmath=True)
def marg_func_nopar(c,n,nu,rho):
    return (1-nu)(c/n)**(-rho)

# bequest
@njit(fastmath=True)
def bequest_func(ab,par):
    return bequest_func_nopar(ab,par.thetab,par.K,par.rho)

@njit(fastmath=True)
def bequest_func_nopar(ab,thetab,K,rho):
    return thetab/(1-rho)*((ab+K)**(1-rho))