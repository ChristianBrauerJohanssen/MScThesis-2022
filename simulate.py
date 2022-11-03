####################
#    1. Imports    #
####################

# a. standard packages
import numpy as np
from numba import njit, prange

# b. NumEconCph packages
from consav import linear_interp # for linear interpolation

# c. local modules
import trans
import utility
import mt

###############################
#    2. Simulate lifecyle     #
###############################

@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ simulate full life-cycle given prices and bequest distribution """

    # unpack state containers
    h = sim.h
    d = sim.d
    Td = sim.Td 
    Tda = sim.Tda
    p = sim.p
    y = sim.y
    m = sim.m

    # unpack choice containers
    h_prime = sim.h_prime
    d_prime = sim.d_prime
    Td_prime = sim.Td_prime 
    Tda_prime = sim.Tda_prime 
    c = sim.c
    a = sim.a 
    discrete = sim.discrete # living situation
    
    # simulate forward
    for t in range(par.T):
        for i in prange(par.simN):
            
            # a. beginning of period states
            if t == 0:
                h[t,i] = 0
                d[t,i] = 0
                Td[t,i] = 0
                Tda[t,i] = 0
                p[t,i] = sim.p0[i]
                m[t,i] = sim.a0[i]
            else:
                h[t,i] = h_prime[t-1,i]
                d[t,i] = trans.d_plus_func(d_prime[t-1,i],t,Td_prime[t-1,i],
                                           Tda_prime[t-1,i],par) # with or without trans_func?
                Td[t,i] = Td_prime[t-1,i]
                Tda[t,i] = trans.Tda_plus_func(Tda_prime[t-1,i]) 
                p[t,i] = trans.p_plus_func(p[t-1,i],sim.psi[t,i],par,t-1)
                m[t,i] = trans.m_plus_func(a[t-1,i],p[t,i],sim.xi[t,i],par,t)
            
            # scale mortgage grid
            d_prime_high = par.q*h[t,i]
            grid_d_prime = np.linspace(0,d_prime_high,par.Nd)

            y[t,i] = p[t,i]*sim.xi[t,i] # add trans func

            # b. optimal choices and post decision states
            optimal_choice(i,t,h[t,i],d[t,i],Td[t,i],Tda[t,i],p[t,i],m[t,i],
                           h_prime[t,i:],d_prime[t,i:],grid_d_prime,Td_prime[t,i:],
                           Tda_prime[t,i:],c[t,i:],a[t,i:],discrete[t,i:],sol,par)
            
@njit            
def optimal_choice(i,t,h,d,Td,Tda,p,m,h_prime,d_prime,grid_d_prime,Td_prime,Tda_prime,c,a,discrete,sol,par):

    # a. compute gross cash-on-hand
    x = trans.x_plus_func(m,n,par)

    # b. find indices of discrete vars
    i_h = np.where(par.grid_h == h)
    i_Td = Td - par.Td_bar
    i_Tda = Tda
    
    # c. discrete choices
        # i. find best rent size
    inv_v_rent = 0 
    for i_ht in range(par.Nhtilde):
        inv_v_rent_temp = linear_interp.interp_2d(par.grid_y,par.grid_m,sol.inv_v_keep[t,i_ht],p,m) 
        if inv_v_rent_temp > inv_v_rent:
            inv_v_rent = inv_v_rent_temp
            i_ht_best = i_ht
            h_tilde = par.grid_htilde[i_ht_best]     
    
        # ii. find best overall discrete choice
    inv_v_stay = linear_interp.interp_3d(grid_d_prime,par.grid_p,par,par.grid_m,sol.inv_v_stay[t,i_h,:,i_Td,i_Tda,:,:],d,p,m)    
    inv_v_ref = linear_interp.interp_3d(grid_d_prime,par.grid_p,par,par.grid_m,sol.inv_v_ref[t,i_h,:,i_Td,i_Tda,:,:],d,p,m)    
    inv_v_buy = linear_interp.interp_3d(grid_d_prime,par.grid_p,par,par.grid_m,sol.inv_v_buy[t,i_h,:,i_Td,i_Tda,:,:],d,p,m)    

    if h == 0:
        discrete_choice = np.amax([inv_v_buy,inv_v_rent])
    else: 
        discrete_choice = np.amax([inv_v_stay,inv_v_ref,inv_v_buy,inv_v_rent])

    # d. continuous choices
        # i. stay and pay
    if discrete_choice == inv_v_stay:

        discrete[0] = 0
        h_prime[0] = h
        d_prime[0] = d
        Td_prime[0] = Td
        Tda_prime[0] = Tda


        c[0] = linear_interp.interp_3d(grid_d_prime,par.grid_p,par.grid_m,
                                       sol.c_stay[t,i_h,:,i_Td,i_Tda,:,:],d_prime,p,x)
        
        tot_exp = c[0] + par.delta*par.q*h + mt.property_tax(par.q,h,par)

        ## ensure feasibility
        if tot_exp > x: 
            c[0] = c[0] - (tot_exp-x)
            #if c[0] <= 0:
                # do default
            a[0] = 0.0
        else:
            a[0] = x - tot_exp
        
        # ii. stay and refinance
    elif discrete_choice == inv_v_ref: 
            
        discrete[0] = 1
        h_prime[0] = h
        d_prime[0] = linear_interp.interp_3d(grid_d_prime,par.grid_p,par.grid_m,
                                             sol.d_prime_ref[t,i_h,:,i_Td,i_Tda,:,:],
                                             d,p,m)
        Td_prime[0] = mt.Td_func(t,par)
        Tda_float = linear_interp.interp_3d(grid_d_prime,par.grid_p,par.grid_m,
                                             sol.Tda_prime_ref[t,i_h,:,i_Td,i_Tda,:,:],
                                             d,p,m)
        Tda_prime[0] = np.round(Tda_float,0)

        c[0] = linear_interp.interp_3d(grid_d_prime,par.grid_p,par.grid_m,
                                       sol.c_ref[t,i_h,:,i_Td,i_Tda,:,:],
                                       d,p,m)
        
        ## ensure feasibility (come back to this later)
        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]

        # iii. buy new house
    elif discrete_choice == inv_v_buy: 
        
        discrete[0] = 2

        ## house purchase
        h_prime_offgrid = linear_interp.interp_3d(grid_d_prime,par.grid_p,par.grid_m,
                                             sol.h_buy[t,i_h,:,i_Td,i_Tda,:,:],
                                             d,p,m)
        i_h_prime = np.searchsorted(par.grid_h,h_prime_offgrid,side='left') # find nearest point in grid
        h_prime = par.grid_h[i_h_prime-1] #??

        ## scale new mortgage grid
        d_prime_high_buy = par.q*h_prime
        grid_d_prime_buy = np.linspace(0,d_prime_high_buy,par.Nd)
        d_prime[0] = linear_interp.interp_3d(grid_d_prime_buy,par.grid_p,par.grid_m,
                                             sol.d_prime_ref[t,i_h,:,i_Td,i_Tda,:,:],
                                             d,p,m)
        Td_prime[0] = mt.Td_func(t,par)
        Tda_float = linear_interp.interp_3d(grid_d_prime,par.grid_p,par.grid_m,
                                             sol.Tda_prime_ref[t,i_h,:,i_Td,i_Tda,:,:],
                                             d,p,m)
        Tda_prime[0] = np.round(Tda_float,0)

        c[0] = linear_interp.interp_3d(grid_d_prime,par.grid_p,par.grid_m,
                                       sol.c_ref[t,i_h,:,i_Td,i_Tda,:,:],
                                       d,p,m)

        ## ensure feasibility (come back to this later)
        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]


    elif discrete_choice == inv_v_rent:
        
        discrete[0] = 3
        h_prime[0] = 0
        d_prime[0] = 0
        Td_prime[0] = 0
        Tda_prime[0] = 0

        c[0] = linear_interp.interp_2d(par.grid_p,par.grid_m,
                                       sol.c_rent[t,i_ht_best],
                                       p,m)
        
        # back out gross cash on hand and ensure feasibility
        # come back to this later

@njit            
def euler_errors(sim,sol,par):

    # unpack
    euler_error = sim.euler_error
    euler_error_c = sim.euler_error_c
    
    for i in prange(par.simN):
        
        discrete_plus = np.zeros(1)
        d_plus = np.zeros(1)
        c_plus = np.zeros(1)        
        c_bump_plus = np.zeros(1)
        a_plus = np.zeros(1)

        for t in range(par.T-1):

            constrained = sim.a[t,i] < par.euler_cutoff
            
            if constrained:

                euler_error[t,i] = np.nan
                euler_error_c[t,i] = np.nan
                continue

            else:

                RHS = 0.0
                for ishock in range(par.Nshocks):
                        
                    # i. shocks
                    psi = par.psi[ishock]
                    psi_w = par.psi_w[ishock]
                    xi = par.xi[ishock]
                    xi_w = par.xi_w[ishock]

                    # ii. next-period states
                    p_plus = trans.p_plus_func(sim.p[t,i],psi,par)
                    n_plus = trans.n_plus_func(sim.d[t,i],par)
                    m_plus = trans.m_plus_func(sim.a[t,i],p_plus,xi,par,t)

                    # iii. weight
                    weight = psi_w*xi_w

                    # iv. next-period choices

                    optimal_choice(t+1,p_plus,n_plus,m_plus,discrete_plus,d_plus,c_plus,a_plus,sol,par,c_bump_plus)

                    # v. next-period marginal utility

                    RHS += weight*par.beta*par.R*utility.marg_func(c_plus[0],d_plus[0],par)
                

                euler_error[t,i] = sim.c[t,i] - utility.inv_marg_func(RHS,sim.d[t,i],par)

                euler_error_c[t,i] = sim.c[t,i]

@njit(parallel=True)
def calc_utility(sim,sol,par):
    """ calculate utility for each individual """

    # unpack
    u = sim.utility
    
    for t in range(par.T):
        for i in prange(par.simN):
            
            u[i] += par.beta**t*utility.func(sim.c[t,i],sim.d[t,i],par)
            
