####################
#    1. Imports    #
####################

# a. standard packages
import numpy as np
from numba import njit, prange

# b. NumEconCph packages
from consav import linear_interp # for linear interpolation
from consav.markov import choice # for lookup in transition matrix

# c. local modules
import trans
import utility
import mt

###############################
#  2. Simulate lifecyle - PE  #
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
            
            # a. shock realisation
            if t == 0:    
                p_y = sim.p_y_ini[i]
                i_y_lag = choice(p_y,par.w_ergodic_cumsum)
            else:
                i_y_lag = sim.i_y[t-1,i]
            
            p_y = sim.p_y[t,i]
            i_y_ = sim.i_y[t,i] = choice(p_y,par.w_trans_cumsum[i_y_lag,:])

            # b. beginning of period states and income
            if t == 0:
                h[t,i] = 0
                d[t,i] = 0
                Td[t,i] = 0
                Tda[t,i] = 0                
                p[t,i] = par.grid_w[i_y_]
                y[t,i] = trans.p_to_y_func(i_y_,p[t,i],t,par)
                m[t,i] = sim.a0[i] + y[t,i]
                
            else:
                h[t,i] = h_prime[t-1,i]
                d[t,i] = trans.d_plus_func(d_prime[t-1,i],t,Td_prime[t-1,i],
                                           Tda_prime[t-1,i],par) # with or without trans_func?
                Td[t,i] = Td_prime[t-1,i]
                Tda[t,i] = trans.Tda_plus_func(Tda_prime[t-1,i]) 
                p[t,i] = par.grid_w[i_y_]
                y[t,i] = trans.p_to_y_func(i_y_,p[t,i],t,par)
                m[t,i] = trans.m_plus_func(a[t-1,i],y[t,i],d[t-1,i],Td[t-1,i],Tda[t-1,i],par,t)

            # c. scale mortgage grid
            d_prime_high = par.q*h[t,i]
            grid_d_prime = np.linspace(0,d_prime_high,par.Nd)
        
            # d. optimal choices and post decision states
            optimal_choice(i,i_y_,t,h[t,i],d[t,i],Td[t,i],Tda[t,i],m[t,i],
                           h_prime[t,i:],d_prime[t,i:],grid_d_prime,Td_prime[t,i:],Tda_prime[t,i:],
                           c[t,i:],a[t,i:],discrete[t,i:],sol,par)
            
@njit            
def optimal_choice(i,i_y_,t,h,d,Td,Tda,m,h_prime,d_prime,grid_d_prime,Td_prime,Tda_prime,c,a,discrete,sol,par):

    # a. compute gross cash-on-hand
    m_gross_stay = m-par.delta*par.q*h-mt.property_tax(par.q,h,par)
    m_gross_rent = m_gross_buy = m-d+(1-par.delta-par.C_sell)*par.q*h-mt.property_tax(par.q,h,par)
    m_gross_ref = m_gross_stay-d

    # b. find indices of discrete vars
    i_Td = int(Td - par.Td_bar)
    i_Tda = int(Tda)
    i_m_gross_ref = find_nearest(par.grid_x,m_gross_ref)
    i_m_gross_buy = find_nearest(par.grid_x,m_gross_buy)
    
    # c. discrete choices
        # i. find best rent size
    inv_v_rent = 0 
    for i_ht in range(par.Nhtilde):
        m_net_rent = m_gross_rent - par.q_r*par.grid_htilde[i_ht]
        inv_v_rent_temp = linear_interp.interp_1d(par.grid_m,sol.inv_v_rent[t,i_ht,i_y_],m_net_rent) 
        if inv_v_rent_temp > inv_v_rent:
            inv_v_rent = inv_v_rent_temp
            i_ht_best = i_ht
            h_tilde = par.grid_htilde[i_ht_best]     
        
        # ii. buy  
    inv_v_buy = linear_interp.interp_1d(par.grid_x,sol.inv_v_buy_fast[t,i_y_,:],m_gross_buy) 

        # iii. stay and refinance
    if h != 0:
        i_h = np.where(par.grid_h == h)[0].item()
        inv_v_stay = linear_interp.interp_2d(grid_d_prime,par.grid_m,sol.inv_v_stay[t,i_h,:,i_Td,i_Tda,i_y_,:],d,m_gross_stay)    
        inv_v_ref = linear_interp.interp_1d(par.grid_x,sol.inv_v_ref_fast[t,i_h,i_y_,:],m_gross_ref)    
    
    # d. find behaviour given discrete choice
    if h == 0:
        discrete_choice = np.amax(np.array([inv_v_buy,inv_v_rent]))
        # o. buy new house?
        if discrete_choice == inv_v_buy: 
            
            discrete[0] = 2

            ## house purchase
            h_prime[0] = sol.h_buy_fast[t,i_y_,i_m_gross_buy]

            ## mortgage plan choice
            d_prime[0] = sol.d_prime_buy_fast[t,i_y_,i_m_gross_ref]
            Td_prime[0] = mt.Td_func(t,par)
            Tda_prime[0] = sol.Tda_prime_buy_fast[t,i_y_,i_m_gross_buy]

            ## consumption choice
            c[0] = linear_interp.interp_1d(par.grid_x,sol.c_buy_fast[t,i_y_,:],m_gross_buy)

            ## ensure feasibility (come back to this later)
            loan = int(d_prime[0]>0)
            m_net_buy = m_gross_buy-(1+par.C_buy)*par.q*h_prime[0]-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime[0]
            if c[0] > m_net_buy : 
                c[0] = m_net_buy
                a[0] = 0.0
            else:
                a[0] = m_net_buy - c[0]
        
        # oo. rent 
        elif discrete_choice == inv_v_rent:
            ## discrete choices (all fixed)
            discrete[0] = 3
            h_prime[0] = 0
            d_prime[0] = 0
            Td_prime[0] = 0
            Tda_prime[0] = 0

            ## consumption choice
            m_net_rent = m_gross_rent - par.q_r*h_tilde
            c[0] = linear_interp.interp_1d(par.grid_m,sol.c_rent[t,i_ht_best,i_y_],m_net_rent)

            ## ensure feasibility
            if c[0] > m_net_rent:
                c[0] = m_net_rent
                a[0] = 0.0
            else: 
                a[0] = m_net_rent - c[0]

    else: 
        discrete_choice = np.amax(np.array([inv_v_stay,inv_v_ref,inv_v_buy,inv_v_rent]))

        # o. stay and pay
        if discrete_choice == inv_v_stay:
            ## discrete choices (all fixed)
            discrete[0] = 0
            h_prime[0] = h
            d_prime[0] = d
            Td_prime[0] = Td
            Tda_prime[0] = Tda

            ## consumption choice
            c[0] = linear_interp.interp_2d(grid_d_prime,par.grid_m,sol.c_stay[t,i_h,:,i_Td,i_Tda,i_y_,:],
                                           d,m_gross_stay)

            ## credit constrained? 
            if c[0] > m_gross_stay:
                c[0] = m_gross_stay
                #if c[0] <= 0:
                    # do default
                a[0] = 0.0
            else:
                a[0] = m_gross_stay - c[0]

        # oo. stay and refinance
        elif discrete_choice == inv_v_ref: 
            ## fixed discrete choices
            discrete[0] = 1
            h_prime[0] = h
            Td_prime[0] = mt.Td_func(t,par)

            ## mortgage plan choice
            d_prime[0] = sol.d_prime_ref_fast[t,i_h,i_y_,i_m_gross_ref]                                 
            Tda_prime[0] = sol.Tda_prime_ref_fast[t,i_h,i_y_,i_m_gross_ref]

            ## consumption choice
            c[0] = linear_interp.interp_1d(par.grid_x,sol.c_ref_fast[t,i_h,i_y_,:],m_gross_ref)

            ## ensure feasibility
            loan = int(d_prime[0]>0)
            m_net_ref = m_gross_ref-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime[0]
            if c[0] > m_net_ref:
                c[0] = m_net_ref
                a[0] = 0.0
            else:
                a[0] = m_net_ref - c[0]

        # ooo. buy new house
        elif discrete_choice == inv_v_buy: 

            discrete[0] = 2

            ## house purchase
            h_prime[0] = sol.h_buy_fast[t,i_y_,i_m_gross_buy]

            ## mortgage plan choice
            d_prime[0] = sol.d_prime_buy_fast[t,i_y_,i_m_gross_ref]
            Td_prime[0] = mt.Td_func(t,par)
            Tda_prime[0] = sol.Tda_prime_buy_fast[t,i_y_,i_m_gross_buy]

            ## consumption choice
            c[0] = linear_interp.interp_1d(par.grid_x,sol.c_buy_fast[t,i_y_,:],m_gross_buy)

            ## ensure feasibility (come back to this later)
            loan = int(d_prime[0]>0)
            m_net_buy = m_gross_buy-(1+par.C_buy)*par.q*h_prime[0]-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime[0]
            if c[0] > m_net_buy : 
                c[0] = m_net_buy
                a[0] = 0.0
            else:
                a[0] = m_net_buy - c[0]
        
        # oooo. rent
        elif discrete_choice == inv_v_rent:
            ## discrete choices (all fixed)
            discrete[0] = 3
            h_prime[0] = 0
            d_prime[0] = 0
            Td_prime[0] = 0
            Tda_prime[0] = 0

            ## consumption choice
            m_net_rent = m_gross_rent - par.q_r*h_tilde
            c[0] = linear_interp.interp_1d(par.grid_m,sol.c_rent[t,i_ht_best,i_y_],m_net_rent)

            ## ensure feasibility
            if c[0] > m_net_rent:
                c[0] = m_net_rent
                a[0] = 0.0
            else: 
                a[0] = m_net_rent - c[0]

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
def calc_utility(sim,par):
    """ calculate utility for each individual """

    # unpack
    u = sim.utility
    
    for t in range(par.T):
        for i in prange(par.simN):
            
            u[i] += par.beta**t*utility.func(sim.c[t,i],sim.d[t,i],par)
            
@njit
def find_nearest(array,value):
    
    # end points
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    
    # initialise limits for interior points
    jl = 0                                                  
    ju = n-1

    # repeat until ju and jl are neighbour indices                              
    while (ju-jl > 1):
        jm=(ju+jl) >> 1             
        if (value >= array[jm]):
            jl=jm # and replace either the lower limit
        else:
            ju=jm # or the upper limit, as appropriate.
    
    # return index
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl