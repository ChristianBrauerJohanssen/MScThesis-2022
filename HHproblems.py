####################
#    1. Imports    #
####################

# a. standard packages
import numpy as np
from numba import njit, prange

# b. NumEconCph packages
from consav import golden_section_search, upperenvelope, linear_interp

# c. local modules
import utility
import trans
import mt

##############################
# 2. Last period and bequest #
##############################

## a. objective
#@njit
#def obj_last_period(c,m,h,n,move,rent,par):
#    """ objective function in last period """
#    
#    # implied consumption (rest)
#    ab = m - c
#    
#    last = -utility.func(c,h,n,move,rent,par) + utility.bequest_func(ab,par)
#
#    return last

@njit(parallel=True)
def solve_last_period(t,sol,rent,par):
    """ solve the problem in the last period """ 

    # unpack
    c_endo = sol.c_endo[t]
    m_endo = sol.m_endo[t]

    inv_v_stay = sol.inv_v_stay[t]
    inv_marg_u_stay = sol.inv_marg_u_stay[t]
    c_stay = sol.c_stay[t]
    inv_v_adj = sol.inv_v_adj[t]

    inv_marg_u_rent = sol.inv_marg_u_rent[t]
    htilde = sol.htilde[t]
    c_rent = sol.c_rent[t]

    # a. stay
    for i_h in prange(par.Nh):
        # i. states
        h = par.grid_n[i_h]
        for i_a in prange(par.Na):
            # i. states
            a = par.grid_a[i_a]
            ab_plus = (1+par.r)*a - (1-par.C_sell-par.delta)*par.q*h - mt.property_tax(par.q,h)
            
            # back out optimal consumption 
            c_endo[a,h] = ((1+r)*par.beta*par.thetab/(1-par.nu))**(1/-par.rho)*par.n[t]*(ab_plus+par.K)
            m_endo[a,h] = a + c_endo[a,h]

            #post_shape: (par.T-1,par.Na,par.Nh,par.Nd,par.Td_bar,par.Tda_bar,par.Nw)

            # interpolate with upper envelope step 
            negm_upperenvelope = upperenvelope.create(utility.func,use_inv_w=True)
       
            # ii. optimal choice
            c_stay[i_p,i_n,i_m] = m
         # iii. optimal value
            v_stay = utility.func(c_stay[i_p,i_n,i_m],n,par)
            inv_v_stay[i_p,i_n,i_m] = -1.0/v_stay
            inv_marg_u_stay[i_p,i_n,i_m] = 1.0/utility.marg_func(c_stay[i_p,i_n,i_m],n,par)

    # b. rent
    for i_ht in prange(par.Nhtilde):
        for i_x in range(par.Nx):
            
            # i. states
            x = par.grid_x[i_x]

            if x == 0: # forced c = d = 0
                d_adj[i_p,i_x] = 0
                c_adj[i_p,i_x] = 0
                inv_v_adj[i_p,i_x] = 0
                inv_marg_u_adj[i_p,i_x] = 0
                continue

            # ii. optimal choices
            d_low = np.fmin(x/2,1e-8)
            d_high = np.fmin(x,par.n_max)            
            d_adj[i_p,i_x] = golden_section_search.optimizer(obj_last_period,d_low,d_high,args=(x,par),tol=par.tol)
            c_adj[i_p,i_x] = x-d_adj[i_p,i_x]

            # iii. optimal value
            v_adj = -obj_last_period(d_adj[i_p,i_x],x,par)
            inv_v_adj[i_p,i_x] = -1.0/v_adj
            inv_marg_u_adj[i_p,i_x] = 1.0/utility.marg_func(c_adj[i_p,i_x],d_adj[i_p,i_x],par)

####################
# 3. Post decision # 
####################
@njit(parallel=True)
def postdecision_compute_wq(t,sol,par,compute_q=True):
    """ compute the post-decision functions w and/or q """
    # unpack 
    inv_w = sol.inv_w[t]
    q = sol.q[t]

    # loop over outermost post-decision state
    for i_w in prange(par.Nw):

        # allocate temporary containers
        m_plus = np.zeros(par.Na) # container, same lenght as grid_a
        d_plus = np.zeros(par.Na)
        w = np.zeros(par.Na)

        inv_v_stay_plus = np.zeros(par.Na)
        inv_marg_u_stay_plus = np.zeros(par.Na)
        #inv_v_adj_plus = np.zeros(par.Na)
        #inv_marg_u_adj_plus = np.zeros(par.Na)
        
        # loop over other outer post-decision states
        for i_h in range(par.Nh):

            # a. permanent income and durable stock
            w = par.grid_w[i_w]
            h = par.grid_h[i_h]

            # b. initialize at zero
            for i_a in range(par.Na):
                v_bar[i_a] = 0.0
                q[i_w,i_h,i_a] = 0.0

            # c. loop over shocks and then end-of-period assets
            for ishock in range(par.Nshocks):
                
                # i. 
                ## income shocks
                psi_plus = par.psi[ishock]
                psi_plus_w = par.psi_w[ishock]
                xi_plus = par.xi[ishock]
                xi_plus_w = par.xi_w[ishock]

                ## housing shocks
                z_plus = par.z[ishock]
                z_plus_w = par.z_w[ishock]

                # ii. next-period income and durables
                p_plus = trans.p_plus_func(p,psi_plus,par,t)
                n_plus = trans.n_plus_func(n,par,z_plus)

                # iii. prepare interpolators
                prep_keep = linear_interp.interp_3d_prep(par.grid_p,par.grid_n,p_plus,n_plus,par.Na)
                prep_adj = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)

                # iv. weight
                weight = psi_plus_w*xi_plus_w*z_plus_w

                # v. next-period cash-on-hand and total resources
                for i_a in range(par.Na):
        
                    m_plus[i_a] = trans.m_plus_func(par.grid_a[i_n,i_a],p_plus,xi_plus,par,t)
                
                # vi. interpolate
                linear_interp.interp_3d_only_last_vec_mon(prep_keep,par.grid_p,par.grid_n,par.grid_m,sol.inv_v_keep[t+1],p_plus,n_plus,m_plus,inv_v_keep_plus)
                linear_interp.interp_2d_only_last_vec_mon(prep_adj,par.grid_p,par.grid_x,sol.inv_v_adj[t+1],p_plus,x_plus,inv_v_adj_plus)
                if compute_q:
                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_keep,par.grid_p,par.grid_n,par.grid_m,sol.inv_marg_u_keep[t+1],p_plus,n_plus,m_plus,inv_marg_u_keep_plus)
                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_adj,par.grid_p,par.grid_x,sol.inv_marg_u_adj[t+1],p_plus,x_plus,inv_marg_u_adj_plus)
                     
                # vii. max and accumulate
                if compute_q:

                    for i_a in range(par.Na):                                

                        keep = inv_v_keep_plus[i_a] > inv_v_adj_plus[i_a]
                        if keep:
                            v_plus = -1/inv_v_keep_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_keep_plus[i_a]
                        else:
                            v_plus = -1/inv_v_adj_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_adj_plus[i_a]

                        w[i_a] += weight*par.beta*v_plus
                        q[i_p,i_n,i_a] += weight*par.beta*par.R*marg_u_plus

                else:

                    for i_a in range(par.Na):
                        w[i_a] += weight*par.beta*(-1.0/np.fmax(inv_v_keep_plus[i_a],inv_v_adj_plus[i_a]))
        
            # d. transform post decision value function
            for i_a in range(par.Na):
                inv_w[i_p,i_n,i_a] = -1/w[i_a]

####################
# 4. Stay problem  # 
####################
@njit
def solve_stay(t,sol,par):
    pass 

####################
# 5. Ref. problem  # 
####################
@njit
def solve_ref(t,sol,par):
    pass 

####################
# 6. Buy problem   # 
####################
def solve_buy(t,sol,par):
    pass

####################
# 7. Rent problem   # 
####################
@njit
def solve_rent(t,sol,par):
    pass


