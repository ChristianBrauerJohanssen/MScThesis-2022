####################
#    1. Imports    #
####################

# a. standard packages
import numpy as np
from numba import njit, prange

# b. NumEconCph packages
from consav import golden_section_search
from consav import linear_interp

# c. local modules
import utility
import trans

##############################
# 2. Last period and bequest #
##############################

# a. objective
@njit
def obj_last_period(c,m,h,n,move,rent,par):
    """ objective function in last period """
    
    # implied consumption (rest)
    ab = m - c
    
    last = -utility.func(c,h,n,move,rent,par) + utility.bequest_func(ab,par)

    return last

@njit(parallel=True)
def solve_last_period(t,sol,par):
    """ solve the problem in the last period """

    # unpack
    inv_v_stay = sol.inv_v_stay[t]
    inv_marg_u_stay = sol.inv_marg_u_stay[t]
    c_stay = sol.c_stay[t]
    inv_v_adj = sol.inv_v_adj[t]

    inv_marg_u_adj = sol.inv_marg_u_adj[t]
    d_adj = sol.d_adj[t]
    c_adj = sol.c_adj[t]

    # a. stay
    for i_p in prange(par.Np):
        for i_n in range(par.Nn):
            for i_m in range(par.Nm):
                            
                # i. states
                n = par.grid_n[i_n]
                m = par.grid_m[i_m]

                if m == 0: # forced c = 0 
                    c_stay[i_p,i_n,i_m] = 0
                    inv_v_stay[i_p,i_n,i_m] = 0
                    inv_marg_u_stay[i_p,i_n,i_m] = 0
                    continue
                
                # ii. optimal choice
                c_stay[i_p,i_n,i_m] = m

                # iii. optimal value
                v_stay = utility.func(c_stay[i_p,i_n,i_m],n,par)
                inv_v_stay[i_p,i_n,i_m] = -1.0/v_stay
                inv_marg_u_stay[i_p,i_n,i_m] = 1.0/utility.marg_func(c_stay[i_p,i_n,i_m],n,par)

    # b. adj
    for i_p in prange(par.Np):
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
@njit
def postdecision_compute_wq(t,sol,par,compute_q=True):
    pass

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


