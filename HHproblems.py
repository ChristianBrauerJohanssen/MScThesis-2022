####################
#    1. Imports    #
####################

# a. standard packages
from math import nan
import numpy as np
from numba import njit, prange

# b. NumEconCph packages
from consav import golden_section_search, linear_interp, linear_interp_1d, linear_interp_2d, linear_interp_3d, upperenvelope 
 
# c. local modules
import utility
import trans
import mt

# d. prepare NEGM upper envelope step 
negm_upperenvelope = upperenvelope.create(utility.func,use_inv_w=True)

##############################
# 2. Last period and bequest #
##############################

@njit(parallel=True)
def last_period_v_bar_q(t,sol,par):
    """ 
    compute the last period post decision value function 
    and  marginal value of cash, both given by the bequest
    motive 
    """ 

    # a. unpack
    inv_v_bar = sol.inv_v_bar[t]
    q = sol.q[t]
    #q_stay = sol.q_stay[t]
    #q_rent = sol.q_rent[t]

    # b. find terminal period
    Td_max = mt.Td_func(t,par)

    # c. compute post decision given by bequest 
    for i_Tda in prange(2):
        for i_Td in prange(Td_max-par.Td_bar):
            for i_h in range(par.Nh+1):
                
                # i. own or rent?
                if i_h == 0: 
                    h = 0
                else: 
                    h = par.grid_h[i_h-1]

                # ii. find maximum loan size
                i_dmax = 0
                while par.grid_d[i_dmax] < par.q*h:
                    i_dmax += 1
                    
                for i_d in prange(i_dmax+1):    
                    for i_a in prange(par.Na):
                        
                        # o. mortgage? 
                        if i_Td < t: 
                            d = 0
                        else: 
                            d = par.grid_d[i_d]

                        # oo. remaining post decision states
                        Tda = i_Tda
                        a = par.grid_a[i_a]

                        # ooo. compute negative inverse post decision function
                        ab = trans.ab_plus_func(a,d,Tda,h,par)
                        inv_v_bar[i_a,i_h,i_d,i_Td,i_Tda,:] = -1/utility.bequest_func(ab,par)
                        q[i_a,i_h,i_d,i_Td,i_Tda,:] = par.r*par.beta*utility.marg_bequest_func(ab,par)
                        #q_stay[i_a,i_h,i_d,i_Td,i_Tda,:] = par.r*par.beta*utility.marg_bequest_func(ab,par)
                        #q_rent[i_a,i_h,i_d,i_Td,i_Tda,:] = par.r*par.beta*utility.marg_bequest_func(ab,par)
                        # ((1+par.r)*par.beta*par.thetab/(1-par.nu))**(1/-par.rho)*par.n[t]*(ab_plus+par.K) 

####################
# 3. Post decision # 
####################
@njit(parallel=True)
def postdecision_compute_v_bar_q(t,sol,par,compute_q=True):
    """ compute the post-decision functions w and/or q """
    # a. unpack 
    #inv_v_bar_stay = sol.inv_v_bar_stay[t]
    #inv_v_bar_ref = sol.inv_v_bar_ref[t]
    #inv_v_bar_buy = sol.inv_v_bar_buy[t]
    #inv_v__bar_rent = sol.inv_v_bar_rent[t]
    q = sol.q[t]

    # b. loop over outermost post-decision state
    for i_w in prange(par.Nw): 

        # i. allocate temporary containers
        m_plus = np.zeros(par.Na) # container, same lenght as grid_a
        d_plus = np.zeros(par.Na)
        v_bar = np.zeros(par.Na)

        inv_v_stay_plus = np.zeros(par.Na)
        inv_v_ref_plus = np.zeros(par.Na)
        inv_v_buy_plus = np.zeros(par.Na)
        inv_v_rent_plus = np.zeros(par.Na)

        inv_marg_u_stay_plus = np.zeros(par.Na)
        inv_marg_u_ref_plus = np.zeros(par.Na)
        inv_marg_u_buy_plus = np.zeros(par.Na)
        inv_marg_u_rent_plus = np.zeros(par.Na)
        
        # ii. loop over other outer post-decision states
        for i_h in range(par.Nh):

            # o. permanent income and durable stock
            w = par.grid_w[i_w]
            h = par.grid_h[i_h]

            # oo. initialize at zero
            for i_a in range(par.Na):
                v_bar[i_a] = 0.0
                q[i_w,i_h,i_a] = 0.0

            # ooo. loop over shocks and then end-of-period assets
            for ishock in range(par.Nw):               
                # i. 
                ## income shocks
                w_plus = par.grid_w[ishock]
                w_plus_weight = par.w_trans[i_w,ishock]
                #xi_plus = par.xi[ishock]
                #xi_plus_w = par.xi_w[ishock]

                ## housing shocks
                #z_plus = par.z[ishock]
                #z_plus_w = par.z_w[ishock]

                ## next-period income and durables
                h_plus = h

                ## prepare interpolators
                prep_stay = linear_interp.interp_3d_prep(par.grid_w,par.grid_h,w_plus,h_plus,par.Na)
                prep_rent = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)

                ## weight
                weight = w_plus_weight

                ## next-period cash-on-hand and total resources
                for i_a in range(par.Na):
        
                    m_plus[i_a] = trans.m_plus_func(par.grid_a[i_n,i_a],p_plus,xi_plus,par,t)
                
                # vi. interpolate
                linear_interp.interp_3d_only_last_vec_mon(prep_stay,par.grid_w,par.grid_h,par.grid_m,sol.inv_v_keep[t+1],p_plus,n_plus,m_plus,inv_v_stay_plus)
                linear_interp.interp_2d_only_last_vec_mon(prep_rent,par.grid_p,par.grid_x,sol.inv_v_adj[t+1],p_plus,x_plus,inv_v_adj_plus)
                if compute_q:
                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_stay,par.grid_p,par.grid_n,par.grid_m,sol.inv_marg_u_keep[t+1],p_plus,n_plus,m_plus,inv_marg_u_keep_plus)
                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_rent,par.grid_p,par.grid_x,sol.inv_marg_u_adj[t+1],p_plus,x_plus,inv_marg_u_adj_plus)
                     
                # vii. max and accumulate
                if compute_q:

                    for i_a in range(par.Na):                                
                        # 0 = stay, 1 = ref, 2 = buy, 3 = rent
                        discrete = np.array([inv_v_stay_plus[i_a],
                                            inv_v_ref_plus[i_a],
                                            inv_v_buy_plus[i_a]],
                                            inv_v_rent_plus[i_a])
                        choice = np.argmax(discrete)
                        if choice == 0:
                            v_plus = -1/inv_v_stay_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_stay_plus[i_a]
                        elif choice == 1:
                            v_plus = -1/inv_v_ref_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_ref_plus[i_a]
                        elif choice == 2:
                            v_plus = -1/inv_v_buy_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_buy_plus[i_a]
                        else: 
                            v_plus = -1/inv_v_rent_plus[i_a]
                            marg_u_plus = -1/inv_marg_u_rent_plus[i_a]
                        v_bar[i_a] += weight*par.beta*v_plus
                        q[i_p,i_n,i_a] += weight*par.beta*par.R*marg_u_plus

                else:
                    pass    
                    #for i_a in range(par.Na):
                    #    w[i_a] += weight*par.beta*(-1.0/np.fmax(inv_v__plus[i_a],inv_v_adj_plus[i_a]))
        
            # d. transform post decision value function
            for i_a in range(par.Na):
                inv_v_bar_stay[i_p,i_n,i_a] = -1/w[i_a]

####################
# 4. Stay problem  # 
####################
@njit
def solve_stay(t,sol,par): 
    """ solve bellman equation for stayers using negm """
    # a. unpack input and endogenous arrays
    c_endo = sol.c_endo_stay[t]
    m_endo = sol.m_endo_stay[t]
    inv_v_bar = sol.inv_v_bar[t]
    q_stay = sol.q[t]

    # b. unpack output
    inv_v_stay = sol.inv_v_stay[t]
    inv_marg_u_stay = sol.inv_marg_u_stay[t]
    c_stay = sol.c_stay[t]
    
    # c. set counter and restrict loops
    count = 0                           # initiate counter
    Td_max = mt.Td_func(t,par)          # current terminal mortgage period

    # d. loop through states
    for i_w in prange(par.Nw):
        for i_d in prange(par.Nd): 
            for i_Tda in prange(1): #np.fmax(par.Tda_bar,Td_max)):
                for i_Td in prange(Td_max-par.Td_bar):
                    for i_h in prange(par.Nh):
                        count = count+1
                        if count%100 == 0: 
                            print(f'Iteration no. {count}')#

                        # i. temporary container and states
                        v_stay_vec = np.zeros(par.Nm)
                        h = par.grid_h[i_h]

                        for i_a in prange(par.Na):
                            # o. post decision assets
                            a = par.grid_a[i_a]
 
                            # oo. back out optimal consumption 
                            c_endo[i_a,i_h,i_d,i_Td,i_Tda,i_w] = par.n[t]*(q_stay[i_a,i_h,i_d,i_Td,i_Tda,i_w]/(1-par.nu))**(1/-par.rho)
                            m_endo[i_a,i_h,i_d,i_Td,i_Tda,i_w] = a + c_endo[i_a,i_h,i_d,i_Td,i_Tda,i_w] + par.delta*par.q*h + mt.property_tax(par.q,h,par)
                            
                        # ii. interpolate from post decision space to beginning of period states
                        move = 0
                        rent = 0
                        negm_upperenvelope(par.grid_a,m_endo[:,i_h,i_d,i_Td,i_Tda,i_w],c_endo[:,i_h,i_d,i_Td,i_Tda,i_w],
                         inv_v_bar[:,i_h,i_d,i_Td,i_Tda,i_w],par.grid_m,c_stay[:,i_h,i_d,i_Td,i_Tda,i_w],
                         v_stay_vec,h,move,rent,t,par)
 
                        # iii. optimal value func and marg u - (negative) inverse 
                        for i_m in range(par.Nm): 
                            inv_v_stay[i_m,i_h,i_d,i_Td,i_Tda,i_w] = -1/v_stay_vec[i_m]
                            inv_marg_u_stay[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 1/utility.marg_func_nopar(c_stay[i_m,i_h,i_d,i_Td,i_Tda,i_w],
                                                                                                    par.nu,par.rho,par.n[t])
#own_shape = (par.T,par.Nm,par.Nh,par.Nd,par.Td_bar,par.Tda_bar,par.Nw)
#rent_shape = (par.T,par.Nm,par.Nhtilde,par.Nw)
#post_shape = (par.T,par.Na,par.Nh+1,par.Nd,par.Td_bar,par.Tda_bar,par.Nw)


####################
# 5. Ref. problem  # 
####################
@njit
def obj_ref(d_prime,m,d,h,inv_v_stay_slice,grid_m,grid_d_prime,par):
    """ interpolate on stayers solution for refinancers """

    # a. take new loan?
    loan = 0
    if d_prime > 0:
        loan = 1

    # b. net cash-on-hand
    m_net = m+d-par.delta*par.q*h-mt.property_tax(par.q,h,par)-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime 
    
    # c. value-of-choice
    return -linear_interp.interp_2d(grid_m,grid_d_prime,inv_v_stay_slice,m_net,d_prime)  # we are minimizing

#@njit(parallel=True)
def solve_ref(t,sol,par):
    """solve bellman equation for refinancers using nvfi"""

    # a. unpack output
    inv_v_ref = sol.inv_v_ref[t]
    inv_marg_u_ref = sol.inv_marg_u_ref[t]
    c_ref = sol.c_ref[t]
    d_prime_ref = sol.d_prime_ref[t]
    Tda_prime_ref = sol.Tda_prime_ref[t]

    # b. unpack input
    inv_v_stay = sol.inv_v_stay[t]
    c_stay = sol.c_stay[t]
    grid_d_prime = par.grid_d_prime
    grid_m = par.grid_m
    grid_Tda = np.array(range(np.fmax(par.Tda_bar+1,par.T-t)),dtype='int8')
    nu = par.nu
    rho = par.rho
    n = par.n[t]

    # c. loop over outer states
    for i_w in prange(par.Nw):
        w = par.w_grid[i_w]
        for i_h in prange(par.Nh):
            h = par.grid_h[i_h]
            for i_d in prange(par.Nd):
                d = par.grid_d[i_d]
                for i_Tda in prange(par.Tda_bar):

                    # i. loop over cash on hand state
                    for i_m in range(par.Nm):                        
                        m = par.grid_m[i_m]

                        # o. enforce financial regulation
                        ## terminal mortgage period
                        Td = mt.Td_func(t,par) 
                        i_Td = Td - par.Td_bar 

                        ## cap grid at maximum mortgage balance
                        d_prime_high = np.fmax(par.omega_ltv*par.q*h,par.omega_dti*w) 
                        i_dp_max = 0
                        while grid_d_prime[i_dp_max] < d_prime_high:
                            i_dp_max += 1
                        grid_d_prime = np.append(grid_d_prime[0:i_dp_max+1],d_prime_high)

                        # oo. loop over mortage plan choices 
                        inv_v_ref_best = 0
                        d_prime_best = np.nan
                        Tda_best = np.nan

                        for i_dp in prange(len(grid_d_prime)): 
                            for Tda in prange(max(grid_Tda)):
                                # evaluate choice
                                inv_v_ref_new = obj_ref(grid_d_prime[i_dp],m,h,d,
                                                        inv_v_stay[:,i_h,:,i_Td,Tda,i_w],
                                                        grid_m,grid_d_prime,par)

                                # update optimal value and choices?
                                if inv_v_ref_new > inv_v_ref_best:
                                    inv_v_ref_best = inv_v_ref_new
                                    d_prime_best = grid_d_prime[i_dp]
                                    Tda_best = Tda

                        # ooo. save optimal value and choices
                        d_prime_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = d_prime_best
                        Tda_prime_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = Tda_best

                        ## refinancer's net cash on hand equation
                        m_net = m - d - par.delta*par.q*h - mt.property_tax(par.q,h,par) - par.Cf_ref + (1-par.Cp_ref)*d_prime_best  

                        ## enforce non-negativity constraint
                        if m_net <= 0:
                            d_prime_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 0
                            Tda_prime_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 0
                            c_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 0
                            inv_v_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 0
                            inv_marg_u_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 0        
                            continue

                        ## now interpolate on stayer consumption and value function
                        c_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = linear_interp_1d.interp_1d(grid_m,c_stay[:,i_h,i_d,i_Td,i_Tda,i_w],m_net)

                        inv_v_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = linear_interp_1d.interp_1d(grid_m,inv_v_stay[:,i_h,i_d,i_Td,i_Tda,i_w],m_net)
                        inv_marg_u_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 1/utility.marg_func_nopar(c_ref[i_m,i_h,i_d,i_Td,i_Tda,i_w],nu,rho,n)

# own_shape = (par.T,par.Nm,par.Nh,par.Nd,par.T-par.Td_bar,par.Tda_bar,par.Nw)

####################
# 6. Buy problem   # 
####################
@njit
def obj_buy(d_prime,Tda,h_buy,i_Td,i_w,m,inv_v_keep,grid_m,grid_h,grid_d_prime,grid_Tda,kappa):
    """ interpolate on stayer solution for buyers """

    # a. net cash-on-hand
    m_net = m

    # index of housing choice
    
    # b. slice value function
    inv_v_keep_slice = inv_v_keep[:,:,:,i_Td,:,i_w]
    # own_shape = (par.T,par.Nm,par.Nh,par.Nd,par.T-par.Td_bar,par.Tda_bar,par.Nw)
    
    # c. value-of-choice
    return -(linear_interp.interp_4d(grid_m,grid_h,grid_d_prime,grid_Tda,
                                    inv_v_keep_slice,m_net,h_buy,d_prime,Tda)-kappa)  # we are minimizing

def solve_buy(t,sol,par):
    
    # a. unpack output
    inv_v_buy = sol.inv_v_buy[t]
    inv_marg_u_buy = sol.inv_marg_u_buy[t]
    c_buy = sol.c_buy[t]
    d_prime_buy = sol.d_prime_buy[t]
    Tda_prime_buy = sol.Tda_prime_buy[t]
    h_buy = sol.h_buy[t]

    # b. unpack input
    inv_v_stay = sol.inv_v_stay[t]
    c_stay = sol.c_stay[t]
    grid_m = par.grid_m
    grid_h = par.grid_h
    grid_d_prime = par.grid_d_prime
    nu = par.nu
    rho = par.rho
    n = par.n[t]
    kappa = par.kappa

    # c. loop over outer states
    for i_w in prange(par.Nw):
        for i_d in prange(par.Nd):
            for i_Tda in prange(par.Tda_bar): #np.fmax(par.Tda_bar,Td_max)):
                for i_Td in prange(Td_max-par.Td_bar):
                    for i_h in range(par.Nh+1):
                    
                        # i. own or rent?
                        if i_h == 0: 
                            h = 0
                        else: 
                            h = grid_h[i_h-1]

                        # ii. loop over cash on hand state
                        for i_m in prange(par.Nm):
                            
                            # o. compute net cash on hand
                            m = grid_m[i_m]
                            m_net = trans.m_plus_func_buy()
                    
                            # oo. enforce non-negativity constraint
                            if m_net <= 0:
                                d_prime_buy[i_m,i_h,i_d,:,:,i_w] = 0
                                c_buy[i_m,i_h,i_d,:,:,i_w] = 0
                                inv_v_buy[i_m,i_h,i_d,:,:,i_w] = 0
                                h_buy[i_m,i_h,i_d,:,:,i_w] = 0 # how to force renting???
                                inv_marg_u_buy[i_m,i_h,i_d,:,:,i_w] = 0        
                                continue
                            
                            # ooo. terminal mortgage period
                            Td = mt.Td_func(t,par)
                            i_Td = Td - par.Td_bar

                            # oooo. optimal choice of mortgage plan and house size
                            for i_hb in prange(par.Nh):
                                h_buy = grid_h[i_h]
                                d_prime_low = 0
                                d_prime_high = np.fmax(par.omega_ltv*par.q*h,par.omega_dti*w) # enforce financial regulation
                                grid_Tda = range(1,par.Tda+1) # periods of deferred amortisation

                            d_prime_ref = golden_section_search.optimizer(obj_ref,d_prime_low,d_prime_high,args=(x,inv_v_stay[i_p],grid_n,grid_m),tol=par.tol)
                            Tda_choice = 0

                            # ooooo. optimal value
                            m_net = m # buyer's net cash on hand equation 
                            c_buy[i_m,i_h,i_d,] = linear_interp.interp_2d(par.grid_n,par.grid_m,c_stay[i_p],d[i_p,i_x],m)
                            d_prime_buy = 0
                            Tda_buy = 0
                            inv_v_buy[i_p,i_x] = -obj_ref(d[i_p,i_x],x,inv_v_keep[i_p],grid_n,grid_m)
                            inv_marg_u_buy[i_p,i_x] = 1/utility.marg_func_nopar(c_ref[i_p,i_x],nu,rho,n)

                        
####################
# 7. Rent problem   # 
####################
@njit
def solve_rent(t,sol,par):
    """ solve bellman equation for renters using negm """
    # unpack input and endogenous arrays
    c_endo = sol.c_endo_rent[t]
    m_endo = sol.m_endo_rent[t]
    inv_v_bar = sol.inv_v_bar[t]
    q_rent = sol.q[t]
    #inv_v_bar_rent = sol.inv_v_bar_rent[t]

    # unpack solution arrays
    inv_v_rent = sol.inv_v_rent[t]
    inv_marg_u_rent = sol.inv_marg_u_rent[t]
    htilde = sol.htilde[t]
    c_rent = sol.c_rent[t]
    
    for i_w in prange(par.Nw):
        for i_ht in prange(par.Nhtilde):

            # i. temporary container and states
            v_rent_vec = np.zeros(par.Nm)
            htilde = par.grid_htilde[i_ht]   

            for i_a in range(par.Na):

                # o. post decision assets
                a = par.grid_a[i_a]
                
                # oo. back out optimal consumption 
                c_endo[i_a,i_ht] = par.n[t]*(q_rent[i_a,0,0,0,0,i_w]/(1-par.nu))**(1/-par.rho)
                m_endo[i_a,i_ht] = a + c_endo[i_a,i_ht,i_w] + par.q_r*htilde

                #post_shape = (par.T,par.Na,par.Nh+1,par.Nd,par.Td_bar,par.Tda_bar,par.Nw)

            # ii. interpolate from post decision space to beginning of period states
            move = 0    # no costs of moving when renting 
            rent = 1
            negm_upperenvelope(par.grid_a,m_endo[:,i_ht,i_w],c_endo[:,i_ht,i_w],
             inv_v_bar[:,0,0,0,0,i_w],par.grid_m,c_rent[:,i_ht,i_w],
             v_rent_vec,htilde,move,rent,t,par)
            sol.htilde[:,i_w] = htilde 

            # iii. optimal value func and marg u - (negative) inverse 
            for i_m in range(par.Nm): 
                inv_v_rent[i_m,i_ht,i_w] = -1/v_rent_vec[i_m]
                inv_marg_u_rent[i_m,i_ht,i_w] = 1/utility.marg_func_nopar(c_rent[i_m,i_ht,i_w],
                                                                          par.nu,par.rho,par.n[t])