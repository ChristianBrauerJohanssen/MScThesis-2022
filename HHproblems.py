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

# d. prepare NEGM upper envelope step 
negm_upperenvelope = upperenvelope.create(utility.func,use_inv_w=True)

##############################
# 2. Last period and bequest #
##############################

@njit(parallel=True)
def last_period_v_bar_q(t,sol,par):
    """ 
    compute the last period post decision value 
    function given by the bequest motive 
    """ 

    # a. unpack
    inv_v_bar = sol.inv_v_bar[t]
    q_stay = sol.q_stay[t]
    q_rent = sol.q_rent[t]

    # c. compute post decision given by bequest 
    for i_Tda in prange(2):
        for i_Td in prange(2):
            for i_h in range(par.Nh+1):
                
                # own or rent?
                if i_h == 0: 
                    h = 0
                else: 
                    h = par.grid_h[i_h-1]

                # find maximum loan size
                i_dmax = 0
                while par.grid_d[i_dmax] < h:
                    i_dmax += 1
                    
                for i_d in prange(i_dmax+1):    
                    for i_a in prange(par.Na):
                        
                        # mortgage? 
                        if i_Td == 0: 
                            d = 0
                        else: 
                            d = par.grid_d[i_d]

                        # remaining post decision states
                        Tda = i_Tda
                        a = par.grid_a[i_a]

                        # compute negative inverse post decision function
                        ab = trans.ab_plus_func(a,d,Tda,h,par)
                        inv_v_bar[i_a,i_h,i_d,i_Td,i_Tda,:] = -1/utility.bequest_func(ab,par)
                        q_stay[i_a,i_h,i_d,i_Td,i_Tda,:] = utility.marg_bequest_func(ab,par)
                        q_rent[i_a,i_h,i_d,i_Td,i_Tda,:] = utility.marg_bequest_func(ab,par)
                        # ((1+par.r)*par.beta*par.thetab/(1-par.nu))**(1/-par.rho)*par.n[t]*(ab_plus+par.K) 

####################
# 3. Post decision # 
####################
@njit(parallel=True)
def postdecision_compute_wq(t,sol,par,compute_q=True):
    """ compute the post-decision functions w and/or q """
    # unpack 
    inv_v_bar_stay = sol.inv_v_bar_stay[t]
    inv_v_bar_ref = sol.inv_v_bar_ref[t]
    inv_v_bar_buy = sol.inv_v_bar_buy[t]
    inv_v__bar_rent = sol.inv_v_bar_rent[t]
    q = sol.q[t]

    # loop over outermost post-decision state
    for i_w in prange(par.Nw): 

        # allocate temporary containers
        m_plus = np.zeros(par.Na) # container, same lenght as grid_a
        d_plus = np.zeros(par.Na)
        v_bar = np.zeros(par.Na)

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

                # ii. next-period income and durables
                h_plus = h

                # iii. prepare interpolators
                prep_stay = linear_interp.interp_3d_prep(par.grid_w,par.grid_h,w_plus,h_plus,par.Na)
                #prep_rent = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)

                # iv. weight
                weight = w_plus_weight

                # v. next-period cash-on-hand and total resources
                for i_a in range(par.Na):
        
                    m_plus[i_a] = trans.m_plus_func(par.grid_a[i_n,i_a],p_plus,xi_plus,par,t)
                
                # vi. interpolate
                linear_interp.interp_3d_only_last_vec_mon(prep_stay,par.grid_w,par.grid_h,par.grid_m,sol.inv_v_keep[t+1],p_plus,n_plus,m_plus,inv_v_stay_plus)
                linear_interp.interp_2d_only_last_vec_mon(prep_adj,par.grid_p,par.grid_x,sol.inv_v_adj[t+1],p_plus,x_plus,inv_v_adj_plus)
                if compute_q:
                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_stay,par.grid_p,par.grid_n,par.grid_m,sol.inv_marg_u_keep[t+1],p_plus,n_plus,m_plus,inv_marg_u_keep_plus)
                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_adj,par.grid_p,par.grid_x,sol.inv_marg_u_adj[t+1],p_plus,x_plus,inv_marg_u_adj_plus)
                     
                # vii. max and accumulate
                if compute_q:

                    for i_a in range(par.Na):                                

                        keep = inv_v_keep_plus[i_a] > inv_v_adj_plus[i_a]
                        if keep:
                            v_stay_plus = -1/inv_v_stay_plus[i_a]
                            marg_u_stay_plus = 1/inv_marg_u_stay_plus[i_a]
                        else:
                            v_stay_plus = -1/inv_v_adj_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_adj_plus[i_a]

                        w[i_a] += weight*par.beta*v_plus
                        q[i_p,i_n,i_a] += weight*par.beta*par.R*marg_u_plus

                else:

                    for i_a in range(par.Na):
                        w[i_a] += weight*par.beta*(-1.0/np.fmax(inv_v__plus[i_a],inv_v_adj_plus[i_a]))
        
            # d. transform post decision value function
            for i_a in range(par.Na):
                inv_v_bar_stay[i_p,i_n,i_a] = -1/w[i_a]

####################
# 4. Stay problem  # 
####################
@njit
def solve_stay(t,sol,par):
    pass 


    #c_endo = sol.c_endo[t]
    #m_endo = sol.m_endo[t]
    
    #inv_v_stay = sol.inv_v_stay[t]
    #inv_marg_u_stay = sol.inv_marg_u_stay[t]
    #c_stay = sol.c_stay[t]
    #
    #inv_v_rent = sol.inv_v_rent[t]
    #inv_marg_u_rent = sol.inv_marg_u_rent[t]
    #htilde = sol.htilde[t]
    #c_rent = sol.c_rent[t]
    #c_endo_rent = sol.c_endo_rent[t] 
    #m_endo_rent = sol.m_endo_rent[t] 
    #inv_v_bar_rent = np.zeros(par.Na,par.Nhtilde,par.Nw)

    ## b. set counter and restrict loops
    #count = 0                           # initiate counter
    #Td = mt.Td_func(t,par)              # find relevant term lengths
    #Td_min = np.fmax(1,t-Td)
    #Td_max = np.fmin(par.T,t+Td)

#    # b. last period stay
#    for i_w in prange(par.Nw):
#        for i_d in prange(par.Nd): 
#            for i_Tda in prange(np.fmax(par.Tda_bar,Td_max)):
#                for i_Td in prange(Td_min,Td_max+1):
#                    for i_h in prange(par.Nh):
#                        count = count+1
#                        if count%100 == 0: 
#                            print(f'Iteration no. {count}')#

#                        # i. temporary container and states
#                        v_ast_vec = np.zeros(par.Nm)
#                        h = par.grid_h[i_h]#

#                        for i_a in prange(par.Na):
#                            # assets and bequest
#                            a = par.grid_a[i_a]
#                            ab_plus = trans.ab_plus_func(a,h,par)
# 
#                            # compute neg inverse of implied post decision
#                            inv_v_bar[i_a,i_h,i_d,i_Td,i_Tda,i_w] = -1/par.beta*utility.bequest_func(ab_plus,par,t)
# 
#                            # back out optimal consumption 
#                            c_endo[i_a,i_h,i_d,i_Td,i_Tda,i_w] = ((1+par.r)*par.beta*par.thetab/(1-par.nu))**(1/-par.rho)*par.n[t]*(ab_plus+par.K)
#                            m_endo[i_a,i_h,i_d,i_Td,i_Tda,i_w] = a + c_endo[i_a,i_h,i_d,i_Td,i_Tda,i_w]
# 
#                        # ii. interpolate from post decision space to beginning of period states
#                        move = 0
#                        rent = 0
#                        negm_upperenvelope(par.grid_a,m_endo[:,i_h,i_d,i_Td,i_Tda,i_w],c_endo[:,i_h,i_d,i_Td,i_Tda,i_w],
#                         inv_v_bar[:,i_h,i_d,i_Td,i_Tda,i_w],par.grid_m,c_stay[:,i_h,i_d,i_Td,i_Tda,i_w],
#                         v_ast_vec,h,move,rent,t,par)
# 
#                        # iii. optimal value and negative inverse 
#                        for i_m in range(par.Nm): 
#                            inv_v_stay[i_m,i_h,i_d,i_Td,i_Tda,i_w] = -1/v_ast_vec[i_m]
#                            inv_marg_u_stay[i_m,i_h,i_d,i_Td,i_Tda,i_w] = 1/utility.marg_func_nopar(c_stay[i_m,i_h,i_d,i_Td,
#                                                                                                    i_Tda,i_w],par.nu,
#                                                                                                    par.rho,par.n[t])

#own_shape = (par.T,par.Nm,par.Nh,par.Nd,par.Td_bar,par.Tda_bar,par.Nw)
#rent_shape = (par.T,par.Nm,par.Nw)
#post_shape = (par.T,par.Na,par.Nh,par.Nd,par.Td_bar,par.Tda_bar,par.Nw)

    ## b. rent
    #for i_ht in prange(par.Nhtilde):
    #    
    #    # i. temporary container and states
    #    v_ast_vec = np.zeros(par.Nm)
    #    htilde = par.grid_htilde[i_ht]
    #    
    #    for i_a in range(par.Na):
    #        
    #        # assets and bequest
    #        a = par.grid_a[i_a]
    #        ab_plus = trans.ab_plus_func(a,0,par)
    #        
    #        # compute implied post decision
    #        inv_v_bar_rent[i_a,i_h,i_w] = -1/par.beta*utility.bequest_func(ab_plus,par,t)

    #        # back out optimal consumption 
    #        c_endo_rent[i_a,i_ht] = ((1+par.r)*par.beta*par.thetab/(1-par.nu))**(1/-par.rho)*par.n[t]*(ab_plus+par.K)
    #        m_endo_rent[i_a,i_ht] = a + c_endo[i_a,i_ht]

    #    # ii. interpolate from post decision space to beginning of period states
    #    move = 0
    #    rent = 1
    #    negm_upperenvelope(par.grid_a,m_endo[:,i_h,i_d,i_Td,i_Tda,i_w],c_endo_rent[:,i_ht,i_w],
    #     inv_v_bar_rent[:,i_ht,i_w],par.grid_m,c_rent[:,i_ht,i_w],
    #     v_ast_vec,htilde,move,rent,t,par)

    #    # iii. optimal value and negative inverse 
    #    for i_m in range(par.Nm): 
    #        inv_v_rent[i_m,i_ht,i_w] = -1/v_ast_vec[i_m]
    #        inv_marg_u_rent[i_m,i_ht,i_w] = 1/utility.marg_func_nopar(c_rent[i_m,i_ht,i_w],par.nu,par.rho,par.n[t])
    #        
    #    # iii. optimal value
    #    v_adj = 0
    #    inv_v_rent[i_p,i_x] = -1.0/v_adj
    #    inv_marg_u_rent[i_p,i_x] = 1.0/utility.marg_func(c_rent[i_p,i_x],d_adj[i_p,i_x],par)



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


