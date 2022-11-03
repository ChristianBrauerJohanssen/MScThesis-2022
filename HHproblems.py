####################
#    1. Imports    #
####################

# a. standard packages
import numpy as np
from numba import njit, prange

# b. NumEconCph packages
from consav import linear_interp, upperenvelope

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
    compute the last period post decision functions 
    inv_v_bar and q, both given by the bequest motive 
    """

    # a. unpack solution arrays
    inv_v_bar = sol.inv_v_bar[t]
    q = sol.q[t]

    # b. find terminal period
    Td_max = mt.Td_func(t,par)
    
    # c. compute post decision given by bequest 
    for i_Tda in prange(2): 
        for i_Td in prange(par.Td_shape):
            for i_h in range(par.Nh+1):
                
                # i. own or rent?
                if i_h == 0: 
                    h = 0
                else: 
                    h = par.grid_h[i_h-1]

                # ii. scale mortgage grid
                d_high = par.q*h
                grid_d = np.linspace(0,d_high,par.Nd)
                    
                for i_d in range(par.Nd):    
                    for i_a in range(par.Na):
                        
                        # o. mortgage? 
                        if Td_max < t: 
                            d = 0
                        else: 
                            d = grid_d[i_d]

                        # oo. remaining post decision states
                        Tda = i_Tda
                        a = par.grid_a[i_a]

                        # ooo. compute negative inverse post decision function
                        ab = trans.ab_plus_func(a,d,Tda,h,par)
                        inv_v_bar[i_h,i_d,i_Td,i_Tda,:,i_a] = -1/utility.bequest_func(ab,par)
                        q[i_h,i_d,i_Td,i_Tda,:,i_a] = (1+par.r)*par.beta*utility.marg_bequest_func(ab,par)


####################
# 3. Post decision # 
####################
@njit(parallel=True)
def postdecision_compute_v_bar_q(t,sol,par):
    """ compute the post-decision functions inv_v_bar and q """
    # unpack solution arrays
    inv_v_bar = sol.inv_v_bar[t]
    q = sol.q[t]

    # b. restrict loop over terminal periods
    Td_len = np.fmin(t+2,par.Td_shape) # fx 26 years old: terminal period can be 0, 55 og 56

    # c. counter
    count = 0 
    # loop over outermost post-decision state
    for i_w in prange(par.Nw): 
#m_plus_stay = np.zeros(par.Na)
#m_plus_ref = np.zeros(par.Na)
#m_net_ref = np.zeros(par.Na)
#m_plus_buy = np.zeros(par.Na)
#m_net_buy = np.zeros(par.Na)
        # a. allocate temporary containers
        m_plus_own = np.zeros(par.Na)
        m_plus_net_stay = np.zeros(par.Na)
        m_plus_gross_ref = np.zeros(par.Na)
        m_plus_gross_buy = np.zeros(par.Na) 

        m_plus_rent = np.zeros(par.Na) # container, same lenght as grid_a
        m_net_rent = np.zeros(par.Na)
        
        v_bar = np.zeros(par.Na)

        inv_v_stay_plus = np.zeros(par.Na)
        inv_v_ref_plus = np.zeros(par.Na)
        inv_v_buy_plus = np.zeros(par.Na)
        inv_v_rent_plus = np.zeros((par.Na,par.Nhtilde))

        inv_marg_u_stay_plus = np.zeros(par.Na)
        inv_marg_u_ref_plus = np.zeros(par.Na)
        inv_marg_u_buy_plus = np.zeros(par.Na)
        inv_marg_u_rent_plus = np.zeros((par.Na,par.Nhtilde))

        # b. loop over other outer post-decision states
        for i_h in prange(par.Nh+1):
            # housing stock (own or rent)
            if i_h == 0: 
                h = 0
            else: 
                h = par.grid_h[i_h-1]

            for i_Td in range(Td_len):
                    for i_Tda in range(np.fmin(par.Tda_bar,par.T-t+1)):
                        # mortgage plan and scale grid
                        Tda = i_Tda
                        Td = mt.Td_func(t,par)

                        d_prime_high = par.q*h
                        grid_d_prime = np.linspace(0,d_prime_high,par.Nd)
                        
                        for i_dp in range(par.Nd):
                            # i. permanent income
                            w = par.grid_w[i_w]

                            # ii. next period mortgage balance beginning of period
                            d_plus = trans.d_plus_func(grid_d_prime[i_dp],t,Td,Tda,par)
                        
                            # iii. initialize at zero
                            for i_a in range(par.Na):
                                v_bar[i_a] = 0.0
                                q[i_h,i_dp,i_Td,i_Tda,i_w,i_a] = 0.0

                            # iv. loop over shocks and then end-of-period assets
                            for i_shock in range(par.Nw):                
                                # o. next-period (permanent) income 
                                work = 1                                # employed next period?
                                xi = 1                                  # placeholder transitory shock
                                w_plus = trans.w_plus_func(w,xi,work,par)
                                w_plus_weight = par.w_trans[i_w,i_shock]

                                # oo. compute weight 
                                weight = w_plus_weight

                                # ooo. evaluate next period living situation
                                ## rent next period
                                    # prepare interpolator
                                prep_rent = linear_interp.interp_2d_prep(par.grid_w,w_plus,par.Na)
        
                                    # loop through rental sizes
                                for i_ht in range(par.Nhtilde):  
                                    for i_a in range(par.Na):
                                        i_ht = 0
                                        htilde = par.grid_htilde[i_ht]

                                        # find gross cash-on-hand
                                        m_plus_rent[i_a] = trans.m_plus_func(par.grid_a[i_a],w_plus,grid_d_prime[i_dp],Td,Tda,par,t)
                                        m_net_rent[i_a] = trans.m_to_mnet_rent(m_plus_rent[i_a],htilde,par)

                                    # interpolate on inverse funcs given rental choice
                                    linear_interp.interp_2d_only_last_vec_mon(prep_rent,par.grid_w,par.grid_m,
                                                                              sol.inv_v_rent[t+1,i_ht],w_plus,
                                                                              m_plus_rent,inv_v_rent_plus[:,i_ht])
                                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_rent,par.grid_w,par.grid_m,
                                                                                  sol.inv_marg_u_rent[t+1,i_ht],
                                                                                  w_plus,m_plus_rent,inv_marg_u_rent_plus[:,i_ht])
        
                                ## own next period
                                # prepare interpolator
                                prep_own = linear_interp.interp_3d_prep(grid_d_prime,par.grid_w,d_plus,w_plus,par.Na)
                                
                                # cash-on-hand
                                for i_a in range(par.Na):
                                    m_plus_own[i_a] = trans.m_plus_func(par.grid_a[i_a],w_plus,d_plus,Td,Tda,par,t)
                                    m_plus_net_stay[i_a] = m_plus_own[i_a]-par.delta*par.q*h-mt.property_tax(par.q,h,par)  #trans.m_to_mnet_stay(m_plus_own[i_a],h,par)
                                    m_plus_gross_ref[i_a] = m_plus_own[i_a]-grid_d_prime[i_dp]-par.delta*par.q*h-mt.property_tax(par.q,h,par)
                                    m_plus_gross_buy[i_a] = m_plus_own[i_a]-grid_d_prime[i_dp]+(1-par.delta+par.C_sell)*par.q*h-mt.property_tax(par.q,h,par)

                                ## print for debugging
                                #print(f'prep_own is {prep_own}')
                                #print(f'par.grid_m is {par.grid_m}')
                                #print(f'grid_d_prime is {grid_d_prime}')
                                #print(f'par.grid_w is {par.grid_w}')
                                #print(f'solution array is {sol.inv_v_buy[t+1,:,i_h,:,i_Td,i_Tda,:]}')
                                #print(f'd_plus is {d_plus}')
                                #print(f'w_plus is {w_plus}')
                                #print(f'm_plus_gross {m_plus_net_stay}')
                                #print(f'm_plus_gross_ref is {m_plus_gross_ref}')
                                #print(f'm_plus_gross_buy is {m_plus_gross_buy}')

                                # condition on owning this period (otherwise stay or ref are not in the choice set)
                                if h!=0:
                                    # interpolate to get inverse funcs for stayers
                                    linear_interp.interp_3d_only_last_vec_mon(prep_own,grid_d_prime,par.grid_w,par.grid_m,
                                                                              sol.inv_v_stay[t+1,i_h-1,:,i_Td,i_Tda],
                                                                              d_plus,w_plus,m_plus_net_stay,inv_v_stay_plus)
                                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_own,grid_d_prime,par.grid_w,par.grid_m,
                                                                                  sol.inv_marg_u_stay[t+1,i_h-1,:,i_Td,i_Tda],
                                                                                  d_plus,w_plus,m_plus_net_stay,inv_marg_u_stay_plus)

                                    # interpolate to get inverse funcs for refinancers
                                    linear_interp.interp_3d_only_last_vec_mon(prep_own,grid_d_prime,par.grid_w,par.grid_m,
                                                                              sol.inv_v_ref[t+1,i_h-1,:,i_Td,i_Tda],d_plus,
                                                                              w_plus,m_plus_gross_ref,inv_v_ref_plus)
                                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_own,grid_d_prime,par.grid_w,par.grid_m,
                                                                                  sol.inv_marg_u_ref[t+1,i_h-1,:,i_Td,i_Tda],d_plus,
                                                                                  w_plus,m_plus_gross_ref,inv_marg_u_ref_plus)
                            
                                    # interpolate on inverse funcs for buyers with house now
                                    linear_interp.interp_3d_only_last_vec_mon(prep_own,grid_d_prime,par.grid_w,par.grid_m,
                                                                              sol.inv_v_buy[t+1,i_h-1,:,i_Td,i_Tda],d_plus,
                                                                              w_plus,m_plus_gross_buy,inv_v_buy_plus)
                                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_own,grid_d_prime,par.grid_w,par.grid_m,
                                                                                  sol.inv_marg_u_buy[t+1,i_h,:,i_Td,i_Tda],d_plus,
                                                                                  w_plus,m_plus_gross_buy,inv_marg_u_buy_plus)
                                
                                # condition on renting this period (buyers need not interpolate on grid_d_prime)
                                if h == 0: 
                                    prep_buy_h0 = prep_rent = linear_interp.interp_2d_prep(par.grid_w,w_plus,par.Na)
                                    #test_grid = np.linspace(0.1,1,10)
                                    linear_interp.interp_2d_only_last_vec_mon(prep_buy_h0,par.grid_w,par.grid_m,
                                                                              sol.inv_v_buy[t+1,i_h,i_dp,i_Td,i_Tda],
                                                                              w_plus,m_plus_gross_buy,inv_v_buy_plus)
                                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_buy_h0,par.grid_w,par.grid_m,
                                                                              sol.inv_marg_u_buy[t+1,i_h,i_dp,i_Td,i_Tda],
                                                                              w_plus,m_plus_gross_buy,inv_marg_u_buy_plus)
                                
                                # oooo. max and accumulate
                                    ## find best rental choice given states
                                rent_choice = np.nan*np.zeros(par.Na)                                                            
                                for i_a in range(par.Na):
                                    discrete_rent = np.array([inv_v_rent_plus[i_a,0],
                                                             inv_v_rent_plus[i_a,1],
                                                             inv_v_rent_plus[i_a,2]])
                                    rent_choice[i_a] = np.argmax(discrete_rent)
                                    
                                    ## find best discrete choice given states
                                for i_a in range(par.Na):  
                                    if h == 0:
                                        discrete = np.array([inv_v_buy_plus[i_a],
                                                            inv_v_rent_plus[i_a,int(rent_choice[i_a])]])
                                        choice = np.argmax(discrete) # 0 = buy, 1 = rent
                                        if choice == 0:     
                                            v_plus = -1/inv_v_buy_plus[i_a]
                                            marg_u_plus = 1/inv_marg_u_buy_plus[i_a]
                                        else: 
                                            v_plus = -1/inv_v_rent_plus[i_a,int(rent_choice[i_a])]
                                            marg_u_plus = 1/inv_marg_u_rent_plus[i_a,int(rent_choice[i_a])]
                                    else:                                   
                                        discrete = np.array([inv_v_stay_plus[i_a],
                                                            inv_v_ref_plus[i_a],
                                                            inv_v_buy_plus[i_a],
                                                            inv_v_rent_plus[i_a,int(rent_choice[i_a])]])
                                        choice = np.argmax(discrete) # 0 = stay, 1 = ref, 2 = buy, 3 = rent
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
                                            v_plus = -1/inv_v_rent_plus[i_a,int(rent_choice[i_a])]
                                            marg_u_plus = 1/inv_marg_u_rent_plus[i_a,int(rent_choice[i_a])]
                                    v_bar[i_a] += weight*par.beta*v_plus
                                    q[i_h,i_dp,i_Td,i_Tda,i_w,i_a] += weight*par.beta*(1+par.r)*marg_u_plus

                            # v. transform post decision value function
                            for i_a in range(par.Na):
                                inv_v_bar[i_h,i_dp,i_Td,i_Tda,i_w,i_a] = -1/v_bar[i_a]
                            count += 1
                            if count%10**5 == 0:
                                print(f'Iteration number {count}')


####################
# 4. Stay problem  # 
####################
@njit(parallel=True)
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
    
    # c. restrict loop over terminal periods
    Td_len = np.fmin(t+2,par.Td_shape) # fx 26 years old: terminal period can be 0, 55 og 56
    count = 0
    # d. loop through states
    for i_w in prange(par.Nw):
        for i_d in prange(par.Nd): 
            for i_Tda in range(np.fmin(par.Tda_bar,par.T-t+1)):
                for i_Td in range(Td_len):
                    for i_h in range(par.Nh):

                        # i. temporary container and states
                        v_stay_vec = np.zeros(par.Nm)
                        h = par.grid_h[i_h]

                        for i_a in range(par.Na):
                            # o. post decision assets
                            a = par.grid_a[i_a]
 
                            # oo. back out optimal consumption 
                            c_endo[i_h,i_d,i_Td,i_Tda,i_w,i_a] = par.n[t]*(q_stay[i_h,i_d,i_Td,i_Tda,i_w,i_a]/(1-par.nu))**(1/-par.rho)
                            m_endo[i_h,i_d,i_Td,i_Tda,i_w,i_a] = a + c_endo[i_h,i_d,i_Td,i_Tda,i_w,i_a] + par.delta*par.q*h + mt.property_tax(par.q,h,par)
                            
                        # ii. interpolate from post decision space to beginning of period states
                        move = 0
                        rent = 0
                        negm_upperenvelope(par.grid_a,m_endo[i_h,i_d,i_Td,i_Tda,i_w,:],c_endo[i_h,i_d,i_Td,i_Tda,i_w,:],
                         inv_v_bar[i_h,i_d,i_Td,i_Tda,i_w,:],par.grid_m,c_stay[i_h,i_d,i_Td,i_Tda,i_w,:],
                         v_stay_vec,h,move,rent,t,par)
                        
                        count += 1
                        #print('iteration')
                        #print(count)
                        #print('v_stay vec:')
                        #print(v_stay_vec)
 
                        # iii. optimal value func and marg u - (negative) inverse 
                        for i_m in range(par.Nm): 
                            inv_v_stay[i_h,i_d,i_Td,i_Tda,i_w,i_m] = -1/v_stay_vec[i_m]
                            inv_marg_u_stay[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 1/utility.marg_func_nopar(c_stay[i_h,i_d,i_Td,i_Tda,i_w,i_m],
                                                                                                    par.nu,par.rho,par.n[t])


####################
# 5. Ref. problem  # 
####################
@njit
def obj_ref(d_prime,m_gross,inv_v_stay_slice,grid_m,grid_d_prime,par):
    """ interpolate on stayers solution for refinancers """

    # a. take new loan?
    loan = 0
    if d_prime > 0:
        loan = 1

    # b. net cash-on-hand
    m_net = m_gross-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime 
    #m_net = trans.m_to_mnet_ref(m,h,d,d_prime,par) m+d-par.delta*par.q*h-mt.property_tax(par.q,h,par)
    # c. value-of-choice
    return linear_interp.interp_2d(grid_m,grid_d_prime,inv_v_stay_slice,m_net,d_prime)  # we are minimizing

@njit(parallel=True)
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
    grid_m = par.grid_m
    grid_Tda = np.arange(0,np.fmin(par.Tda_bar,par.T-t+1),1)
    
    nu = par.nu
    rho = par.rho
    n = par.n[t]

    # c. restrict loop over terminal periods
    Td_len = np.fmin(t+2,par.Td_shape)

    # d. loop over outer states
    for i_w in prange(par.Nw):
        w = par.grid_w[i_w]
        for i_h in prange(par.Nh):
            h = par.grid_h[i_h]
            grid_d = np.linspace(0,par.q*h,par.Nd)
            for i_d in range(par.Nd):
                d = grid_d[i_d]
                for i_Td in range(Td_len):
                    for i_Tda in range(np.fmin(par.Tda_bar,par.T-t+1)):

                        # i. loop over cash on hand state
                        for i_m in range(par.Nm):                        
                            m = par.grid_m[i_m]
                            m_gross = m-d-par.delta*par.q*h-mt.property_tax(par.q,h,par)
                            # o. enforce financial regulation
                                ## terminal mortgage period
                            Td_new = mt.Td_func(t,par) # fx 30 years old --> t = 5, Td_new = 35
                            i_Td_new = int(Td_new - par.Td_bar) # i_Td new = 5

                                ## scale post decision mortgage grid
                            d_prime_high = np.fmin(par.omega_ltv*par.q*h,par.omega_dti*w) 
                            grid_d_prime = np.linspace(0,d_prime_high,par.Nd)

                            # oo. loop over mortage plan choices 
                            inv_v_ref_best = 0

                            for i_dp in range(par.Nd): 
                                for Tda in range(max(grid_Tda)):
                                    ## count and print 
                                    #count = count+1
                                    #if count%10**6 == 0: 
                                    #    print(f'Iteration no. {count}')

                                    # evaluate choice
                                    inv_v_ref_new = obj_ref(grid_d_prime[i_dp],m_gross,
                                                            inv_v_stay[i_h,:,i_Td_new,Tda,i_w,:],
                                                            grid_m,grid_d_prime,par)

                                    # update optimal value and choices?
                                    if inv_v_ref_new > inv_v_ref_best:
                                        inv_v_ref_best = inv_v_ref_new
                                        d_prime_best = grid_d_prime[i_dp]
                                        Tda_best = Tda
                                        i_dp_best = i_dp

                            # ooo. save optimal value and choices
                            d_prime_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = d_prime_best
                            Tda_prime_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = Tda_best

                            ## refinancer's net cash on hand equation
                            m_net = m_gross - par.Cf_ref + (1-par.Cp_ref)*d_prime_best  
                            #m_net = trans.m_to_mnet_ref(m,h,d,d_prime_best,par)
                            ## enforce non-negativity constraint
                            if m_net <= 0:
                                d_prime_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                Tda_prime_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                c_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                inv_v_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                inv_marg_u_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0        
                                continue

                            ## now interpolate on stayer consumption and value function
                            c_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = linear_interp.interp_1d(grid_m,c_stay[i_h,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
                            inv_v_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = linear_interp.interp_1d(grid_m,inv_v_stay[i_h,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
                            inv_marg_u_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 1/utility.marg_func_nopar(c_ref[i_h,i_d,i_Td,i_Tda,i_w,i_m],nu,rho,n)


####################
# 6. Buy problem   # 
####################
@njit
def obj_buy(d_prime,h_buy,m_gross,inv_v_stay_slice,grid_m,grid_d_prime,par):
    """ interpolate on stayer solution for buyers """

    # a. take new loan?
    loan = 0
    if d_prime > 0:
        loan = 1

    # b. net cash-on-hand
    m_net = m_gross-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime-(1+par.C_buy)*par.q*h_buy
    #m_net = trans.m_to_mnet_buy(m,h,d,d_prime,hbuy,par)
    # c. value-of-choice
    return (linear_interp.interp_2d(grid_m,grid_d_prime,inv_v_stay_slice,m_net,d_prime)-par.kappa)  # we are minimizing

@njit(parallel=True)
def solve_buy(t,sol,par):
    """ solve bellman equation for buyers using nvfi"""
    
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
    grid_Tda = np.arange(0,np.fmin(par.Tda_bar,par.T-t+1),1)
    
    nu = par.nu
    rho = par.rho
    n = par.n[t]
    
    # c. restrict loop over terminal mortgage periods 
    Td_len = np.fmin(t+2,par.Td_shape)
    
    # d. loop over outer states
    for i_w in prange(par.Nw):
        w = par.grid_w[i_w]
        for i_h in prange(par.Nh+1):
            # own or rent?
            if i_h == 0: 
                h = 0
            else: 
                h = grid_h[i_h-1]
            
            # scale mortgage grid
            grid_d = np.linspace(0,par.q*h,par.Nd)

            for i_d in range(par.Nd):
                d = grid_d[i_d]
                for i_Td in range(Td_len): 
                    for i_Tda in range(np.fmin(par.Tda_bar,par.T-t+1)):

                        # i. loop over cash on hand and house purchase
                        for i_m in range(par.Nm):                        
                            m = par.grid_m[i_m]
                            m_gross = m-d+(1-par.delta+par.C_sell)*par.q*h-mt.property_tax(par.q,h,par)
                            for i_hb in range(par.Nh):
                                h_buy_now = grid_h[i_hb]

                                # o. enforce financial regulation
                                    ## terminal mortgage period
                                Td_new = mt.Td_func(t,par) 
                                i_Td_new = int(Td_new - par.Td_bar)

                                    ## scale post decision mortgage grid
                                d_prime_high = np.fmin(par.omega_ltv*par.q*h_buy_now,par.omega_dti*w) 
                                grid_d_prime = np.linspace(0,d_prime_high,par.Nd)

                                # oo. loop over mortage plan choices 
                                inv_v_buy_best = 0

                                for i_dp in range(len(grid_d_prime)): 
                                    for Tda in range(max(grid_Tda)):
                                        ## count and print 
                                        #count = count+1
                                        #if count%10**6 == 0: 
                                        #    print(f'Iteration no. {count}')

                                        # evaluate choice
                                        inv_v_buy_new = obj_buy(grid_d_prime[i_dp],h_buy_now,m,
                                                                inv_v_stay[i_hb,:,i_Td,Tda,i_w,:],
                                                                grid_m,grid_d_prime,par)

                                        # update optimal value and choices?
                                        if inv_v_buy_new > inv_v_buy_best:
                                            inv_v_buy_best = inv_v_buy_new
                                            d_prime_best = grid_d_prime[i_dp]
                                            Tda_best = Tda
                                            hbuy_best = h_buy_now
                                            i_hb_best = i_hb
                                            i_dp_best = i_dp

                                # ooo. save optimal value and choices
                                d_prime_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = d_prime_best
                                Tda_prime_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = Tda_best
                                h_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = hbuy_best

                                ## buyer's net cash on hand equation
                                m_net = m_gross-par.Cf_ref+(1-par.Cp_ref)*d_prime_best-(1+par.C_buy)*par.q*hbuy_best
                                #m_net = trans.m_to_mnet_buy(m,h,d,d_prime_best,hbuy_best,par)
                                ## enforce non-negativity constraint
                                if m_net <= 0:
                                    d_prime_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                    Tda_prime_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                    h_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                    c_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                    inv_v_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0
                                    inv_marg_u_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 0        
                                    continue
                                
                                ## now interpolate on stayer consumption and value function
                                c_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = linear_interp.interp_1d(grid_m,c_stay[i_hb_best,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
                                inv_v_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = linear_interp.interp_1d(grid_m,inv_v_stay[i_hb_best,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
                                inv_marg_u_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 1/utility.marg_func_nopar(c_buy[i_h,i_d,i_Td,i_Tda,i_w,i_m],nu,rho,n)


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

    # unpack solution arrays
    inv_v_rent = sol.inv_v_rent[t]
    inv_marg_u_rent = sol.inv_marg_u_rent[t]
    c_rent = sol.c_rent[t]
    
    for i_w in prange(par.Nw):
        for i_ht in range(par.Nhtilde):
            
            # i. temporary container and states
            v_rent_vec = np.zeros(par.Nm)
            htilde = par.grid_htilde[i_ht]   

            for i_a in range(par.Na):

                # o. post decision assets
                a = par.grid_a[i_a]
                
                # oo. back out optimal consumption and net cash-on-hand
                c_endo[i_ht,i_w,i_a] = par.n[t]*(q_rent[0,0,0,0,i_w,i_a]/(1-par.nu))**(1/-par.rho)
                m_endo[i_ht,i_w,i_a] = a + c_endo[i_ht,i_w,i_a] + par.q_r*htilde

            # ii. interpolate from post decision space to beginning of period states
            move = 0    # no costs of moving when renting 
            rent = 1
            negm_upperenvelope(par.grid_a,m_endo[i_ht,i_w,:],c_endo[i_ht,i_w,:],
             inv_v_bar[0,0,0,0,i_w,:],par.grid_m,c_rent[i_ht,i_w,:],
             v_rent_vec,htilde,move,rent,t,par)
            sol.htilde[t,i_ht,i_w,:] = htilde 

            # iii. optimal value func and marg u - (negative) inverse 
            for i_m in range(par.Nm): 
                inv_v_rent[i_ht,i_w,i_m] = -1/v_rent_vec[i_m]
                inv_marg_u_rent[i_ht,i_w,i_m] = 1/utility.marg_func_nopar(c_rent[i_ht,i_w,i_m],
                                                                          par.nu,par.rho,par.n[t])