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
    v_bar and q, both given by the bequest motive 
    """

    # a. unpack solution arrays
    inv_v_bar = sol.inv_v_bar[t]
    q = sol.q[t]

    # b. find terminal period
    Td_max = mt.Td_func(t,par)
    Td_len = np.fmin(t+2,par.Td_shape)
    
    # c. compute post decision given by bequest
    for i_Tda in prange(2): 
        for i_Td in prange(Td_len):
            for i_h in range(par.Nh+1):
                
                # i. own or rent?
                if i_h == 0: 
                    h = 0
                else: 
                    h = par.grid_h[i_h-1]
                
                # ii. scale mortgage grid
                d_high = par.q*h #np.fmin(par.omega_ltv*par.q*h,par.omega_dti*y)
                grid_d = np.linspace(0,d_high,par.Nd)
                
                for i_d in range(par.Nd):    
                    for i_a in range(par.Na):
                        
                        ## o. mortgage 
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
def postdecision_compute_v_bar_q_rent(t,sol,par):
    """ compute the post-decision functions w and/or q for renters """
    # unpack solution arrays 
    inv_v_rent_plus = sol.inv_v_rent_plus[t] 
    inv_marg_u_rent_plus = sol.inv_marg_u_rent_plus[t] 
    rent_choice = sol.rent_choice[t]

    inv_v_bar = sol.inv_v_bar[t,0]
    q = sol.q[t,0]

    # loop over outermost post-decision state
    for i_w in prange(par.Nw): 
        
        # a. allocate temporary container
        m_plus_rent = np.zeros(par.Na)
        
        m_plus_gross_buy = np.zeros(par.Na)
        inv_v_buy_plus = np.zeros(par.Na)
        inv_marg_u_buy_plus = np.zeros(par.Na)

        v_bar = np.zeros(par.Na)

        # iii. initialise at zero
        for i_a in range(par.Na):
            q[:,:,:,i_w,i_a] = 0.0

        # b. loop over shocks and then end-of-period assets
        for i_shock in range(par.Nw):                
            # i. next-period income
            p = par.grid_w[i_w] 
            p_plus = par.grid_w[i_shock]
            y_plus = trans.p_to_y_func(i_y=i_w,p=p_plus,p_lag=p,t=t+1,par=par)

             # ii. compute weight 
            p_plus_weight = par.w_trans[i_w,i_shock]
            weight = p_plus_weight

            # iii. evaluate next period living situation
            ## rent next period
            for i_ht in range(par.Nhtilde):  
                for i_a in range(par.Na):
                    # find gross cash-on-hand
                    m_plus_rent[i_a] = np.fmin(trans.m_plus_func(par.grid_a[i_a],y_plus,0,0,0,par,t+1) - par.q_r*par.grid_htilde[i_ht],par.a_max) # t+1?
                
                # interpolate on inverse funcs given rental choice
                linear_interp.interp_1d_vec(par.grid_m,sol.inv_v_rent[t+1,i_ht,i_shock],
                                            m_plus_rent,inv_v_rent_plus[i_w,i_shock,:,i_ht])
                linear_interp.interp_1d_vec(par.grid_m,sol.inv_marg_u_rent[t+1,i_ht,i_shock],
                                            m_plus_rent,inv_marg_u_rent_plus[i_w,i_shock,:,i_ht])
            
            ## buy next period
            for i_a in range(par.Na):
                # gross cash-on-hand
                m_plus_gross_buy[i_a] = np.fmin(trans.m_plus_func(par.grid_a[i_a],y_plus,0,0,0,par,t+1),par.a_max) # t+1?

            ## interpolate on inverse funcs for buyers
            linear_interp.interp_1d_vec(par.grid_x,sol.inv_v_buy_fast[t+1,i_shock],m_plus_gross_buy,inv_v_buy_plus)
            linear_interp.interp_1d_vec(par.grid_x,sol.inv_marg_u_buy_fast[t+1,i_shock],m_plus_gross_buy,inv_marg_u_buy_plus)
                                
            # iv. find and save best (rental) choice given states                                                         
            for i_a in range(par.Na):
                discrete_rent = np.array([inv_v_rent_plus[i_w,i_shock,i_a,0],
                                          inv_v_rent_plus[i_w,i_shock,i_a,1],
                                          inv_v_rent_plus[i_w,i_shock,i_a,2]])
                rent_choice[i_w,i_shock,i_a] = np.argmax(discrete_rent)
            
            for i_a in range(par.Na):
                discrete = np.array([inv_v_buy_plus[i_a],
                                    inv_v_rent_plus[i_w,i_shock,i_a,int(rent_choice[i_w,i_shock,i_a])]])
                choice = np.argmax(discrete) # 0 = buy, 1 = rent
                #assert discrete[choice] > 0, print(f'zero inverse value function best choice at {i_w, i_shock, i_a, y_plus}. The array is {discrete}')  
                if choice == 0:     
                    v_plus = -1/inv_v_buy_plus[i_a]
                    marg_u_plus = 1/inv_marg_u_buy_plus[i_a]
                else: 
                    v_plus = -1/inv_v_rent_plus[i_w,i_shock,i_a,int(rent_choice[i_w,i_shock,i_a])]
                    marg_u_plus = 1/inv_marg_u_rent_plus[i_w,i_shock,i_a,int(rent_choice[i_w,i_shock,i_a])]
                v_bar[i_a] += weight*par.beta*v_plus
                q[:,:,:,i_w,i_a] += weight*par.beta*(1+par.r)*marg_u_plus
            
            #assert np.any(q[:,:,:,i_w,i_a]) > 0, print(f'zero post decision marginal value of cash at {i_w, i_shock, i_a, y_plus}. The entries {q[0,0,0,i_w,i_a]}')                      
        # v. transform post decision value function
        for i_a in range(par.Na):
            inv_v_bar[:,:,:,i_w,i_a] = -1/v_bar[i_a]
            
@njit(parallel=True)
def postdecision_compute_v_bar_q_own(t,sol,par):
    """ compute the post-decision functions w and/or q for owners """
    # unpack solution arrays
    inv_v_bar = sol.inv_v_bar[t,1:] # 0 is renter solution
    q = sol.q[t,1:] # 0 is renter solution

    inv_v_rent_plus = sol.inv_v_rent_plus[t] # from renter's post decision computation
    inv_marg_u_rent_plus = sol.inv_marg_u_rent_plus[t] # from renter's post decision computation
    rent_choice = sol.rent_choice[t] # from renter's post decision computation

    # restrict loop over terminal periods
    Td_len = np.fmin(t+2,par.Td_shape) # fx 26 years old: terminal period can be 0, 55 og 56

    # loop over outermost post-decision state
    for i_w in prange(par.Nw): 

        # a. allocate temporary containers
        m_plus_stay = np.zeros(par.Na)
        m_plus_gross_ref = np.zeros(par.Na)
        m_plus_gross_buy = np.zeros(par.Na) 

        v_bar = np.zeros(par.Na)

        inv_v_stay_plus = np.zeros(par.Na)
        inv_v_ref_plus = np.zeros(par.Na)
        inv_v_buy_plus = np.zeros(par.Na)

        inv_marg_u_stay_plus = np.zeros(par.Na)
        inv_marg_u_ref_plus = np.zeros(par.Na)
        inv_marg_u_buy_plus = np.zeros(par.Na)

        # b. loop over other outer post-decision states
        for i_h in prange(par.Nh):
            h = par.grid_h[i_h]

            for i_Td in range(Td_len):
                    for i_Tda in range(np.fmin(par.Tda_bar,par.T-t+1)):
                        # mortgage plan and scale grid
                        Tda = i_Tda
                        Td = mt.Td_func(t,par)

                        d_prime_high = par.q*h
                        grid_d_prime = np.linspace(0,d_prime_high,par.Nd) # end-of-this period mortgage balance
                        
                        for i_dp in range(par.Nd):

                            # i. next period mortgage balance
                            d_plus = trans.d_plus_func(grid_d_prime[i_dp],t+1,Td,Tda,par)
                        
                            # ii. initialise at zero
                            for i_a in range(par.Na):
                                v_bar[i_a] = 0.0
                                q[i_h,i_dp,i_Td,i_Tda,i_w,i_a] = 0.0 

                            # iii. loop over shocks and then end-of-period assets
                            for i_shock in range(par.Nw):                
                                # o. next-period income
                                p = par.grid_w[i_w] 
                                p_plus = par.grid_w[i_shock]
                                y_plus = trans.p_to_y_func(i_y=i_w,p=p_plus,p_lag=p,t=t+1,par=par)
                                
                                # oo. compute weight 
                                p_plus_weight = par.w_trans[i_w,i_shock]
                                weight = p_plus_weight

                                # ooo. evaluate next period living situation
                                ## prepare interpolator
                                prep_stay = linear_interp.interp_2d_prep(grid_d_prime,d_plus,par.Na)
                                
                                ## cash-on-hand
                                for i_a in range(par.Na):                     
                                    m_plus_stay[i_a] = np.fmin(trans.m_plus_func(par.grid_a[i_a],y_plus,grid_d_prime[i_dp],Td,Tda,par,t+1) - par.delta*par.q*h - mt.property_tax(par.q,h,par),par.a_max) 
                                    m_plus_gross_ref[i_a] = np.fmax(par.x_min,np.fmin(m_plus_stay[i_a] - grid_d_prime[i_dp],par.a_max))
                                    m_plus_gross_buy[i_a] = np.fmax(par.x_min,np.fmin(m_plus_stay[i_a] - grid_d_prime[i_dp] + (1-par.C_sell)*par.q*h,par.a_max))

                                ## interpolate to get inverse funcs for stayers
                                linear_interp.interp_2d_only_last_vec_mon(prep_stay,grid_d_prime,par.grid_m,
                                                                          sol.inv_v_stay[t+1,i_h,:,i_Td,i_Tda,i_shock],
                                                                          d_plus,m_plus_stay,inv_v_stay_plus)
                                linear_interp.interp_2d_only_last_vec_mon_rep(prep_stay,grid_d_prime,par.grid_m,
                                                                              sol.inv_marg_u_stay[t+1,i_h,:,i_Td,i_Tda,i_shock],
                                                                              d_plus,m_plus_stay,inv_marg_u_stay_plus)
                                ## interpolate to get inverse funcs for refinancers
                                linear_interp.interp_1d_vec(par.grid_x,sol.inv_v_ref_fast[t+1,i_h,i_shock],
                                                            m_plus_gross_ref,inv_v_ref_plus)
                                linear_interp.interp_1d_vec(par.grid_x,sol.inv_marg_u_ref_fast[t+1,i_h,i_shock],
                                                            m_plus_gross_ref,inv_marg_u_ref_plus)
                            
                                ## interpolate on inverse funcs for buyers
                                linear_interp.interp_1d_vec(par.grid_x,sol.inv_v_buy_fast[t+1,i_shock],m_plus_gross_buy,inv_v_buy_plus)
                                linear_interp.interp_1d_vec(par.grid_x,sol.inv_marg_u_buy_fast[t+1,i_shock],m_plus_gross_buy,inv_marg_u_buy_plus)
                                
                                # oooo. max and accumulate - find best discrete choice given states
                                for i_a in range(par.Na):                                    
                                    discrete = np.array([inv_v_stay_plus[i_a],
                                                        inv_v_ref_plus[i_a],
                                                        inv_v_buy_plus[i_a],
                                                        inv_v_rent_plus[i_w,i_shock,i_a,int(rent_choice[i_w,i_shock,i_a])]])
                                    choice = np.argmax(discrete) # 0 = stay, 1 = ref, 2 = buy, 3 = rent
                                    
                                    #assert discrete[choice] > 0, print(f'zero inverse value function best choice at {i_h, i_dp, i_Td, i_Tda, i_w, i_a, i_shock, y_plus}. The array is {discrete}')  
                                    if discrete[choice] <= 0:
                                        inv_v_def_plus,inv_marg_u_def_plus = force_default(y_plus,i_shock,t,sol,par) 
                                        v_plus = -1/inv_v_def_plus
                                        marg_u_plus = 1/inv_marg_u_def_plus
                                    elif choice == 0:
                                        v_plus = -1/inv_v_stay_plus[i_a]
                                        marg_u_plus = 1/inv_marg_u_stay_plus[i_a]
                                    elif choice == 1:
                                        v_plus = -1/inv_v_ref_plus[i_a]
                                        marg_u_plus = 1/inv_marg_u_ref_plus[i_a]
                                    elif choice == 2:
                                        v_plus = -1/inv_v_buy_plus[i_a]
                                        marg_u_plus = 1/inv_marg_u_buy_plus[i_a]
                                    elif choice == 3: 
                                        v_plus = -1/inv_v_rent_plus[i_w,i_shock,i_a,int(rent_choice[i_w,i_shock,i_a])]
                                        marg_u_plus = 1/inv_marg_u_rent_plus[i_w,i_shock,i_a,int(rent_choice[i_w,i_shock,i_a])]
                                    #assert marg_u_plus > 0, print(f'negative marginal utility of post decision. Index is ({i_h, i_dp, i_Td, i_Tda, i_w, i_a,}) choice = {choice}, discrete = {discrete}, {marg_u_plus}, {rent_choice}, {m_plus_gross_ref}, {m_plus_gross_buy}')
                                    v_bar[i_a] += weight*par.beta*v_plus
                                    q[i_h,i_dp,i_Td,i_Tda,i_w,i_a] += weight*par.beta*(1+par.r)*marg_u_plus # +1 to not overwrite rent solution
                                    
                            # iv. transform post decision value function
                            for i_a in range(par.Na):
                                inv_v_bar[i_h,i_dp,i_Td,i_Tda,i_w,i_a] = -1/v_bar[i_a] # +1 to not overwrite rent solution                           


####################
# 4. Stay problem  # 
####################
@njit(parallel=True)
def solve_stay(t,sol,par): 
    """ solve bellman equation for stayers using negm """
    # a. unpack input and endogenous arrays
    c_endo = sol.c_endo_stay[t]
    m_endo = sol.m_endo_stay[t]
    inv_v_bar = sol.inv_v_bar[t,1:] # 0'th entry in h-dimension is renter solution
    q_stay = sol.q[t,1:] # 0'th entry in h-dimension is renter solution
    n = par.n[t]

    # b. unpack output
    inv_v_stay = sol.inv_v_stay[t]
    inv_marg_u_stay = sol.inv_marg_u_stay[t]
    c_stay = sol.c_stay[t]
    
    # c. restrict loop over terminal periods
    Td_len = np.fmin(t+2,par.Td_shape) # fx 26 years old: terminal period can be 0, 55 og 56
    
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
                            c_endo[i_h,i_d,i_Td,i_Tda,i_w,i_a] = n*(q_stay[i_h,i_d,i_Td,i_Tda,i_w,i_a]/(1-par.nu))**(1/-par.rho) 
                            m_endo[i_h,i_d,i_Td,i_Tda,i_w,i_a] = a + c_endo[i_h,i_d,i_Td,i_Tda,i_w,i_a]
                            
                        # ii. interpolate from post decision space to beginning of period states
                        move = 0
                        rent = 0
                        negm_upperenvelope(
                            par.grid_a,
                            m_endo[i_h,i_d,i_Td,i_Tda,i_w,:],
                            c_endo[i_h,i_d,i_Td,i_Tda,i_w,:],
                            inv_v_bar[i_h,i_d,i_Td,i_Tda,i_w,:], 
                            par.grid_m,
                            c_stay[i_h,i_d,i_Td,i_Tda,i_w,:],
                            v_stay_vec,
                            h,move,rent,t,par)
 
                        # iii. optimal value func and marg u - (negative) inverse 
                        for i_m in range(par.Nm): 
                            inv_v_stay[i_h,i_d,i_Td,i_Tda,i_w,i_m] = -1/v_stay_vec[i_m]
                            inv_marg_u_stay[i_h,i_d,i_Td,i_Tda,i_w,i_m] = 1/utility.marg_func_nopar(c_stay[i_h,i_d,i_Td,i_Tda,i_w,i_m],
                                                                                                    par.nu,par.rho,n)


####################
# 5. Ref. problem  # 
####################
@njit
def obj_ref(d_prime,m_net,inv_v_stay_slice,grid_m,grid_d_prime,par):
    """ interpolate on stayers solution for refinancers """ 

    # value-of-choice
    return linear_interp.interp_2d(grid_d_prime,grid_m,inv_v_stay_slice,d_prime,m_net)

@njit(parallel=True)
def solve_ref_fast(t,sol,par):
    """solve bellman equation for refinancers using nvfi"""
    count = 0
    # a. unpack output
    inv_v_ref = sol.inv_v_ref_fast[t]
    inv_marg_u_ref = sol.inv_marg_u_ref_fast[t]
    c_ref = sol.c_ref_fast[t]
    d_prime_ref = sol.d_prime_ref_fast[t]
    Tda_prime_ref = sol.Tda_prime_ref_fast[t]

    # b. unpack input
    inv_v_stay = sol.inv_v_stay[t]
    c_stay = sol.c_stay[t]
    grid_Tda = np.arange(0,np.fmin(par.Tda_bar,par.T-t+1),1)
    
    nu = par.nu
    rho = par.rho
    omega_dti = par.omega_dti[t]
    n = par.n[t]

    # c. loop over outer states
    for i_w in prange(par.Nw):
        p = par.grid_w[i_w]
        y = trans.p_to_y_func(i_y=i_w,p=p,p_lag=p,t=t,par=par)
        for i_h in prange(par.Nh):
            h = par.grid_h[i_h]

            # i. loop over gross resources
            for i_x in range(par.Nx):                        
                m_gross = par.grid_x[i_x]
                
                # o. enforce financial regulation
                Td_new = mt.Td_func(t,par) 
                i_Td_new = int(Td_new - par.Td_bar)
                    
                d_prime_high = np.fmin(par.omega_ltv*par.q*h,omega_dti*y) 
                grid_d_prime = np.linspace(0,d_prime_high,par.Nd)

                # oo. loop over mortage plan choices 
                inv_v_ref_best = 0
                for i_dp in range(par.Nd): 
                    d_prime_now = grid_d_prime[i_dp]
                    for Tda in range(max(grid_Tda)):

                        ## refinancer's net cash on hand equation
                        loan = int(d_prime_now > 0)
                        m_net = m_gross-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime_now  

                        ## enforce non-negativity constraint
                        if m_net <= 0:
                            count += 1
                            d_prime_ref[i_h,i_w,i_x] = 0
                            Tda_prime_ref[i_h,i_w,i_x] = 0
                            c_ref[i_h,i_w,i_x] = 0
                            inv_v_ref[i_h,i_w,i_x] = 0
                            inv_marg_u_ref[i_h,i_w,i_x] = 0        
                            continue

                        ## evaluate choice
                        inv_v_ref_new = obj_ref(d_prime_now,m_net,
                                                inv_v_stay[i_h,:,i_Td_new,Tda,i_w,:],
                                                par.grid_m,grid_d_prime,par)

                        ## update optimal value and choices?
                        if inv_v_ref_new > inv_v_ref_best:
                            inv_v_ref_best = inv_v_ref_new
                            d_prime_best = d_prime_now
                            Tda_best = Tda
                            i_dp_best = i_dp
                
                # ooo. save optimal value and choices
                d_prime_ref[i_h,i_w,i_x] = d_prime_best
                Tda_prime_ref[i_h,i_w,i_x] = Tda_best
                
                ## refinancer's net cash on hand equation
                loan = int(d_prime_best > 0)
                m_net = m_gross-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime_best  
               
                ## enforce non-negativity constraint
                if m_net <= 0:
                    count += 1
                    d_prime_ref[i_h,i_w,i_x] = 0
                    Tda_prime_ref[i_h,i_w,i_x] = 0
                    c_ref[i_h,i_w,i_x] = 0
                    inv_v_ref[i_h,i_w,i_x] = 0
                    inv_marg_u_ref[i_h,i_w,i_x] = 0        
                    continue
                
                # oooo. now interpolate on stayer consumption and value function
                c_ref[i_h,i_w,i_x] = linear_interp.interp_1d(par.grid_m,c_stay[i_h,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
                inv_v_ref[i_h,i_w,i_x] = linear_interp.interp_1d(par.grid_m,inv_v_stay[i_h,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
                inv_marg_u_ref[i_h,i_w,i_x] = 1/utility.marg_func_nopar(c_ref[i_h,i_w,i_x],nu,rho,n)


####################
# 6. Buy problem   # 
####################
@njit
def obj_buy_fast(d_prime,m_net,inv_v_stay_slice,grid_m,grid_d_prime,par):
    """ interpolate on stayer solution for buyers """

    # a. subtract pecuniary cost of moving
    v_buy = -1/np.fmax(linear_interp.interp_2d(grid_d_prime,grid_m,inv_v_stay_slice,d_prime,m_net),par.tol)-par.kappa
    
    # b. compute value-of-choice
    return -1/v_buy

@njit(parallel=True)
def solve_buy_fast(t,sol,par):
    """ solve bellman equation for buyers using nvfi"""
    count = 0
    # a. unpack output
    inv_v_buy = sol.inv_v_buy_fast[t]
    inv_marg_u_buy = sol.inv_marg_u_buy_fast[t]
    c_buy = sol.c_buy_fast[t]
    d_prime_buy = sol.d_prime_buy_fast[t]
    Tda_prime_buy = sol.Tda_prime_buy_fast[t]
    h_buy = sol.h_buy_fast[t]

    # b. unpack input
    inv_v_stay = sol.inv_v_stay[t]
    c_stay = sol.c_stay[t]
    grid_Tda = np.arange(0,np.fmin(par.Tda_bar,par.T-t+1),1)
    
    nu = par.nu
    rho = par.rho
    omega_dti = par.omega_dti[t]
    n = par.n[t]
    
    # c. loop over outer states
    for i_w in prange(par.Nw):
        p = par.grid_w[i_w]
        y = trans.p_to_y_func(i_y=i_w,p=p,p_lag=p,t=t,par=par)
        for i_x in prange(par.Nx):               
            m_gross = par.grid_x[i_x]

            #i. initialise best choice at zero
            inv_v_buy_best = 0

            # ii. loop over house purchase
            for i_hb in range(par.Nh):
                h_buy_now = par.grid_h[i_hb]

                # o. enforce financial regulation
                Td_new = mt.Td_func(t,par)            # terminal mortgage period
                i_Td_new = int(Td_new - par.Td_bar)   # index of terminal period

                d_prime_high = np.fmin(par.omega_ltv*par.q*h_buy_now,omega_dti*y)   
                grid_d_prime = np.linspace(0,d_prime_high,par.Nd)                       

                # oo. loop over mortage plan choices 
                for i_dp in range(len(grid_d_prime)): 
                    d_prime_now = grid_d_prime[i_dp]
                    for Tda in range(max(grid_Tda)):
                        
                        ## buyer's net cash on hand equation
                        loan = int(d_prime_now > 0)
                        m_net = m_gross-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime_now-(1+par.C_buy)*par.q*h_buy_now

                        ## enforce non-negativity constraint
                        if m_net <= 0:
                            count += 1
                            d_prime_buy[i_w,i_x]= 0
                            Tda_prime_buy[i_w,i_x] = 0
                            h_buy[i_w,i_x] = 0
                            c_buy[i_w,i_x] = 0
                            inv_v_buy[i_w,i_x] = 0
                            inv_marg_u_buy[i_w,i_x] = 0        
                            continue
                        
                        ## evaluate choice
                        inv_v_buy_new = obj_buy_fast(d_prime_now,m_net,
                                                inv_v_stay[i_hb,:,i_Td_new,Tda,i_w,:],
                                                par.grid_m,grid_d_prime,par)

                        ## update optimal value and choices?
                        if inv_v_buy_new > inv_v_buy_best:
                            inv_v_buy_best = inv_v_buy_new
                            d_prime_best = grid_d_prime[i_dp]
                            Tda_best = Tda
                            h_buy_best = h_buy_now
                            i_hb_best = i_hb
                            i_dp_best = i_dp

            # ooo. save optimal value and choices
            d_prime_buy[i_w,i_x] = d_prime_best
            Tda_prime_buy[i_w,i_x] = Tda_best
            h_buy[i_w,i_x] = h_buy_best

            ## buyer's net cash on hand equation
            loan = int(d_prime_best > 0)
            m_net = m_gross-loan*par.Cf_ref+(1-par.Cp_ref)*d_prime_best-(1+par.C_buy)*par.q*h_buy_best
            
            ## enforce non-negativity constraint
            if m_net <= 0:
                count += 1
                d_prime_buy[i_w,i_x]= 0
                Tda_prime_buy[i_w,i_x] = 0
                h_buy[i_w,i_x] = 0
                c_buy[i_w,i_x] = 0
                inv_v_buy[i_w,i_x] = 0
                inv_marg_u_buy[i_w,i_x] = 0        
                continue

            # oooo. interpolate on stayer consumption and value function
            c_buy[i_w,i_x]= linear_interp.interp_1d(par.grid_m,c_stay[i_hb_best,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
            inv_v_buy[i_w,i_x] = linear_interp.interp_1d(par.grid_m,inv_v_stay[i_hb_best,i_dp_best,i_Td_new,Tda_best,i_w,:],m_net)
            inv_marg_u_buy[i_w,i_x] = 1/utility.marg_func_nopar(c_buy[i_w,i_x],nu,rho,n)


####################
# 7. Rent problem   # 
####################
@njit(parallel=True)
def solve_rent(t,sol,par):
    """ solve bellman equation for renters using negm """
    # unpack input and endogenous arrays
    c_endo = sol.c_endo_rent[t]
    m_endo = sol.m_endo_rent[t]
    inv_v_bar = sol.inv_v_bar[t,0,0,0,0]
    q_rent = sol.q[t,0,0,0,0]

    # unpack solution arrays
    inv_v_rent = sol.inv_v_rent[t]
    inv_marg_u_rent = sol.inv_marg_u_rent[t]
    #htilde = sol.htilde[t]
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
                c_endo[i_ht,i_w,i_a] = par.n[t]*(q_rent[i_w,i_a]/(1-par.nu))**(1/-par.rho)
                m_endo[i_ht,i_w,i_a] = a + c_endo[i_ht,i_w,i_a] #+ par.q_r*htilde

            # ii. interpolate from post decision space to beginning of period states
            move = 0    # no costs of moving when renting?
            rent = 1
            negm_upperenvelope(
                par.grid_a,
                m_endo[i_ht,i_w,:],
                c_endo[i_ht,i_w,:],
                inv_v_bar[i_w,:],
                par.grid_m,
                c_rent[i_ht,i_w,:],
                v_rent_vec,
                htilde,move,rent,t,par)
            sol.htilde[t,i_ht,i_w,:] = htilde 

            # iii. optimal value func and marg u - (negative) inverse 
            for i_m in range(par.Nm): 
                inv_v_rent[i_ht,i_w,i_m] = -1/v_rent_vec[i_m]
                inv_marg_u_rent[i_ht,i_w,i_m] = 1/utility.marg_func_nopar(c_rent[i_ht,i_w,i_m],
                                                                          par.nu,par.rho,par.n[t])


####################
# 8.   Default     # 
####################
@njit
def force_default(y_plus,i_shock,t,sol,par):
    # after tax income given foreclosure
    y_tilde_plus = mt.income_aftertax(y_plus,0,0,par)
    
    # initialise at zero
    inv_v_default = 0

    # find best rent option
    for i_ht in range(par.Nhtilde):
        # cash on hand net of rental cost
        m_net_rent = y_tilde_plus - par.q_r*par.grid_htilde[i_ht]
        
        # interp and add utility cost of default
        inv_v_rent = np.fmax(
            linear_interp.interp_1d(par.grid_m,
            sol.inv_v_rent[t+1,i_ht,i_shock,:],
            m_net_rent),par.tol)
        v_def_temp = -1/inv_v_rent - par.zeta

        # convert back to inverse and update 
        inv_v_def_temp = -1/v_def_temp
        if inv_v_def_temp > inv_v_default:
            inv_v_default = inv_v_def_temp
            inv_marg_u_def = np.fmax(
                                linear_interp.interp_1d(
                                    par.grid_m,
                                    sol.inv_marg_u_rent[t+1,i_ht,i_shock],
                                    m_net_rent),
                                par.tol)
    return inv_v_default, inv_marg_u_def              


#@njit(parallel=True)
#def postdecision_compute_v_bar_q(t,sol,par):
#    """ compute the post-decision functions w and/or q """
#    # unpack solution arrays
#    inv_v_bar = sol.inv_v_bar[t]
#    q = sol.q[t]
#
#    # b. restrict loop over terminal periods
#    Td_len = np.fmin(t+2,par.Td_shape) # fx 26 years old: terminal period can be 0, 55 og 56
#
#    # c. counter
#    count = 0 
#    # loop over outermost post-decision state
#    for i_w in prange(par.Nw): 
#
#        # a. allocate temporary containers
#        m_plus_stay = np.zeros(par.Na)
#        m_plus_gross_ref = np.zeros(par.Na)
#        m_plus_gross_buy = np.zeros(par.Na) 
#        m_plus_rent = np.zeros(par.Na) # container, same lenght as grid_a
#
#        v_bar = np.zeros(par.Na)
#
#        inv_v_stay_plus = np.zeros(par.Na)
#        inv_v_ref_plus = np.zeros(par.Na)
#        inv_v_buy_plus = np.zeros(par.Na)
#        inv_v_rent_plus = np.zeros((par.Na,par.Nhtilde))
#
#        inv_marg_u_stay_plus = np.zeros(par.Na)
#        inv_marg_u_ref_plus = np.zeros(par.Na)
#        inv_marg_u_buy_plus = np.zeros(par.Na)
#        inv_marg_u_rent_plus = np.zeros((par.Na,par.Nhtilde))
#
#        # b. loop over other outer post-decision states
#        for i_h in prange(par.Nh+1):
#            # housing stock (own or rent)
#            if i_h == 0: 
#                h = 0
#            else: 
#                h = par.grid_h[i_h-1]
#
#            for i_Td in range(Td_len):
#                    for i_Tda in range(np.fmin(par.Tda_bar,par.T-t+1)):
#                        # mortgage plan and scale grid
#                        Tda = i_Tda
#                        Td = mt.Td_func(t,par)
#
#                        d_prime_high = par.q*h
#                        grid_d_prime = np.linspace(0,d_prime_high,par.Nd)
#                        
#                        for i_dp in range(par.Nd):
#                            # i. permanent income
#                            #w = par.grid_w[i_w]
#
#                            # ii. next period mortgage balance
#                            d_plus = trans.d_plus_func(grid_d_prime[i_dp],t,Td,Tda,par)
#                        
#                            # iii. initialise at zero
#                            for i_a in range(par.Na):
#                                v_bar[i_a] = 0.0
#                                q[i_h,i_dp,i_Td,i_Tda,i_w,i_a] = 0.0
#
#                            # iv. loop over shocks and then end-of-period assets
#                            for i_shock in range(par.Nw):                
#                                # o. next-period income
#                                p = par.grid_w[i_w] 
#                                p_plus = par.grid_w[i_shock]
#                                y_plus = trans.p_to_y_func(i_y=i_w,p=p_plus,p_lag=p,t=t+1,par=par)
#                                p_plus_weight = par.w_trans[i_w,i_shock]
#
#                                # oo. compute weight 
#                                weight = p_plus_weight
#
#                                # ooo. evaluate next period living situation
#                                ## rent next period
#                                    # loop through rental sizes
#                                for i_ht in range(par.Nhtilde):  
#                                    for i_a in range(par.Na):
#
#                                        # find gross cash-on-hand
#                                        m_plus_rent[i_a] = trans.m_plus_func(par.grid_a[i_a],y_plus,grid_d_prime[i_dp],Td,Tda,par,t) - par.q_r*par.grid_htilde[i_ht]
#                                    
#                                    # interpolate on inverse funcs given rental choice
#                                    linear_interp.interp_1d_vec(par.grid_m,sol.inv_v_rent[t+1,i_ht,i_shock],
#                                                                m_plus_rent,inv_v_rent_plus[:,i_ht])
#                                    linear_interp.interp_1d_vec(par.grid_m,sol.inv_marg_u_rent[t+1,i_ht,i_shock],
#                                                                m_plus_rent,inv_marg_u_rent_plus[:,i_ht])
#        
#                                ## own next period
#                                # prepare interpolator
#                                prep_stay = linear_interp.interp_2d_prep(grid_d_prime,d_plus,par.Na)
#                                
#                                # cash-on-hand
#                                for i_a in range(par.Na):
#                                    m_plus_stay[i_a] = trans.m_plus_func(par.grid_a[i_a],y_plus,d_plus,Td,Tda,par,t) - par.delta*par.q*h - mt.property_tax(par.q,h,par)
#                                    m_plus_gross_ref[i_a] = m_plus_stay[i_a] - grid_d_prime[i_dp]
#                                    m_plus_gross_buy[i_a] = m_plus_stay[i_a] - grid_d_prime[i_dp] + (1-par.C_sell)*par.q*h
#
#                                # condition on owning this period (otherwise stay or ref are not in the choice set)
#                                if h!=0:
#                                    # interpolate to get inverse funcs for stayers
#                                    linear_interp.interp_2d_only_last_vec_mon(prep_stay,grid_d_prime,par.grid_m,
#                                                                              sol.inv_v_stay[t+1,i_h-1,:,i_Td,i_Tda,i_shock],
#                                                                              d_plus,m_plus_stay,inv_v_stay_plus)
#                                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_stay,grid_d_prime,par.grid_m,
#                                                                                  sol.inv_marg_u_stay[t+1,i_h-1,:,i_Td,i_Tda,i_shock],
#                                                                                  d_plus,m_plus_stay,inv_marg_u_stay_plus)
#
#                                    # interpolate to get inverse funcs for refinancers
#                                    linear_interp.interp_1d_vec(par.grid_x,sol.inv_v_ref_fast[t+1,i_h-1,i_shock],
#                                                                m_plus_gross_ref,inv_v_ref_plus)
#                                    linear_interp.interp_1d_vec(par.grid_x,sol.inv_marg_u_ref_fast[t+1,i_h-1,i_shock],
#                                                                m_plus_gross_ref,inv_marg_u_ref_plus)
#                            
#                                # interpolate on inverse funcs for buyers
#                                linear_interp.interp_1d_vec(par.grid_x,sol.inv_v_buy_fast[t+1,i_shock],m_plus_gross_buy,inv_v_buy_plus)
#                                linear_interp.interp_1d_vec(par.grid_x,sol.inv_marg_u_buy_fast[t+1,i_shock],m_plus_gross_buy,inv_marg_u_buy_plus)
#                                
#                                # oooo. max and accumulate
#                                    ## find best rental choice given states
#                                rent_choice = np.nan*np.zeros(par.Na)                                                            
#                                for i_a in range(par.Na):
#                                    discrete_rent = np.array([inv_v_rent_plus[i_a,0],
#                                                             inv_v_rent_plus[i_a,1],
#                                                             inv_v_rent_plus[i_a,2]])
#                                    rent_choice[i_a] = np.argmax(discrete_rent)
#                                    
#                                    ## find best discrete choice given states
#                                for i_a in range(par.Na):  
#                                    if h == 0:
#                                        discrete = np.array([inv_v_buy_plus[i_a],
#                                                            inv_v_rent_plus[i_a,int(rent_choice[i_a])]])
#                                        choice = np.argmax(discrete) # 0 = buy, 1 = rent
#                                        #assert discrete[choice] > 0, print(f'zero inverse value function best choice at {i_h, i_dp, i_Td, i_Tda, i_w, i_a, y_plus}. The array is {discrete}')  
#                                        if choice == 0:     
#                                            v_plus = -1/inv_v_buy_plus[i_a]
#                                            marg_u_plus = 1/inv_marg_u_buy_plus[i_a]
#                                        else: 
#                                            v_plus = -1/inv_v_rent_plus[i_a,int(rent_choice[i_a])]
#                                            marg_u_plus = 1/inv_marg_u_rent_plus[i_a,int(rent_choice[i_a])]
#                                    else:                                   
#                                        discrete = np.array([inv_v_stay_plus[i_a],
#                                                            inv_v_ref_plus[i_a],
#                                                            inv_v_buy_plus[i_a],
#                                                            inv_v_rent_plus[i_a,int(rent_choice[i_a])]])
#                                        choice = np.argmax(discrete) # 0 = stay, 1 = ref, 2 = buy, 3 = rent
#                                        
#                                        #assert discrete[choice] > 0, print(f'zero inverse value function best choice at {i_h, i_dp, i_Td, i_Tda, i_w, i_a, y_plus}. The array is {discrete}')  
#                                        if discrete[choice] <= 0:
#                                            inv_v_def_plus,inv_marg_u_def_plus = force_default(y_plus,i_shock,t,sol,par) 
#                                            v_plus = -1/inv_v_def_plus
#                                            marg_u_plus = 1/inv_marg_u_def_plus
#                                        elif choice == 0:
#                                            v_plus = -1/inv_v_stay_plus[i_a]
#                                            marg_u_plus = 1/inv_marg_u_stay_plus[i_a]
#                                        elif choice == 1:
#                                            v_plus = -1/inv_v_ref_plus[i_a]
#                                            marg_u_plus = 1/inv_marg_u_ref_plus[i_a]
#                                        elif choice == 2:
#                                            v_plus = -1/inv_v_buy_plus[i_a]
#                                            marg_u_plus = 1/inv_marg_u_buy_plus[i_a]
#                                        elif choice == 3: 
#                                            v_plus = -1/inv_v_rent_plus[i_a,int(rent_choice[i_a])]
#                                            marg_u_plus = 1/inv_marg_u_rent_plus[i_a,int(rent_choice[i_a])]
#                                    #assert marg_u_plus > 0, print(f'negative marginal utility of post decision. Index is ({i_h, i_dp, i_Td, i_Tda, i_w, i_a,}) choice = {choice}, discrete = {discrete}, {marg_u_plus}, {rent_choice}, {m_plus_gross_ref}')
#                                    v_bar[i_a] += weight*par.beta*v_plus
#                                    q[i_h,i_dp,i_Td,i_Tda,i_w,i_a] += weight*par.beta*(1+par.r)*marg_u_plus
#                                    
#                            # v. transform post decision value function
#                            for i_a in range(par.Na):
#                                inv_v_bar[i_h,i_dp,i_Td,i_Tda,i_w,i_a] = -1/v_bar[i_a]
                                                             