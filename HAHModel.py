""" HAHModel

Solves a Heterogeneous Agent Housing Market (HAH) model building on the EconModel and 
consav packages as well as the NVFI and NEGM algorithms proposed by Jeppe Druedahl.

Developer: Christian Brauer Johanssen
Create date: 2022-10-05
Version: incomplete

"""
##########################
#       1. Imports       #
##########################

# a. standard packages
import numpy as np
import time

# b. NumEconCph packages
from EconModel import EconModelClass, jit
from consav import jit
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, find_ergodic, choice
from consav.quadrature import log_normal_gauss_hermite

# c. local modules
import steady_state
import HHproblems as hhp
import trans
import utility
import simulate
import steady_state 
import figs

##########################
#  2. Build model class  #
##########################
class HAHModelClass(EconModelClass):    

    #############
    #   setup   #
    #############

    def settings(self):
        """ fundamental settings """

        # a. namespaces 
        self.namespaces = ['par','sim','sol'] 
        
        # b. other attributes
        self.other_attrs = []

        # c. savefolder
        self.savefolder = 'saved'

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # a. preferences
        par.beta = 0.965                                # subjective discount factor
        par.rho = 2.0                                   # CRRA coefficient
        par.alpha = 1.1                                 # housing curvature
        par.nu = 0.26                                   # weight on housing
        par.phi = 0.85                                  # scaling of housing services from rent
        par.kappa = 0.34                                # disutility of moving
        par.thetab = 100                                # strength of bequest motive 
        par.K = 7.7                                     # extent of bequest as luxury
        par.zeta = 10.0                                  # disutility of default
        
        # b. demographics and life cycle profile
        par.median_income = 670_000                     # for normalisation
        par.Tmin = 25                                   # age when entering the model
        par.T = 80 - par.Tmin                           # age of death
        par.Tr = 65 - par.Tmin                          # retirement age
        par.chi = np.ones(par.T)                        # placeholder for deterministic income process
        par.n = np.ones(par.T)                          # placeholder for equivalence scale 

        # c. income process
        par.rho_p = 0.96                                # AR(1) parameter
        par.sigma_psi = 0.20                            # std. of persistent shock
        par.sigma_psi_ini = 0.52                        # std. of initial income shock
        par.Np = 7                                      # number of permanent income states

        par.sigma_xi = 0.0                              # std. dev. of transitory shock
        par.Nxi = 1                                     # quadrature nodes for transitory shock

        par.pi = 0.025                                  # unemployment probability
        par.b = 0.2                                     # unemployment benefits

        par.pension = 1.0                               # scaling of pension income 

        # d. interest rates and financial regulation
        par.r = 0.02                                    # return on liquid assets
        par.r_m = par.r+0.0056                          # amortising mortgage interest rate     
        par.r_da = par.r_m+0.0008                       # deferred amortisation mortgage rate
        par.omega_ltv = 0.9                             # loan-to-value ratio  
        par.omega_dti = 4.5*np.ones(par.T)              # debt-to-income ratio working life
        par.omega_dti[par.Tr:] = 2.5                    # debt-to-income ratio retired
        par.Cp_ref = 0.017                              # proportional refinancing cost JEJA
        par.Cf_ref = 8250/par.median_income             # fixed refinancing cost
        par.Td_bar = 30                                 # maximum regulatory mortgage term length
        par.Td_shape = 27                               # sample space for mortgage terminal periods
        par.Tda_bar = 11                                # maximum terms with deferred amortisation +1
        
        # e. housing and rental markets
        par.delta = 0.015                               # proportional maintenance cost
        par.gamma = 0.014                               # per rental unit operating cost
        par.C_buy = 0.02                                # proportional house sale cost
        par.C_sell = 0.04                               # proportional house purchase cost
        #par.sigma_epsilon = 0.04                       # std. dev. of housing shock
        #par.Nepsilon = 5                               # quadrature nodes for housing shock

        # f. taxation
        par.tau_y0 = 0.32                               # income tax function parameter 1    
        par.tau_y1 = 0.19                               # income tax function parameter 2
        par.tau_h0 = 0.0092                             # bottom-bracket property tax rate
        par.tau_h1 = 0.03                               # top-bracket property tax rate
        par.tau_r0 = 0.336                              # low bracket tax value of interest rate expenses
        par.tau_r1 = 0.256                              # high bracket tax value of interet rate expenses
        par.qh_bar = 3_040_000/par.median_income        # top-bracket property tax threshold
        par.rd_bar = 75_000/par.median_income           # high tax value of interest deduction threshold

        # g. price guesses for stationary equilibrium
        par.q = 1.0                                                   # house price guess
        par.q_r = par.gamma+(par.r+par.delta)/(1+par.r)*par.q         # implied rental price

        # h. grids
        par.Nh = 6                                      # points in owner occupied housing grid
        par.h_min = 1.42                                # minimum owner occupied houze size 
        par.h_max = 5.91                                # maximum owner occupied house size
        par.Nhtilde = 3                                 # points in rental house size grid
        par.htilde_min = 1.07                           # minimum rental house size
        par.htilde_max = 1.89                           # maximum rental house size
        par.Nd = 20                                     # points in mortgage balance grid
        par.Nm = 30                                     # points in cash on hand grid
        par.Nx = 45                                     # points in gross resources grid
        par.Na = 30                                     # points in assets grid
        par.m_max = 35.0                                # maximum cash-on-hand and gross resources
        par.x_min = -7.0                                # minimum gross resources (before refinancing)
        par.a_max = par.m_max                           # maximum assets

        # i. simulation
        par.mu_a0 = -1.99                               # mean initial assets
        par.sigma_a0 = 1.25                             # standard dev. of initial assets
        
        par.simN = 100_000                              # number of simulated agents
        par.sim_seed = 1995                             # seed for random number generator
        par.euler_cutoff = 0.02                         # euler error cutoff

        # i. misc
        par.t = 0                                       # initial time
        par.tol = 1e-12                                 # solution precision tolerance
        par.do_print = True                            # whether to print solution progress
        par.do_print_period = False                     # whether to print solution progress every period
        par.max_iter_simulate = 500                     # max iterations when finding stable bequest
        par.include_unemp = False                       # unemployment extension?

    def allocate(self):
        """ allocate model """

        par = self.par

        self.create_grids()                             # create grids
        self.solve_prep()                               # allocate solution arrays
        self.simulate_prep()                            # allocate simulation arrays

    def create_grids(self):
        """ construct grids for states and shocks """
        
        par = self.par

        # a. beginning of period states (income is approxed by Np-state Markov Proces, mortgage is dynamic)
        par.grid_h = np.array([par.h_min, 1.89, 2.51, 3.34, 4.44, par.h_max],dtype='double')
        par.grid_m = equilogspace(0,par.m_max,par.Nm)
        par.grid_x = np.append(np.linspace(par.x_min,-.5,par.Nx-par.Nm),par.grid_m)
        par.grid_htilde = np.array([par.htilde_min, 1.42, par.htilde_max],dtype='double')
        
        # b. post-decision assets
        par.grid_a = equilogspace(0,par.a_max,par.Na)
        
        # c. shocks and income process
            # i. persistent shock/permanent income states
        _out = log_rouwenhorst(par.rho_p,par.sigma_psi,par.Np)
        par.p_grid,par.p_trans,par.p_ergodic,par.p_trans_cumsum,par.p_ergodic_cumsum = _out
        _,_,_,_,par.p_ini_ergodic_cumsum = log_rouwenhorst(par.rho_p,par.sigma_psi_ini,par.Np) 
        
            # ii. transitory income shock
        if par.sigma_xi > 0 and par.Nxi > 1:
            par.xi_grid,par.xi_weights = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)
            par.xi_trans = np.broadcast_to(par.xi_weights,(par.Nxi,par.Nxi))
        else:
            par.xi_grid = np.ones(1)
            par.xi_weights = np.ones(1)
            par.xi_trans = np.ones((1,1))

            # iii. combine and add unemployment
        if par.include_unemp: 
            par.Nw = par.Nxi*par.Np+1 # +1 to add unemployment outcome
            grid_w_emp = (np.repeat(par.xi_grid,par.Np)*np.tile(par.p_grid,par.Nxi)-par.pi*par.b)/(1-par.pi)
            par.grid_w = np.sort(np.append(grid_w_emp,par.b))
            
            par.w_trans = np.zeros((len(par.grid_w),len(par.grid_w)))
            w_trans_inner = np.kron(par.xi_trans,par.p_trans)*(1-par.pi)
            par.w_trans[1:len(par.grid_w),1:len(par.grid_w)] = w_trans_inner
            par.w_trans[0,:] = par.pi # fill top row with probability par.pi
            par.w_trans[:,0] = par.pi # fill leftmost col with probability par.pi
        else:
            par.Nw = par.Nxi*par.Np 
            par.grid_w = np.repeat(par.xi_grid,par.Np)*np.tile(par.p_grid,par.Nxi)
            par.w_trans = np.kron(par.xi_trans,par.p_trans)
        
        par.w_trans_cumsum = np.cumsum(par.w_trans,axis=1)
        par.w_ergodic = find_ergodic(par.w_trans)
        par.w_ergodic_cumsum = np.cumsum(par.w_ergodic)
        #par.w_trans_T = par.w_trans.T

        # d. timing
        par.time_vbarq = np.zeros(par.T)
        par.time_stay = np.zeros(par.T)
        par.time_ref = np.zeros(par.T)
        par.time_buy = np.zeros(par.T)
        par.time_rent = np.zeros(par.T)
        par.time_full = np.zeros(par.T)

    #############
    #   solve   #
    #############

    def precompile_numba(self):
        """ solve the model with very coarse grids and simulate with very few persons
            --> quicker for numba to analyse the code 
        """

        par = self.par

        tic = time.time()

        # a. define points in coarse grids
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_period'] = True
        fastpar['T'] = 4
        fastpar['Td_shape'] = 3
        fastpar['Td_bar'] = 2
        fastpar['Tda_bar'] = 2
        fastpar['Np'] = 3
        fastpar['Nxi'] = 1
        fastpar['Nh'] = 6
        fastpar['Nhtilde'] = 3 
        fastpar['Nd'] = 3
        fastpar['Nm'] = 3
        fastpar['Nx'] = 3
        fastpar['Na'] = 3
        fastpar['simN'] = 2
        fastpar['chi'] = par.chi[0:4]
        fastpar['n'] = par.n[0:4]

        # b. apply
        for key,val in fastpar.items():
            prev = getattr(par,key)
            setattr(par,key,val)
            fastpar[key] = prev

        self.allocate()

        # c. solve
        self.solve(do_assert=False)

        # d. simulate
        self.simulate()

        # e. reiterate
        for key,val in fastpar.items():
            setattr(par,key,val)
        
        self.allocate()

        toc = time.time()
        if par.do_print:
            print(f'numba precompiled in {toc-tic:.1f} secs')

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        # a. shapes
        own_shape = (par.T,par.Nh,par.Nd,par.Td_shape,par.Tda_bar,par.Nw,par.Nm)
        #buy_shape = (par.T,par.Nh+1,par.Nd,par.Td_shape,par.Tda_bar,par.Nw,par.Nm)
        rent_shape = (par.T,par.Nhtilde,par.Nw,par.Nm)
        post_shape = (par.T,par.Nh+1,par.Nd,par.Td_shape,par.Tda_bar,par.Nw,par.Na)

        buy_shape_fast = (par.T,par.Nw,par.Nx)
        ref_shape_fast = (par.T,par.Nh,par.Nw,par.Nx)  

        # b. stay        
        sol.c_stay = np.zeros(own_shape)
        sol.inv_v_stay = np.zeros(own_shape)
        sol.inv_marg_u_stay = np.zeros(own_shape)

        # c. refinance
        sol.c_ref_fast = np.zeros(ref_shape_fast)
        sol.d_prime_ref_fast = np.zeros(ref_shape_fast)
        sol.Tda_prime_ref_fast = np.zeros(ref_shape_fast)   
        sol.inv_v_ref_fast = np.zeros(ref_shape_fast)
        sol.inv_marg_u_ref_fast = np.zeros(ref_shape_fast)

        # d. buy
        sol.c_buy_fast = np.zeros(buy_shape_fast)
        sol.h_buy_fast = np.zeros(buy_shape_fast)
        sol.d_prime_buy_fast = np.zeros(buy_shape_fast)
        sol.Tda_prime_buy_fast = np.zeros(buy_shape_fast)
        sol.inv_v_buy_fast = np.zeros(buy_shape_fast)
        sol.inv_marg_u_buy_fast = np.zeros(buy_shape_fast)

        # e. rent
        sol.c_rent = np.zeros(rent_shape)
        sol.htilde = np.zeros(rent_shape)
        sol.inv_v_rent = np.zeros(rent_shape)
        sol.inv_marg_u_rent = np.zeros(rent_shape)
            
        # f. post decision (overarching) and endogenous arrays 
        sol.inv_v_bar = np.nan*np.zeros(post_shape)
        sol.q = np.nan*np.zeros(post_shape)

        sol.inv_v_rent_plus = np.zeros((par.T,par.Nw,par.Nw,par.Na,par.Nhtilde))
        sol.inv_marg_u_rent_plus = np.zeros((par.T,par.Nw,par.Nw,par.Na,par.Nhtilde))
        sol.rent_choice = np.zeros((par.T,par.Nw,par.Nw,par.Na)) 

        sol.c_endo_stay = np.nan*np.zeros(own_shape)
        sol.m_endo_stay = np.nan*np.zeros(own_shape)
        sol.c_endo_rent = np.nan*np.zeros((par.T,par.Nhtilde,par.Nw,par.Na)) 
        sol.m_endo_rent = np.nan*np.zeros((par.T,par.Nhtilde,par.Nw,par.Na))

    def solve(self,do_assert=True):
        """ solve the household problem using NEGM
        
        Args:

            do_assert (bool,optional): make assertions on the solution
        
        """
        total_solve_time = 0
        
        # backwards induction
        for t in reversed(range(self.par.T)):
            
            self.par.t = t
            tic = time.time()

            with jit(self) as model:

                par = model.par
                sol = model.sol
                
                Td_len = np.fmin(t+2,par.Td_shape)
                Tda_len = np.fmin(par.Tda_bar,par.T-t+1)
                # a. last period or post decision
                if t == par.T-1:
                    tic_w = time.time()
                    hhp.last_period_v_bar_q(t,sol,par)
                    toc_w = time.time()
                    par.time_vbarq[t] = toc_w-tic_w
                    if par.do_print:
                        print(f' last period bequest computed in {toc_w-tic_w:.1f} secs')

                    if do_assert: 
                        assert np.all((sol.inv_v_bar[t,:,:,0:Td_len,0:Tda_len,:,:] > 0) & 
                                      (np.isnan(sol.inv_v_bar[t,:,:,0:Td_len,0:Tda_len,:,:]) == False)), t
                        assert np.all((sol.q[t,:,:,0:Td_len,0:Tda_len,:,:] > 0) & 
                                      (np.isnan(sol.q[t,:,:,0:Td_len,0:Tda_len,:,:]) == False)), t

                else: 
                    tic_w = time.time()
                    #hhp.postdecision_compute_v_bar_q(t,sol,par)
                    hhp.postdecision_compute_v_bar_q_rent(t,sol,par)
                    hhp.postdecision_compute_v_bar_q_own(t,sol,par)                
                    toc_w = time.time()
                    par.time_vbarq[t] = toc_w-tic_w
                    if par.do_print:
                        print(f' v_bar and q computed in {toc_w-tic_w:.1f} secs')
                    
                    if do_assert: 
                        assert np.all((sol.inv_v_bar[t,:,:,0:Td_len,0:Tda_len,:,:] > 0) & 
                                      (np.isnan(sol.inv_v_bar[t,:,:,0:Td_len,0:Tda_len,:,:]) == False)), t
                        assert np.all((sol.q[t,:,:,0:Td_len,0:Tda_len,:,:] > 0) & 
                                      (np.isnan(sol.q[t,:,:,0:Td_len,0:Tda_len,:,:]) == False)), t
                        
                # b. solve and time stayer problem
                tic_stay = time.time()
                hhp.solve_stay(t,sol,par)
                toc_stay = time.time()
                par.time_stay[t] = toc_stay-tic_stay
                
                if par.do_print:
                    print(f' solved stayer problem in {toc_stay-tic_stay:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_stay[t,:,:,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.c_stay[t,:,:,0:Td_len,0:Tda_len,:,:]) == False)), t
                    assert np.all((sol.inv_v_stay[t,:,:,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.inv_v_stay[t,:,:,0:Td_len,0:Tda_len,:,:]) == False)), t
                
                # c. solve and time refinance problem
                tic_ref = time.time()
                hhp.solve_ref_fast(t,sol,par)                  
                toc_ref = time.time()
                par.time_ref[t] = toc_ref-tic_ref
                
                if par.do_print:
                    print(f' solved refinance problem in {toc_ref-tic_ref:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_ref_fast[t] >= 0) & (np.isnan(sol.c_ref_fast[t]) == False)), t
                    assert np.all((sol.d_prime_ref_fast[t] >= 0) & (np.isnan(sol.d_prime_ref_fast[t]) == False)), t
                    assert np.all((sol.inv_v_ref_fast[t] >= 0) & (np.isnan(sol.inv_v_ref_fast[t]) == False)), t
                
                # d. solve and time buyer problem
                tic_buy = time.time()
                hhp.solve_buy_fast(t,sol,par)                  
                toc_buy = time.time()
                par.time_buy[t] = toc_buy-tic_buy
                if par.do_print:
                    print(f' solved buyer problem in {toc_buy-tic_buy:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_buy_fast[t] >= 0) & (np.isnan(sol.c_buy_fast[t]) == False)), t
                    assert np.all((sol.d_prime_buy_fast[t] >= 0) &(np.isnan(sol.d_prime_buy_fast[t]) == False)), t
                    assert np.all((sol.inv_v_buy_fast[t] >= 0) & (np.isnan(sol.inv_v_buy_fast[t]) == False)), t
                
                # e. solve and time renter problem
                tic_rent = time.time()
                hhp.solve_rent(t,sol,par)                  
                toc_rent = time.time()
                par.time_rent[t] = toc_rent-tic_rent
                
                if par.do_print:
                    print(f' solved renter problem in {toc_rent-tic_rent:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_rent[t] >= 0) & (np.isnan(sol.c_rent[t]) == False))
                    assert np.all((sol.inv_v_rent[t] >= 0) & (np.isnan(sol.inv_v_rent[t]) == False))
                
                # f. print
                toc = time.time()
                total_solve_time += toc-tic
                if par.do_print or par.do_print_period:
                    print(f' t = {t} solved in {toc-tic:.1f} secs')
        
        # print total timings
        if par.do_print or par.do_print_period:
            print(f' total precomputation time  = {par.time_vbarq.sum():.1f} secs')
            print(f' total stay-time  = {par.time_stay.sum():.1f} secs')
            print(f' total ref-time   = {par.time_ref.sum():.1f} secs')
            print(f' total buy-time   = {par.time_buy.sum():.1f} secs')
            print(f' total rent-time   = {par.time_rent.sum():.1f} secs')
            print(f' full model solved in = {total_solve_time:.1f} secs')

    ################
    #   simulate   #
    ################

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. initial wealth       
        sim.a0 = np.zeros(par.simN)

        # b. total discounted utility
        sim.utility = np.zeros(par.simN)

        # c. states and choices
        sim_shape = (par.T,par.simN)
        sim.h = np.zeros(sim_shape)
        sim.d = np.zeros(sim_shape)
        sim.Td = np.zeros(sim_shape)
        sim.Tda = np.zeros(sim_shape)
        sim.p = np.zeros(sim_shape)
        sim.y = np.zeros(sim_shape)
        sim.m = np.zeros(sim_shape)

        sim.h_prime = np.zeros(sim_shape)
        sim.h_tilde = np.zeros(sim_shape)
        sim.d_prime = np.zeros(sim_shape)
        sim.Td_prime = np.zeros(sim_shape)
        sim.Tda_prime = np.zeros(sim_shape)
        sim.discrete = np.zeros(sim_shape,dtype=np.int)  
        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)

        # c. additional output
        sim.inc_tax = np.zeros(sim_shape)
        sim.prop_tax = np.zeros(sim_shape)
        sim.interest = np.zeros(sim_shape)
        sim.ird = np.zeros(sim_shape)

        # d. euler
        euler_shape = (par.T-1,par.simN)
        sim.euler_error = np.zeros(euler_shape)
        sim.euler_error_c = np.zeros(euler_shape)
        sim.euler_error_rel = np.zeros(euler_shape)  
        
        # e. shocks
        sim.p_y_ini = np.zeros(par.simN)
        sim.p_ini = np.zeros(par.simN)
        sim.p_y = np.zeros(sim_shape)
        sim.i_y = np.zeros(sim_shape,dtype=np.int_)

    def simulate(self,do_utility=False,do_euler_error=False,ini_wealth_scale=1.0):
        """ simulate the model """
        par = self.par
        sol = self.sol
        sim = self.sim
        tic = time.time()

        # a. random shock to income and initial wealth draw
        np.random.seed(par.sim_seed) # reset seed in every simulation for reproducibility/comparability

        sim_shape = (par.T,par.simN)
        sim.p_y_ini[:] = np.random.uniform(size=par.simN)
        sim.p_y[:,:] = np.random.uniform(size=(sim_shape))
        sim.a0[:] = ini_wealth_scale*np.random.lognormal(mean=par.mu_a0,sigma=par.sigma_a0,size=par.simN)
        
        # b. call
        with jit(self) as model:
            par = model.par
            sol = model.sol
            sim = model.sim
            simulate.lifecycle(sim,sol,par)
        toc = time.time()
      
        if par.do_print or par.do_print_period:
            print(f'model simulated in {toc-tic:.1f} secs')
        
        # d. euler errors
        def norm_euler_errors(model):
            return np.log10(abs(model.sim.euler_error/model.sim.euler_error_c)+1e-8)
        tic = time.time()        
        if do_euler_error:
            with jit(self) as model:
                par = model.par
                sol = model.sol
                sim = model.sim
                simulate.euler_errors(sim,sol,par)

            sim.euler_error_rel[:] = norm_euler_errors(self)
      
        toc = time.time()
        if par.do_print and do_euler_error:
            print(f'euler errors calculated in {toc-tic:.1f} secs')
        
        # e. utility
        tic = time.time()        
        if do_utility:
            simulate.calc_utility(sim,par)
      
        toc = time.time()
        if par.do_print and do_utility:
            print(f'utility calculated in {toc-tic:.1f} secs')

    ################
    # equilibrium  #
    ################
    
#    def find_steady_state(self,ab_guess,H_guess,H_min,H_max):
#        
#        sim = self.sim
#        sol = self.sol
#        par = self.par
#        
#        # find stable bequest level
#        steady_state.bequest_loop(self,ab_guess)#

#        # check housing market clearing
#        steady_state.find_ss(self,do_print=par.do_print,)#

#    
#    #prepare_hh_ss = steady_state.prepare_hh_ss
#    find_ss = steady_state.find_ss

    ################
    #    figures   #
    ################

    def fig_lifecycle_full(self,quantiles=False):        
        figs.lifecycle_full(self,quantiles=quantiles)

    def fig_lifecycle_consav(self):
        figs.lifecycle_consav(self)

    def fig_lifecycle_housing(self):
        figs.lifecycle_housing(self)

    def fig_lifecycle_mortgage(self):
        figs.lifecycle_mortgage(self)
    
    def fig_homeownership(self):
        figs.homeownership(self)

    def fig_decision_functions(self):
        figs.decision_functions(self)
    
    def fig_example_household(self,hh_no=0):
        figs.example_household(self,hhno=hh_no)

    def fig_n_chi_iniwealth(self):
        figs.n_chi_iniwealth(self)




