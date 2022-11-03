""" HAHModel

Solves a Heterogeneous Agent Housing Market (HAH) model building on the EconModel and 
consav packages as well as the NEGM algorithm proposed by Jeppe Druedahl.

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
        self.namespaces = ['par','sim','sol','ss'] 
        
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
        par.zeta = 0.8                                  # disutility of default
        
        # b. demographics and life cycle profile
        par.median_income = 350_000                     # for normalisation
        par.Tmin = 25                                   # age when entering the model
        par.T = 80 - par.Tmin                           # age of death
        par.Tr = 65 - par.Tmin                          # retirement age
        par.chi = np.ones(par.T)                        # placeholder for deterministic income process
        par.n = np.ones(par.T)                          # placeholder for equivalence scale 

        # c. income process
        par.rho_p = 0.96                                # AR(1) parameter
        par.sigma_psi = 0.10                            # std. of persistent shock
        par.Np = 7                                      # number of permanent income states

        par.sigma_xi = 0.0                              # std. dev. of transitory shock
        par.Nxi = 1                                     # quadrature nodes for transitory shock

        par.pi = 0.025                                  # unemployment probability
        par.b = 0.2                                     # unemployment benefits
        par.Nu = 2                                      # employed or unemployed 

        # d. interest rates and financial regulation
        par.r = 0.01                                    # return on liquid assets
        par.r_m = 0.03                                  # amortising mortgage interest rate     
        par.r_da = 0.045                                # deferred amortisation mortgage rate
        par.omega_ltv = 0.8                             # loan-to-value ratio  
        par.omega_dti = 5                               # debt-to-income ratio
        par.Cp_ref = 0.05                               # proportional refinancing cost
        par.Cf_ref = 2                                  # fixed refinancing cost NB: dummy value
        par.Td_bar = 30                                 # maximum regulatory mortgage term length
        par.Td_shape = 27                               # sample space for mortgage terminal periods
        par.Tda_bar = 11                                # maximum terms with deferred amortisation +1
        
        # e. housing and rental markets
        par.delta = 0.015                               # proportional maintenance cost
        par.gamma = 0.008                               # per rental unit operating cost
        par.C_buy = 0.06                                # proportional house sale cost
        par.C_sell = 0.04                               # proportional house purchase cost
        #par.sigma_epsilon = 0.04                       # std. dev. of housing shock
        #par.Nepsilon = 5                               # quadrature nodes for housing shock

        # f. taxation
        par.tau_y0 = 0.19                               # income tax function parameter 1    
        par.tau_y1 = 0.18                               # income tax function parameter 2
        par.tau_h0 = 0.0092                             # bottom-bracket property tax rate
        par.tau_h1 = 0.03                               # top-bracket property tax rate
        par.tau_r = 0.30                                # tax value of interest rate expenses
        par.qh_bar = 3_040_000/par.median_income        # top-bracket property tax threshold
        par.rd_bar = 75_000/par.median_income           # high tax value of interest deduction threshold

        # g. price guesses for stationary equilibrium
        par.q = 1.0                                                     # house price guess
        par.q_r = par.gamma + par.q - (1-par.delta)/(1-par.r)*par.q     # implied rental price

        # h. grids
        par.Nh = 6                                      # number of points in owner occupied housing grid
        par.h_min = 1.42                                # minimum owner occupied houze size 
        par.h_max = 5.91                                # maximum owner occupied house size
        par.Nhtilde = 3                                 # number of points in rental house size grid
        par.htilde_min = 1.07                           # minimum rental house size
        par.htilde_max = 1.89                           # maximum rental house size
        par.Nd = 10                                     # number of points in mortgage balance grid
        par.Nm = 10                                     # number of points in cash on hand grid
        par.m_max = 15.0                                # maximum cash-on-hand level  
        par.Na = 10                                     # number of points in assets grid
        par.a_max = par.m_max+1.0                       # maximum assets

        # i. simulation
        par.sigma_p0 = 0.2                              # standard dev. of initial permanent income
        #par.mu_d0 = 0.8                                 # mean initial housing level 
        
        par.sigma_d0 = 0.2                              # standard dev. of initial housing level
        par.mu_a0 = 0.2                                 # mean initial assets
        
        par.sigma_a0 = 0.1                              # standard dev. of initial assets
        
        par.simN = 10_000                               # number of simulated agents
        par.sim_seed = 1995                             # seed for random number generator
        par.euler_cutoff = 0.02                         # euler error cutoff

        # j. misc
        par.t = 0                                       # initial time
        par.tol = 1e-12                                 # solution precision tolerance
        par.do_print = False                            # whether to print solution progress
        par.do_print_period = False                     # whether to print solution progress every period
        par.max_iter_solve = 50_000                     # max iterations when solving household problem
        par.max_iter_simulate = 50_000                  # max iterations when simulating household problem
        par.include_unemp = True                        # add unemployment to income process

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
        par.grid_htilde = np.array([par.htilde_min, 1.42, par.htilde_max],dtype='double') # strictly speaking, htilde is not a state
        
        # b. post-decision assets
        par.grid_a = equilogspace(0,par.a_max,par.Na)
        
        # c. shocks and income process
            # i. persistent shock/permanent income states
        _out = log_rouwenhorst(par.rho_p,par.sigma_psi,par.Np)
        par.p_grid,par.p_trans,par.p_ergodic,par.p_trans_cumsum,par.p_ergodic_cumsum = _out
        
            # ii. transitory income shock
        if par.sigma_xi > 0 and par.Nxi > 1:
            par.xi_grid,par.xi_weights = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)
            par.xi_trans = np.broadcast_to(par.xi_weights,(par.Nxi,par.Nxi))
        else:
            par.xi_grid = np.ones(1)
            par.xi_weights = np.ones(1)
            par.xi_trans = np.ones((1,1))

            # iii. unemployment
        par.u_grid = np.array([0,1],dtype='int') 
        par.u_weight = (1-par.pi,par.pi)

            # iv. combined
        if par.include_unemp: 
            par.Ny = par.Nxi*par.Np+1 # +1 to add unemployment outcome
            grid_y_emp = np.repeat(par.xi_grid,par.Np)*np.tile(par.p_grid,par.Nxi) - par.pi*par.b
            par.grid_y = np.sort(np.append(grid_y_emp,par.b))
            
            par.y_trans = np.zeros((len(par.grid_y),len(par.grid_y)))
            y_trans_inner = np.kron(par.xi_trans,par.p_trans)*(1-par.pi)
            par.y_trans[1:len(par.grid_y),1:len(par.grid_y)] = y_trans_inner
            par.y_trans[0,:] = par.pi # fill top row with probability par.pi
            par.y_trans[:,0] = par.pi # fill leftmost col with probability par.pi
        else:
            par.Ny = par.Nxi*par.Np 
            par.grid_y = np.repeat(par.xi_grid,par.Np)*np.tile(par.p_grid,par.Nxi) - par.pi*par.b      
            par.y_trans = np.kron(par.xi_trans,par.p_trans)*(1-par.pi)
        
        par.y_trans_cumsum = np.cumsum(par.y_trans,axis=1)
        #par.w_ergodic = find_ergodic(par.p_trans)
        #par.w_ergodic_cumsum = np.cumsum(par.p_ergodic)
        par.y_trans_T = par.y_trans.T
        
            # v. house price shock if selling

        # d. set seed
        np.random.seed(par.sim_seed)

        # e. timing
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
        #fastpar['do_print'] = False
        fastpar['do_print_period'] = False
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
        fastpar['Na'] = 3
        fastpar['simN'] = 2

        # b. apply
        for key,val in fastpar.items():
            prev = getattr(par,key)
            setattr(par,key,val)
            fastpar[key] = prev

        self.allocate()
        
        # c. solve
        self.solve(do_assert=False)

        # d. simulate
        #self.simulate()

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
        own_shape = (par.T,par.Nh,par.Nd,par.Td_shape,par.Tda_bar,par.Ny,par.Nm)
        buy_shape = (par.T,par.Nh+1,par.Nd,par.Td_shape,par.Tda_bar,par.Ny,par.Nm)
        rent_shape = (par.T,par.Nhtilde,par.Ny,par.Nm)
        post_shape = (par.T,par.Nh+1,par.Nd,par.Td_shape,par.Tda_bar,par.Ny,par.Na)

        # b. stay        
        sol.c_stay = np.zeros(own_shape)
        sol.inv_v_stay = np.zeros(own_shape)
        sol.inv_marg_u_stay = np.zeros(own_shape)

        # c. refinance
        sol.c_ref = np.zeros(own_shape)
        sol.d_prime_ref = np.zeros(own_shape)
        sol.Tda_prime_ref = np.zeros(own_shape)
        sol.inv_v_ref = np.zeros(own_shape)
        sol.inv_marg_u_ref = np.zeros(own_shape)

        # d. buy
        sol.c_buy = np.zeros(buy_shape)
        sol.h_buy = np.zeros(buy_shape)
        sol.d_prime_buy = np.zeros(buy_shape)
        sol.Tda_prime_buy = np.zeros(buy_shape)
        sol.inv_v_buy = np.zeros(buy_shape)
        sol.inv_marg_u_buy = np.zeros(buy_shape)

        # e. rent
        sol.c_rent = np.zeros(rent_shape)
        sol.htilde = np.zeros(rent_shape)
        sol.inv_v_rent = np.zeros(rent_shape)
        sol.inv_marg_u_rent = np.zeros(rent_shape)
            
        # f. post decision (overarching) and endogenous arrays 
        sol.inv_v_bar = np.nan*np.zeros(post_shape)
        sol.q = np.nan*np.zeros(post_shape)

        sol.c_endo_stay = np.nan*np.zeros(post_shape)
        sol.m_endo_stay = np.nan*np.zeros(post_shape)
        sol.c_endo_rent = np.nan*np.zeros(rent_shape) 
        sol.m_endo_rent = np.nan*np.zeros(rent_shape)

    def solve(self,do_assert=True):
        """ solve the model using NEGM and NVFI
        
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
                        assert np.all((sol.inv_v_bar[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                      (np.isnan(sol.inv_v_bar[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                        assert np.all((sol.q[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                      (np.isnan(sol.q[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t

                else: 
                    tic_w = time.time()
                    hhp.postdecision_compute_v_bar_q(t,sol,par)                
                    toc_w = time.time()
                    par.time_vbarq[t] = toc_w-tic_w
                    if par.do_print:
                        print(f' v_bar and q computed in {toc_w-tic_w:.1f} secs')
                    
                    if do_assert: 
                        assert np.all((sol.inv_v_bar[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                      (np.isnan(sol.inv_v_bar[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                        assert np.all((sol.q[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                      (np.isnan(sol.q[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                        
                        #assert np.all((sol.htilde[t] >= 0) & (np.isnan(sol.htilde[t]) == False))
                        #assert np.all((sol.h_buy[t] >= 0) & (np.isnan(sol.h_buy[t] == False)))
                
                # b. all other periods
                    # ii. solve and time stayer problem
                tic_stay = time.time()
                hhp.solve_stay(t,sol,par)
                toc_stay = time.time()
                par.time_stay[t] = toc_stay-tic_stay
                
                if par.do_print:
                    print(f' solved stayer problem in {toc_stay-tic_stay:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_stay[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.c_stay[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                    assert np.all((sol.inv_v_stay[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.inv_v_stay[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                
                # iii. solve and time refinance problem
                tic_ref = time.time()
                hhp.solve_ref(t,sol,par)                  
                toc_ref = time.time()
                par.time_ref[t] = toc_ref-tic_ref
                
                if par.do_print:
                    print(f' solved refinance problem in {toc_ref-tic_ref:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_ref[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.c_ref[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                    assert np.all((sol.d_prime_ref[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.d_prime_ref[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                    assert np.all((sol.inv_v_ref[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.inv_v_ref[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                
                # iv. solve and time buyer problem
                tic_buy = time.time()
                hhp.solve_buy(t,sol,par)                  
                toc_buy = time.time()
                par.time_buy[t] = toc_buy-tic_buy
                if par.do_print:
                    print(f' solved buyer problem in {toc_buy-tic_buy:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_buy[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.c_buy[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                    assert np.all((sol.d_prime_buy[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) &
                                  (np.isnan(sol.d_prime_buy[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                    assert np.all((sol.inv_v_buy[t,0,0,0:Td_len,0:Tda_len,:,:] >= 0) & 
                                  (np.isnan(sol.inv_v_buy[t,0,0,0:Td_len,0:Tda_len,:,:]) == False)), t
                
                # v. solve and time renter problem
                tic_rent = time.time()
                hhp.solve_rent(t,sol,par)                  
                toc_rent = time.time()
                par.time_rent[t] = toc_rent-tic_rent
                
                if par.do_print:
                    print(f' solved renter problem in {toc_rent-tic_rent:.1f} secs')
                if do_assert:
                    assert np.all((sol.c_rent[t] >= 0) & (np.isnan(sol.c_rent[t]) == False))
                    assert np.all((sol.inv_v_rent[t] >= 0) & (np.isnan(sol.inv_v_rent[t]) == False))
                
                # c. print
                toc = time.time()
                total_solve_time += toc-tic
                if par.do_print or par.do_print_period:
                    print(f' t = {t} solved in {toc-tic:.1f} secs')
        
        # print total timings
        if par.do_print:
            print(f' total precomputation time  = {par.time_vbarq.sum():.1f} secs')
            print(f' total stay-time  = {par.time_stay.sum():.1f} secs')
            print(f' total ref-time   = {par.time_ref.sum():.1f} secs')
            print(f' total buy-time   = {par.time_buy.sum():.1f} secs')
            print(f' total rent-time   = {par.time_rent.sum():.1f} secs')
            print(f' full model solved in = {total_solve_time} secs')

    ################
    #   simulate   #
    ################

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. initial values
        sim.p0 = np.zeros(par.simN)        
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
        sim.d_prime = np.zeros(sim_shape)
        sim.Td_prime = np.zeros(sim_shape)
        sim.Tda_prime = np.zeros(sim_shape)
        sim.discrete = np.zeros(sim_shape,dtype=np.int)  
        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        
        # c. euler
        euler_shape = (par.T-1,par.simN)
        sim.euler_error = np.zeros(euler_shape)
        sim.euler_error_c = np.zeros(euler_shape)
        sim.euler_error_rel = np.zeros(euler_shape)  
        
        # d. shocks
        sim.psi = np.zeros(sim_shape)
        sim.xi = np.zeros(sim_shape)
        sim.u = np.zeros(sim_shape)
        #sim.z = np.zeros(par.T)    # economy wide shock

    def simulate(self,do_utility=False,do_euler_error=False):
      """ simulate the model """
      par = self.par
      sol = self.sol
      sim = self.sim
      tic = time.time()

      # a. random shock
      sim.p0[:] = np.random.lognormal(mean=-0.2,sigma=par.sigma_p0,size=par.simN)
      sim.a0[:] = par.mu_a0*np.random.lognormal(mean=-0.2,sigma=par.sigma_a0,size=par.simN)
      
      I = np.random.choice(par.Nshocks,
          size=(par.T,par.simN), 
          p=par.psi_w*par.xi_w*par.z_w)
      sim.psi[:,:] = par.psi[I]
      sim.xi[:,:] = par.xi[I]
      sim.u[:,:] = par.u
      #sim.z[:] = par.z[I[:,0]]
      
      # b. call
      with jit(self) as model:
          par = model.par
          sol = model.sol
          sim = model.sim
          simulate.lifecycle(sim,sol,par)
      toc = time.time()
    
      if par.do_print:
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
      if par.do_print:
          print(f'euler errors calculated in {toc-tic:.1f} secs')
      
      # e. utility
      tic = time.time()        
      if do_utility:
          simulate.calc_utility(sim,sol,par)
    
      toc = time.time()
      if par.do_print:
          print(f'utility calculated in {toc-tic:.1f} secs')

    ################
    #    GenEq     #
    ################

    #prepare_hh_ss = steady_state.prepare_hh_ss
    #find_ss = steady_state.find_ss

    ################
    #    Figures   #
    ################

    #def decision_functions(self):
    #    figs.decision_functions(self)

    #def egm(self):        
    #    figs.egm(self)
 
    def lifecycle(self,quantiles=False):        
        figs.lifecycle(self,quantiles=quantiles)
 
    def mpc_over_cash_on_hand(self):
        figs.mpc_over_cash_on_hand(self)
 
    def mpc_over_lifecycle(self):
        figs.mpc_over_lifecycle(self)


    ################
    #    Tables    #
    ################


