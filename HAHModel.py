""" HAHModel

Solves a Heterogeneous Agent Housing Market (HAH) model building on the EconModel and 
consav packages as well as the NVFI and NEGM algorithms proposed by Jeppe Druedahl.

Developer: Christian Brauer Johanssen
Date: 2022-10-05
Version: incomplete

"""
##########################
#       1. Imports       #
##########################

# a. standard packages
import numpy as np
import time

# b. NumEconCph packages
from EconModel import EconModelClass
#from GEModelTools import GEModelClass
from consav import jit
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

# c. local modules
import steady_state
import HHproblems
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
        par.Κ = 7.7                                     # extent of bequest as luxury
        par.zeta = 0.8                                  # disutility of default 

        # b. demographics and life cycle profile
        par.Tmin = 25                                   # age when entering the model
        par.T = 80 - par.Tmin                           # age of death
        par.Tr = 65 - par.Tmin                          # retirement age
        par.G = 1.02                                    # growth in permanent income
        par.L = np.ones(par.T-1)                        # income profile
        par.L[0:par.Tr] = np.linspace(1,1/par.G,par.Tr) # hump shaped permanet income while working
        par.L[par.Tr-1] = 0.67                          # drop in permanent income at retirement age
        par.L[par.Tr-1:] = par.L[par.Tr-1:]/par.G       # constant permanent income after retirement

        # c. income process
        par.sigma_psi = 0.1                             # std. dev. of permanent shock
        par.sigma_xi = 0.1                              # std. dev. of transitory shock
        par.pi = 0.025                                  # unemployment probability
        par.b = 0.2                                     # unemployment benefits
        par.sigma_epsilon = 0.04                        # std. dev. of housing shock
        par.Npsi = 5                                    # quadrature nodes for permanent shock
        par.Nxi = 5                                     # quadrature nodes for transitory shock
        par.Nepsilon = 5                                # quadrature nodes for housing shock

        # d. interest rates and financial regulation
        par.r = 0.01                                    # return on liquid assets
        par.r_m = 0.03                                  # amortising mortgage interest rate     
        par.r_da = 0.045                                # deferred amortisation mortgage rate
        par.omega_ltv = 0.8                             # loan-to-value ratio  
        par.omega_dti = 5                               # debt-to-income ratio
        par.Cp_ref = 0.05                               # proportional refinancing cost
        par.Cf_ref = 2                                  # fixed refinancing cost NB: dummy value  

        # e. housing and rental markets
        par.delta = 0.015                               # proportional maintenance cost
        par.gamma = 0.008                               # per rental unit operating cost
        par.C_buy = 0.06                                # proportional house sale cost
        par.C_sell = 0.04                               # proportional house purchase cost

        # f. taxation
        par.tauy0 = 0.75                                # income tax function parameter 1    
        par.tauy1 = 0.15                                # income tax function parameter 2
        par.tauh0 = 0.0092                              # bottom-bracket property tax rate
        par.tauh1 = 0.03                                # top-bracket property tax rate
        par.qh_bar = 3_040_000                          # top-bracket property tax threshold
        par.rd_bar = 75_000                             # high tax value of interest deduction threshold

        # g. grids
        par.Np = 50                                     # number of points in permanent income grid
        par.p_min = 1e-4                                # minimum permanent income
        par.p_max = 3.0                                 # maximum permanent income

        par.Nh = 100                                    # number of points in housing level grid
        par.n_max = 8.0                                 # maximum housing level
        par.Nm = 100                                    # number of points in housing price grid

        par.m_max = 10.0                                # maximum cash-on-hand level  
        par.Nx = 100                                    # number of points in cash-on-hand (after adj) grid
        par.x_max = par.m_max + par.n_max               # maximum cash-on-hand (after adj)

        par.Na = 100                                    # number of points in assets grid
        par.a_max = par.m_max+1.0                       # maximum assets

        # h. simulation
        par.sigma_p0 = 0.2                              # standard dev. of initial permanent income
        par.mu_d0 = 0.8                                 # mean initial housing level 
        
        par.sigma_d0 = 0.2                              # standard dev. of initial housing level
        par.mu_a0 = 0.2                                 # mean initial assets
        
        par.sigma_a0 = 0.1                              # standard dev. of initial assets
        
        par.simN = 10_000                               # number of simulated agents
        par.sim_seed = 1995                             # seed for random number generator
        par.euler_cutoff = 0.02                         # euler error cutoff

        # i. misc
        par.solmethod = 'negm'                          # default solution method
        par.t = 0                                       # initial time
        par.tol = 1e-12                                 # solution precision tolerance
        par.do_print = False                            # whether to print solution progress
        par.do_print_period = False                     # whether to print solution progress every period
        par.do_marg_u = False                           # calculate marginal utility for use in egm
        par.max_iter_solve = 50_000                     # max iterations when solving household problem
        par.max_iter_simulate = 50_000                  # max iterations when simulating household problem

        # l. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.03
        par.w_ss_target = 1.0

    def allocate(self):
        """ allocate model """

        par = self.par

        self.create_grids()                             # create grids
        self.solve_prep()                               # allocate solution arrays
        self.simulate_prep()                            # allocate simulation arrays

    def create_grids(self):
        """ construct grids for states and shocks """
        
        par = self.par

        if par.solmethod == 'negm': 
            par.do_marg_u = True                       # endogenous grid point method setting

        # a. states        
        par.grid_p = equilogspace(par.p_min,par.p_max,par.Np)
        par.grid_n = equilogspace(0,par.n_max,par.Nn)
        par.grid_m = equilogspace(0,par.m_max,par.Nm)
        par.grid_x = equilogspace(0,par.x_max,par.Nx)
        
        # b. post-decision states
        par.grid_a = np.nan + np.zeros((par.Nn,par.Na))
        # loop for potentially having a dynamic post decision grid
        for i_n in range(par.Nn): 
           par.grid_a[i_n,:] = equilogspace(0,par.a_max,par.Na)
        
        # c. shocks
        shocks = create_PT_shocks(
            sigma_psi=par.sigma_psi,
            Npsi=par.Npsi,
            sigma_xi=par.sigma_xi,
            Nxi=par.Nxi,
            sigma_epsilon=par.sigma_epsilon,
            Nz=par.Nz,
            gamma=par.gamma,
            pi=par.pi,
            )
        par.psi,par.psi_w,par.xi,par.xi_w,par.z,par.z_w,par.Nshocks = shocks

        # d. set seed
        np.random.seed(par.sim_seed)

        # e. timing
        par.time_w = np.zeros(par.T)
        par.time_stay = np.zeros(par.T)
        par.time_adj = np.zeros(par.T)
        par.time_adj_full = np.zeros(par.T)

    #############
    #   Solve   #
    #############

    def precompile_numba(self):
        """ solve the model with very coarse grids and simulate with very few persons
            --> quicker for numba to analyse the code 
        """

        par = self.par

        tic = time.time()

        # a. define
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_period'] = False
        fastpar['T'] = 2
        fastpar['Np'] = 3
        fastpar['Nn'] = 3
        fastpar['Nm'] = 3
        fastpar['Nx'] = 3
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

        # a. standard
        stay_shape = (par.T,par.Np,par.Nn,par.Nm)        
        sol.c_stay = np.zeros(stay_shape)
        sol.inv_v_stay = np.zeros(stay_shape)
        sol.inv_marg_u_stay = np.zeros(stay_shape)

        adj_shape = (par.T,par.Np,par.Nx)
        sol.d_adj = np.zeros(adj_shape)
        sol.c_adj = np.zeros(adj_shape)
        sol.inv_v_adj = np.zeros(adj_shape)
        sol.inv_marg_u_adj = np.zeros(adj_shape)
            
        post_shape = (par.T-1,par.Np,par.Nn,par.Na)
        sol.inv_w = np.nan*np.zeros(post_shape)
        sol.q = np.nan*np.zeros(post_shape)
        sol.q_c = np.nan*np.zeros(post_shape)
        sol.q_m = np.nan*np.zeros(post_shape)


    def solve(self,do_assert=True):
        """ solve the model
        
        Args:

            do_assert (bool,optional): make assertions on the solution
        
        """
        total_solve_time = 0

        tic = time.time()
        
        # backwards induction
        for t in reversed(range(self.par.T)):
            
            self.par.t = t

            with jit(self) as model:

                par = model.par
                sol = model.sol
                
                # i. last period
                if t == par.T-1:

                    last_period.solve(t,sol,par)

                    if do_assert:
                        assert np.all((sol.c_stay[t] >= 0) & (np.isnan(sol.c_stay[t]) == False))
                        assert np.all((sol.inv_v_stay[t] >= 0) & (np.isnan(sol.inv_v_stay[t]) == False))
                        assert np.all((sol.d_adj[t] >= 0) & (np.isnan(sol.d_adj[t]) == False))
                        assert np.all((sol.c_adj[t] >= 0) & (np.isnan(sol.c_adj[t]) == False))
                        assert np.all((sol.inv_v_adj[t] >= 0) & (np.isnan(sol.inv_v_adj[t]) == False))

                # ii. all other periods
                else:
                    
                    # o. compute post-decision functions
                    tic_w = time.time()

                    if par.solmethod == 'nvfi':
                        post_decision.compute_wq(t,sol,par)
                    elif par.solmethod == 'negm': 
                        post_decision.compute_wq(t,sol,par,compute_q=True)                
                    else: 
                        pass
                    toc_w = time.time()
                    par.time_w[t] = toc_w-tic_w
                    if par.do_print and par.solmethod != 'vfi':
                        print(f' w computed in {toc_w-tic_w:.1f} secs')

                    if do_assert and par.solmethod in ['nvfi','negm']:
                        assert np.all((sol.inv_w[t] > 0) & (np.isnan(sol.inv_w[t]) == False)), t 
                        if par.solmethod in ['negm']:                                                       
                            assert np.all((sol.q[t] > 0) & (np.isnan(sol.q[t]) == False)), t

                    # oo. solve stayer problem
                    tic_stay = time.time()
                    
                    if par.solmethod == 'vfi':
                        vfi.solve_stay(t,sol,par)
                    elif par.solmethod == 'nvfi':                
                        nvfi.solve_stay(t,sol,par)
                    elif par.solmethod == 'negm':
                        negm.solve_stay(t,sol,par)                                     

                    toc_stay = time.time()
                    par.time_stay[t] = toc_stay-tic_stay
                    if par.do_print:
                        print(f' solved stayer problem in {toc_stay-tic_stay:.1f} secs')

                    if do_assert:
                        assert np.all((sol.c_stay[t] >= 0) & (np.isnan(sol.c_stay[t]) == False)), t
                        assert np.all((sol.inv_v_stay[t] >= 0) & (np.isnan(sol.inv_v_stay[t]) == False)), t

                    # ooo. solve adjuster problem
                    tic_adj = time.time()
                    
                    if par.solmethod == 'vfi':
                        vfi.solve_adj(t,sol,par)
                    elif par.solmethod in ['nvfi','negm']:
                        nvfi.solve_adj(t,sol,par)                  

                    toc_adj = time.time()
                    par.time_adj[t] = toc_adj-tic_adj
                    if par.do_print:
                        print(f' solved adjuster problem in {toc_adj-tic_adj:.1f} secs')

                    if do_assert:
                        assert np.all((sol.d_adj[t] >= 0) & (np.isnan(sol.d_adj[t]) == False)), t
                        assert np.all((sol.c_adj[t] >= 0) & (np.isnan(sol.c_adj[t]) == False)), t
                        assert np.all((sol.inv_v_adj[t] >= 0) & (np.isnan(sol.inv_v_adj[t]) == False)), t

                # iii. print
                toc = time.time()
                total_solve_time += toc-tic
                if par.do_print or par.do_print_period:
                    print(f' t = {t} solved in {toc-tic:.1f} secs')
        if par.do_print:
            print(f' total precomputation time  = {par.time_w.sum():.1f} secs')
            print(f' total stay-time  = {par.time_stay.sum():.1f} secs')
            print(f' total adj-time   = {par.time_adj.sum():.1f} secs')

    ################
    #   Simulate   #
    ################

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. initial and final
        sim.p0 = np.zeros(par.simN)
        sim.d0 = np.zeros(par.simN)
        sim.a0 = np.zeros(par.simN)

        sim.utility = np.zeros(par.simN)

        # b. states and choices
        sim_shape = (par.T,par.simN)
        sim.p = np.zeros(sim_shape)
        sim.y = np.zeros(sim_shape)
        sim.m = np.zeros(sim_shape)

        sim.n = np.zeros(sim_shape)
        sim.discrete = np.zeros(sim_shape,dtype=np.int)

        sim.d = np.zeros(sim_shape)
        sim.c = np.zeros(sim_shape)
        sim.c_bump = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        sim.mpc = np.zeros(sim_shape)
        
        # c. euler
        euler_shape = (par.T-1,par.simN)
        sim.euler_error = np.zeros(euler_shape)
        sim.euler_error_c = np.zeros(euler_shape)
        sim.euler_error_rel = np.zeros(euler_shape)

        # d. shocks
        sim.psi = np.zeros((par.T,par.simN))
        sim.xi = np.zeros((par.T,par.simN))
        sim.z = np.zeros(par.T)    # economy wide shock

    def simulate(self,do_utility=False,do_euler_error=False):  #,seed=1998):
        """ simulate the model """

        par = self.par
        sol = self.sol
        sim = self.sim

        tic = time.time()

        # a. random shocks
        sim.p0[:] = np.random.lognormal(mean=-0.2,sigma=par.sigma_p0,size=par.simN)
        sim.d0[:] = par.mu_d0*np.random.lognormal(mean=-0.2,sigma=par.sigma_d0,size=par.simN)
        sim.a0[:] = par.mu_a0*np.random.lognormal(mean=-0.2,sigma=par.sigma_a0,size=par.simN)

        I = np.random.choice(par.Nshocks,
            size=(par.T,par.simN), 
            p=par.psi_w*par.xi_w*par.z_w)
        sim.psi[:,:] = par.psi[I]
        sim.xi[:,:] = par.xi[I]
        sim.z[:] = par.z[I[:,0]]

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


    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    ################
    #    Figures   #
    ################

    def decision_functions(self):
        figs.decision_functions(self)

    def egm(self):        
        figs.egm(self)

    def lifecycle(self,quantiles=False):        
        figs.lifecycle(self,quantiles=quantiles)

    def mpc_over_cash_on_hand(self):
        figs.mpc_over_cash_on_hand(self)

    def mpc_over_lifecycle(self):
        figs.mpc_over_lifecycle(self)


    ################
    #    Tables    #
    ################