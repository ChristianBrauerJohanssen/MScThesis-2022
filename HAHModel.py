# imports 
    # a. standard packages
import numpy as np
import time

    # b. NumEconCph packages
from EconModel import EconModelClass
from GEModelTools import GEModelClass
import consav

    # c. local modules
import steady_state
import HHproblems
import trans
import utility
import simulate 
import figs

class HAHModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces 
        self.namespaces = ['par','ini','ss','path','sim'] 
        
        # inherits all this automatically from EconModelClass?
        ## c. savefolder
        #self.savefolder = 'saved'

        ## d. list not-floats for safe type inference
        #self.not_floats = ['housing_shock','solmethod','T','t','simN','sim_seed',
        #                   'Npsi','Nxi','Nm','Np','Nn','Nx','Na','Nshocks',
        #                   'do_print','do_print_period','do_marg_u']

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','w'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used)
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['Gamma'] # exogenous shocks
        self.unknowns = ['K'] # endogenous unknowns
        self.targets = ['clearing_A'] # targets = 0

        # d. all variables
        self.varlist = [
            'A_hh','C_hh','C','clearing_A','clearing_C',
            'Gamma','I','K','L','r','rk','w','Y']

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = None # not used today
        self.block_post = None # not used today

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1 # number of fixed discrete states (none here)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.beta = 0.965    # subjective discount factor
        par.sigma = 2.0     # CRRA coefficient
        par.alpha = 1.1     # housing curvature
        par.nu = 0.26       # weight on housing
        par.phi = 0.85      # scaling of housing services from rent
        par.kappa = 0.34    # utility cost of moving
        par.thetab = 100    # strength of bequest motive 

        # b. life cycle and income
        par.Tmin = 25                                   # age when entering the model
        par.T = 80 - par.Tmin                           # age of death
        par.Tr = 65 - par.Tmin                          # retirement age
        par.G = 1.02                                    # growth in permanent income
        par.L = np.ones(par.T-1)                        # income profile
        par.L[0:par.Tr] = np.linspace(1,1/par.G,par.Tr) # hump shaped permanet income while working
        par.L[par.Tr-1] = 0.67                          # drop in permanent income at retirement age
        par.L[par.Tr-1:] = par.L[par.Tr-1:]/par.G       # constant permanent income after retirement

        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of shock

        # c. production and investment
        par.alpha = 0.36 # cobb-douglas
        par.delta = 0.10 # depreciation rate
        par.Gamma_ss = 1.0 # direct approach: technology level in steady state

        # f. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.03
        par.w_ss_target = 1.0

        # h. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        
    def allocate(self):
        """ allocate model """

        par = self.par

        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss