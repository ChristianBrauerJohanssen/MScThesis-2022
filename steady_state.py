import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed
from consav import jit

import root_finding
import simulate

def bequest_loop(model,draws,bequest_guess=1,step_size=0.5):
    """ simulate the model to match initial wealth with bequests
    
    args:
        model: a HAH model object
        draws: a draw from the initial wealth distribution (fixed in all simulations)
        bequest_guess (optional): guess for scaling of the initial wealth distribution
        step_size (optional): step size for updating the guess
    """


    
    # a. unpack
    sol = model.sol
    sim = model.sim
    par = model.par

    par.do_print = False

    # b. simulate the model for initial guess and fixed draws
    #draws = np.random.lognormal(mean=par.mu_a0,sigma=par.sigma_a0,size=par.simN)
    sim.a0[:] = bequest_guess*draws
        
        # b. call
    with jit(model) as model_jit:
        par = model_jit.par
        sol = model_jit.sol
        sim = model_jit.sim
        simulate.lifecycle(sim,sol,par)
    
    discrepancy = np.mean(sim.a0) - np.mean(sim.a[par.T-1,:])

    # c. iterate until initial wealth match bequest
    iteration = 0
    while np.abs(discrepancy) > par.tol*10**6 and iteration < par.max_iter_simulate:
    
        # update mean initial wealth
        bequest_guess -= step_size*discrepancy

        # simulate model
        with jit(model) as model:
            sim.a0[:] = bequest_guess*draws
            simulate.lifecycle(sim,sol,par)
        
        discrepancy = np.mean(sim.a0) - np.mean(sim.a[par.T-1,:])

        # update counter and print statement
        iteration += 1
        if iteration%1 == 0:
            print(f'iteration = {iteration}, discrepancy = {discrepancy:.6f}')

    # terminal statement
    if np.abs(discrepancy) > par.tol*10**6:
        print(f'stable bequest not found in {iteration} simulations')
    else:
        print(f'convergence achieved in {iteration} simulations, mean bequest = {bequest_guess:.6f}')
    
    par = model.par
    par.do_print = True
    

def obj_ss(H_ss_guess,model,do_print=False):
    """ objective when solving for steady state housing """

    par = model.par
    sim = model.sim
    ss = model.ss

    # a. aggregate housing supply
    ss.H = H_ss_guess

    # b. price guesses
    ss.q = par.q 
    ss.q_r = par.q_r

    # c. household behavior
    if do_print:

        print(f'guess {ss.H = :.4f}')    

    model.solve()
    model.simulate()

    ss.H_hh = np.sum(sim.h)+np.sum

    if do_print: print(f'implied {ss.H_hh = :.4f}')

    # d. market clearing
    ss.clearing_H = ss.H-ss.H_hh

    return ss.clearing_H # target to hit
    
def find_ss(model,method='direct',do_print=False,H_min=1.0,H_max=10.0,NK=10):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,H_min=H_min,H_max=H_max,NK=NK)
    elif method == 'indirect':
        find_ss_indirect(model,do_print=do_print)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,do_print=False,H_min=1.0,H_max=10.0,NK=10):
    """ find steady state using direct method """

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    H_ss_vec = np.linspace(H_min,H_max,NK) # trial values
    clearing_H = np.zeros(H_ss_vec.size) # asset market errors

    for i,H_ss in enumerate(H_ss_vec):
        
        try:
            clearing_H[i] = obj_ss(H_ss,model,do_print=do_print)
        except Exception as e:
            clearing_H[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_H = {clearing_H[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    H_max = np.min(H_ss_vec[clearing_H < 0])
    H_min = np.max(H_ss_vec[clearing_H > 0])

    if do_print: print(f'H in [{H_min:12.8f},{H_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,H_min,H_max,args=(model,),do_print=do_print,
        varname='H_ss',funcname='H_hh-H'
    )

def find_ss_indirect(model,do_print=False):
    """ find steady state using indirect method """

    par = model.par
    sim = model.sim
    ss = model.ss

    # a. exogenous and targets
    ss.q = par.q
    ss.q_r = par.q_r

    assert (1+par.r)*par.beta < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    # b. stock and capital stock from household behavior
    model.solve(do_print=do_print)      
    model.simulate(do_print=do_print) 
    if do_print: print('')

    ss.H = ss.H_hh = np.sum(sim.h)
    
    # e. print
    if do_print:

        print(f'Implied H = {ss.H:6.3f}')
        print(f'House price = {ss.q:6.3f}')
        print(f'Rental price = {ss.q_r:6.3f} ')
        print(f'Mean bequest = ')
        print(f'Mean initial assets = ')
        print(f'Discrepancy in a0-ab_tot = {ss.K-ss.A_hh:12.8f}') # = 0 by construction
        print(f'Discrepancy in H-H_hh = {ss.C-ss.C_hh:12.8f}\n') # != 0 due to numerical error 