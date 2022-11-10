# imports
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

from consav.grids import equilogspace
from consav import linear_interp


#############
# lifecycle #
#############

def lifecycle(model,quantiles:bool=False):
    '''
    Plot the lifecycle of the model.
    Keyword arguments:
    quantiles -- if True, plot quantiles instead of mean + quantiles
    '''
    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(12,12))

    simvarlist = [('y','$y_t$ - mean pre-tax income'),
                  ('h','$h_t$ - mean house size'),
                  #('d','$d_t$ - mean debt beg.'),
                  ('d_prime','$d^{\prime}_t$ - mean debt post'),
                  ('m','$m_t$ - mean cash on hand beg.'),
                  ('c','$c_t$ - mean consumption '),
                  ('a','$a_t$ - mean liquids assets post'),                  
                  ]

    # determine number of rows in figure, given the number of columns
    cols = 2
    rows = math.ceil(len(simvarlist) / cols)

    # x-axis labels
    age = np.arange(par.T)+par.Tmin

    for i,(simvar,simvarlatex) in enumerate(simvarlist):

        ax = fig.add_subplot(rows,cols,i+1)

        simdata = getattr(sim,simvar)[:par.T,:]
        
        # plot
        if quantiles:
            if simvar not in ['discrete','mpc']:
                series = np.percentile(simdata, np.arange(0, 100, 25),axis=1)
                ax.plot(age, series.T,lw=2)
                if i == 0: ax.legend(np.arange(0, 100, 25),title='Quantiles',fontsize=8)
            else:
                ax.plot(age,np.mean(simdata,axis=1),lw=2)

        else:
            ax.plot(age,np.mean(simdata,axis=1),lw=2)
            #if simvar not in ['discrete','mpc']:
            #    ax.plot(age,np.percentile(simdata,25,axis=1),
            #        ls='--',lw=1,color='black')
            #    ax.plot(age,np.percentile(simdata,75,axis=1),
            #        ls='--',lw=1,color='black')
        ax.set_title(simvarlatex,fontsize=16)
        if par.T > 10:
            ax.xaxis.set_ticks(age[::5])
        else:
            ax.xaxis.set_ticks(age)

        ax.grid(True)
        if i in [len(simvarlist)-i-1 for i in range(cols)]:
            ax.set_xlabel('age')
    plt.tight_layout()
    plt.savefig('output/life_cycle_baseline.png')
    plt.show()

def homeownership(model):
    """ 
    Plots the share of renters, homeowners, stayers, buyers and refinancers 
    over the life cycle
    """
    # a. unpack
    par = model.par
    sim = model.sim 
    
    # b. allocate containers and compute shares
    own_share = np.zeros(par.T)
    stay_share = np.zeros(par.T)
    ref_share = np.zeros(par.T)
    buy_share = np.zeros(par.T)
    rent_share = np.zeros(par.T)
    
    for t in range(par.T):
        stay_share[t] = np.sum(sim.discrete[t]==0)/par.simN
        ref_share[t] = np.sum(sim.discrete[t]==1)/par.simN
        buy_share[t] = np.sum(sim.discrete[t]==2)/par.simN
        rent_share[t] = np.sum(sim.discrete[t]==3)/par.simN
        assert np.isclose(stay_share[t]+ref_share[t]+buy_share[t]+rent_share[t],1), print(f'discrete shares does not sum to one in period {t}') 
    
    own_share = stay_share + ref_share + buy_share
    
    simvardict = {#'owners':own_share,
                  'renters':rent_share,         
                  'stayers':stay_share,
                  'refinancers':ref_share,
                  'buyers':buy_share     
                }

    # b. figure
    fig = plt.figure(figsize=(12,12))

    # determine number of rows in figure, given the number of columns
    cols = 2
    rows = 3

    # x-axis labels
    age = np.arange(par.T)+par.Tmin
    
    for i,key in enumerate(simvardict.keys()):
        ax = fig.add_subplot(rows,cols,i+1)
        ax.plot(age,simvardict[key],lw=2)
        ax.set_title(key,fontsize=16)
        if par.T > 10:
            ax.xaxis.set_ticks(age[::5])
        else:
            ax.xaxis.set_ticks(age)
    
    plt.tight_layout()
    plt.savefig('output/homeownership_baseline.png')
    plt.show()

def mpc_over_cash_on_hand(model):
    '''plot mpc as a function of cash-on-hand for given t'''
    p_bar = np.mean(model.sim.p,axis=1)
    n_bar = np.mean(model.sim.n,axis=1)

    c0 = np.zeros(shape=(model.par.T, len(model.par.grid_m)))
    c1 = np.zeros(shape=(model.par.T, len(model.par.grid_m)))
    mpc = np.zeros(shape=(model.par.T, len(model.par.grid_m)))

    m_grid =  equilogspace(0,model.par.m_max,model.par.Nm,1.1) 

    for t in range(model.par.T):
        t = int(t)    
        for i,m in enumerate(m_grid):
            c0[t,i] = linear_interp.interp_3d(
                    model.par.grid_p,model.par.grid_n,model.par.grid_m,model.sol.c_keep[t],
                    p_bar[t],n_bar[t],m)
            c1[t,i] = linear_interp.interp_3d(
                    model.par.grid_p,model.par.grid_n,model.par.grid_m,model.sol.c_keep[t],  
                    p_bar[t],n_bar[t],m+model.par.mpc_eps)
            mpc[t,i] = (c1[t,i]-c0[t,i])/model.par.mpc_eps

    plt.figure(figsize=(9,6))
    for t in np.arange(5,model.par.T,10):
       plt.plot(model.par.grid_m,np.mean(mpc[t:t+9,:],axis=0),label='t={}-{}'.format(t+model.par.Tmin,t+model.par.Tmin+9),lw=2.3)

    plt.xlim(0,5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Cash-on-hand, $m_t$',fontsize=15)
    plt.ylabel('$\mathcal{MPC}_t$',fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig('output/mpc_over_wealth_twoasset.png')
    plt.show()

def mpc_over_lifecycle(model):

    # x-axis labels
    age = np.arange(model.par.T)+model.par.Tmin

    plt.plot(age,np.mean(model.sim.mpc,axis=1),lw=2)

    #setting labels and fontsize
    plt.xlabel('Age',fontsize=13)
    plt.ylabel('$\mathcal{MPC}_{t}$',fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig('output/mpc_lifecycle_twoasset.png')
    plt.show()
