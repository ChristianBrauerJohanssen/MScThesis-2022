###########
# imports #
###########

# a. standard packages
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
import seaborn as sns
sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

# b. NumEconCPH packages
from consav.grids import equilogspace
from consav import linear_interp

# c. local modules
import utility


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

    # b. compute additional output
    #net_wealth = sim.a + par.q*sim.h_prime - sim.d_prime

    # b. figure
    fig = plt.figure(figsize=(12,12))

    simvarlist = [('y','$y_t$ - mean pre-tax income'),
                  #('h','$h_{t-1}$ - mean house size beg.'),
                  ('h_prime','$h_t$ - mean house size'),
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
        #if quantiles:
        #    if simvar not in ['discrete','mpc']:
        #        series = np.percentile(simdata, np.arange(0, 100, 25),axis=1)
        #        ax.plot(age, series.T,lw=2)
        #        if i == 0: ax.legend(np.arange(0, 100, 25),title='Quantiles',fontsize=8)
        #    else:
        #        ax.plot(age,np.mean(simdata,axis=1),lw=2)

        
        ax.plot(age,np.mean(simdata,axis=1),lw=2)
        if simvar in ['y','c']:
            ax.plot(age,np.percentile(simdata,25,axis=1),
                ls='--',lw=1,color='black')
            ax.plot(age,np.percentile(simdata,75,axis=1),
                ls='--',lw=1,color='black')
        ax.set_title(simvarlatex,fontsize=20)
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
        ax.set_title(key,fontsize=20)
        if par.T > 10:
            ax.xaxis.set_ticks(age[::5])
        else:
            ax.xaxis.set_ticks(age)
    
    plt.tight_layout()
    plt.savefig('output/homeownership_baseline.png')
    plt.show()

def nw_and_tc(model):
    
    # a. unpack
    par = model.par
    sim = model.sim

    # b. create x axis
    age = np.arange(par.T)+par.Tmin
    
    # c. booleans for computation
    bool_stay = sim.discrete == 0
    bool_ref = sim.discrete == 1
    bool_buy = sim.discrete == 2
    bool_rent = sim.discrete == 3
    bool_da = sim.Tda > 0
    bool_dp = sim.d_prime > 0

    # d. compute net wealth end of period
    net_wealth = sim.a+sim.h_prime-sim.d_prime

    # e. compute user cost of housing
    rent_cost = bool_rent*par.q_r*sim.h_tilde
    own_cost = par.delta*par.q*sim.h + sim.prop_tax + sim.interest #+ bool_da*par.r_da*sim.d + (1-bool_da)*par.r_m*sim.d

    user_cost = rent_cost + own_cost 

    # f. compute transaction costs
    mortgage_cost = (bool_ref+bool_buy)*bool_dp*par.Cf_ref + (bool_ref+bool_buy)*sim.d*par.Cp_ref
    buy_house_cost = bool_buy*sim.h*par.C_sell + bool_buy*sim.h_prime*par.C_buy
    
    trans_cost = mortgage_cost + buy_house_cost

    # g. plot
    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(1,3,1)
    ax.plot(age,np.mean(net_wealth,axis=1),lw=2)
    ax.set_title('net wealth end of period',fontsize=14)
    plt.vlines(x=[55,75],ymin=1.0,ymax=4.0,color='orange')

    ax = fig.add_subplot(1,3,2)
    ax.plot(age,np.mean(user_cost,axis=1),lw=2)
    ax.set_title('mean user cost of housing (crude measure)',fontsize=14)

    ax = fig.add_subplot(1,3,3)
    ax.plot(age,np.mean(trans_cost,axis=1),lw=2)
    ax.set_title('mean transaction costs',fontsize=14);

######################
# decision functions #
######################

def _decision_functions(model,t,i_h,i_d,i_Td,i_Tda,i_w,i_ht,name):

    #if name == 'discrete':
    #    _discrete(model,t,i_p)
    if name == 'stay':
        _stay(model,t,i_h,i_d,i_Td,i_Tda,i_w)
    elif name == 'refinance':
        _ref(model,t,i_h,i_w)
    elif name == 'buy':
        _buy(model,t,i_w)
    elif name == 'rent':
        _rent(model,t,i_w,i_ht)
    elif name == 'post_decision': #and t <= model.par.T-2:
        _v_bar(model,t,i_h,i_d,i_Td,i_Tda,i_w)        

def decision_functions(model):
    widgets.interact(_decision_functions,
        model=widgets.fixed(model),
        t=widgets.Dropdown(description='t', 
            options=list(range(model.par.T)), value=0),
        i_h=widgets.Dropdown(description='ih',
            options=list(range(model.par.Nh)), value=0),
        i_d=widgets.Dropdown(description='i_d',
            options=list(range(model.par.Nd)), value=0),
        i_Td=widgets.Dropdown(description='i_Td',
            options=list(range(model.par.Td_shape)), value=0),
        i_Tda=widgets.Dropdown(description='i_Tda',
            options=list(range(model.par.Tda_bar)), value=0),      
        i_w=widgets.Dropdown(description='iw', 
            options=list(range(model.par.Nw)), value=np.int(model.par.Nw/2)),
        i_ht=widgets.Dropdown(description='iht', 
            options=list(range(model.par.Nhtilde)), value=0),     
        name=widgets.Dropdown(description='name', 
            options=['stay','refinance','buy','rent','post_decision'], value='stay')
        )

def _discrete(model,t,i_p):

    par = model.par

    # a. interpolation
    n, m = np.meshgrid(par.grid_n,par.grid_m,indexing='ij')
    x = m + (1-par.tau)*n
    
    inv_v_adj = np.zeros(x.size)
    linear_interp.interp_1d_vec(par.grid_x,model.sol.inv_v_adj[t,i_p,:,],x.ravel(),inv_v_adj)
    inv_v_adj = inv_v_adj.reshape(x.shape)

    # f. best discrete choice
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    I = inv_v_adj > model.sol.inv_v_keep[t,i_p,:,:]

    x = m[I].ravel()
    y = n[I].ravel()
    ax.scatter(x,y,s=2,label='adjust')
    
    x = m[~I].ravel()
    y = n[~I].ravel()
    ax.scatter(x,y,s=2,label='keep')
        
    ax.set_title(f'optimal discrete choice ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$)',pad=10)

    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # g. details
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_xlim([par.grid_m[0],par.grid_m[-1]])
    ax.set_ylabel('$n_t$')
    ax.set_ylim([par.grid_n[0],par.grid_n[-1]])
    
    plt.show()

def _stay(model,t,i_h,i_d,i_Td,i_Tda,i_w):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_c = fig.add_subplot(1,2,1,)
    ax_v = fig.add_subplot(1,2,2)

    # c. plot consumption
    ax_c.plot(par.grid_m,sol.c_stay[t,i_h,i_d,i_Td,i_Tda,i_w,:])
    ax_c.set_title(f'$c^{{stay}}$ ($t = {t}$, $w = {par.grid_w[i_w]:.2f}$, $h={par.grid_h[i_h]}$)',pad=10)

    # d. plot value function
    ax_v.plot(par.grid_m,sol.inv_v_stay[t,i_h,i_d,i_Td,i_Tda,i_w,:])
    ax_v.set_title(f'neg. inverse $v^{{stay}}$ ($t = {t}$, $w = {par.grid_w[i_w]:.2f}$, $h={par.grid_h[i_h]}$)',pad=10)

    # e. details
    for ax in [ax_c,ax_v]:

        ax.grid(True)
        ax.set_xlabel('$m_t$')
        ax.set_xlim([par.grid_m[0],par.grid_m[-1]])
    
    plt.tight_layout()
    plt.show()

def _ref(model,t,i_h,i_w):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,12))
    ax_c = fig.add_subplot(2,2,1)
    ax_d = fig.add_subplot(2,2,2)
    ax_Tda = fig.add_subplot(2,2,3)
    ax_v = fig.add_subplot(2,2,4)
    
    # c. plot consumption
    ax_c.plot(par.grid_x,sol.c_ref_fast[t,i_h,i_w,:],lw=2)
    ax_c.set_title(f'$c^{{ref}}$ ($t={t}$, $w={par.grid_w[i_w]:.2f}$, $h={par.grid_h[i_h]}$)',pad=10)

    # d. plot mortgage balance
    ax_d.plot(par.grid_x,sol.d_prime_ref_fast[t,i_h,i_w,:],lw=2)
    ax_d.set_title(f'$d\prime^{{ref}}$ ($t={t}$, $w={par.grid_w[i_w]:.2f}$, $h={par.grid_h[i_h]}$)',pad=10)

    # e. plot DA periods
    ax_Tda.plot(par.grid_x,sol.Tda_prime_ref_fast[t,i_h,i_w,:],lw=2)
    ax_Tda.set_title(f'$Tda^{{ref}}$ ($t={t}$, $w={par.grid_w[i_w]:.2f}$, $h={par.grid_h[i_h]}$)',pad=10)

    # f. plot value function
    ax_v.plot(par.grid_x,sol.inv_v_ref_fast[t,i_h,i_w,:],lw=2)
    ax_v.set_title(f'neg. inverse $v^{{ref}}$ ($t = {t}$, $w={par.grid_w[i_w]:.2f}$, $h={par.grid_h[i_h]}$)',pad=10)

    # g. details
    for ax in [ax_c,ax_d,ax_Tda,ax_v]:
        ax.grid(True)
        ax.set_xlabel('$m^{gross,ref}_t$')
        ax.set_xlim([par.grid_x[0],par.grid_x[-1]])

    plt.tight_layout()
    plt.show()

def _buy(model,t,i_w):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,12))
    ax_c = fig.add_subplot(2,3,1)
    ax_h = fig.add_subplot(2,3,2)
    ax_d = fig.add_subplot(2,3,3)
    ax_Tda = fig.add_subplot(2,3,4)
    ax_v = fig.add_subplot(2,3,5)
    
    # c. plot consumption
    ax_c.plot(par.grid_x,sol.c_buy_fast[t,i_w,:],lw=2)
    ax_c.set_title(f'$c^{{buy}}$ ($t={t}$, $w={par.grid_w[i_w]:.2f}$)',pad=10)

    # d. plot housing choice
    ax_h.plot(par.grid_x,sol.h_buy_fast[t,i_w,:],lw=2)
    ax_h.set_title(f'$h\prime^{{buy}}$ ($t={t}$, $w={par.grid_w[i_w]:.2f}$)',pad=10)

    # e. plot mortgage balance
    ax_d.plot(par.grid_x,sol.d_prime_buy_fast[t,i_w,:],lw=2)
    ax_d.set_title(f'$d\prime^{{buy}}$ ($t={t}$, $w={par.grid_w[i_w]:.2f}$)',pad=10)

    # f. plot DA periods
    ax_Tda.plot(par.grid_x,sol.Tda_prime_buy_fast[t,i_w,:],lw=2)
    ax_Tda.set_title(f'$Tda^{{buy}}$ ($t={t}$, $w={par.grid_w[i_w]:.2f}$)',pad=10)

    # g. plot value function
    ax_v.plot(par.grid_x,sol.inv_v_buy_fast[t,i_w,:],lw=2)
    ax_v.set_title(f'neg. inverse $v^{{buy}}$ ($t = {t}$, $w={par.grid_w[i_w]:.2f}$)',pad=10)

    # h. details
    for ax in [ax_c,ax_h,ax_d,ax_Tda,ax_v]:
        ax.grid(True)
        ax.set_xlabel('$m^{gross,buy}_t$')
        ax.set_xlim([par.grid_x[0],par.grid_x[-1]])

    plt.tight_layout()
    plt.show()

def _rent(model,t,i_w,i_ht):
    
    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_c = fig.add_subplot(2,2,1)
    ax_v = fig.add_subplot(2,2,2)

    # c. plot consumption
    ax_c.plot(par.grid_m,sol.c_rent[t,i_ht,i_w,:])
    ax_c.set_title(f'$c^{{rent}}$ ($t = {t}$, $w = {par.grid_w[i_w]:.2f}$, $h^{{tilde}} = {par.grid_htilde[i_ht]}$ ),',pad=10)

    # e. plot value function
    ax_v.plot(par.grid_m,sol.inv_v_rent[t,i_ht,i_w,:])
    ax_v.set_title(f'neg. inverse $v^{{rent}}$ ($t = {t}$, $w = {par.grid_w[i_w]:.2f}$, $h^{{tilde}} = {par.grid_htilde[i_ht]}$)',pad=10)

    # f. details
    for ax in [ax_c,ax_v]:

        ax.grid(True)
        ax.set_xlabel('$m_t$')
        ax.set_xlim([par.grid_m[0],par.grid_m[-1]])
    
    plt.tight_layout()
    plt.show()

def _v_bar(model,t,i_h,i_d,i_Td,i_Tda,i_w):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    # c. plot post decision value function
    ax.plot(par.grid_a,sol.inv_v_bar[t,i_h,i_d,i_Td,i_Tda,i_w,:])
    ax.set_title(f'neg. inverse  $V^{{bar}}$ ($t = {t}$',pad=10)

    # d. details
    ax.grid(True)
    ax.set_xlabel('$a_t$')
    ax.set_xlim([par.grid_a[0],par.grid_a[-1]])
    
    plt.tight_layout
    plt.show()


#########################################################################################################


###################
# MPCs (old code) #
###################

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
