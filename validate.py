###################################################
# validation of the model solution and simulation #
###################################################

# import packages
import numpy as np


# validate simulation
def val_sim(model):
    """" validate various aspects of the simulation """
    # unpack
    sim = model.sim 
    par = model.par

    # validate consumption decision
    I = sim.c < 0

    print(f'there are {np.sum(I)} cases of negative consumption') 
    if np.sum(I) > 0:
        print(f'ref accounts for {np.sum(I*(sim.discrete==1))} and buy for {np.sum(I*(sim.discrete==2))}.')
        print(f'stay accounts for {np.sum(I*(sim.discrete==0))} and rent for {np.sum(I*(sim.discrete==3))}')
        print(f'the share of negative consumption cases is {np.sum(I)/(par.simN*par.T)}')
        print()
        print('negative simulated consumption occurs in periods:')
        print(np.unique(np.where(sim.c < 0)[0],return_counts=True)[0])
        print('and cases per period are:')
        print(np.unique(np.where(sim.c < 0)[0],return_counts=True)[1])

    # assert that there is no uncollateralised debt in the simulation
    dp_bool = sim.d_prime>0
    hp_bool = sim.h_prime == 0
    assert np.sum(dp_bool*hp_bool) == 0, 'there is uncollateralised debt in the simulation'
    print('there is no uncollateralised debt in the simulation')

    # assert that there is no instances of neither buying nor renting
    hp0 = sim.h_prime == 0
    ht0 = sim.h_tilde == 0
    assert np.sum(hp0*ht0) == 0, f'there are {np.sum(hp0*ht0)} '+'instances of h_t=0 and h^{tilde}_t=0$'
    print('there are no instances of neither buying nor renting')

    # check for errors in housing stock
    bool_buy = sim.discrete == 2
    bool_stay = sim.discrete == 0
    bool_ref = sim.discrete == 1
    bool_rent = sim.discrete == 3
    
    if np.sum(sim.h_prime[bool_buy] == 0) > 0:
        print('there are instances of zero housing stock for buyers')
        print(f'number of buyers with zero housing stock is {np.sum(sim.h_prime[bool_buy] == 0)}')
    if np.sum(sim.h_prime[bool_stay] == 0) > 0:
        print('there are instances of zero housing stock for stayers')
        print(f'number of stayers with zero housing stock is {np.sum(sim.h_prime[bool_stay] == 0)}')
    if np.sum(sim.h_prime[bool_ref] == 0) > 0:
        print('there are instances of zero housing stock for refinancers')
        print(f'number of refinancers with zero housing stock is {np.sum(sim.h_prime[bool_ref] == 0)}')
    if np.sum(sim.h_prime[bool_rent] > 0) > 0:
        print('there are instances of positive housing stock for renters')
        print(f'number of renters with positive housing stock is {np.sum(sim.h_prime[bool_rent] > 0)}')
    
    full_check = np.sum(sim.h_tilde[bool_buy] == 0)*np.sum(sim.h_prime[bool_stay]*np.sum(sim.h_prime[bool_ref] == 0)*np.sum(sim.h_prime[bool_rent] > 0))
    if full_check == 0:
        print('there are no errors in the housing stock')
    

# validate financial regulation impact
def val_finreg(model):
    """ validate the impact of financial regulation in the simulation """
    # a. unpack 
    sim = model.sim
    par = model.par

    # b. compute
    D = sim.d > 0 # take mortgage?
    Dp = sim.d_prime > 0
    DA = sim.Tda_prime > 0 # choose deferred amortisation?
    t_mat = np.outer(np.arange(par.T),np.ones(par.simN))
    D_org = Dp*(sim.Td_prime - (par.Td_bar) == t_mat) # boolean for loan originations

    ltv_denom = sim.h_prime[D_org]
    ltv_num = sim.d_prime[D_org]
    dti_denom = sim.y[D_org]
    ltvs = ltv_num/ltv_denom
    dtis = ltv_num/dti_denom

    # c. print
    print(f'average mortgage size at origination is {np.mean(sim.d_prime[D_org]):.4f}')
    #print(f'sum of outstanding mortgage balances are {np.sum(sim.d):.4f}')    
    #print('the distribution of DA periods is:')
    #print(np.unique(sim.Tda_prime,return_counts=True)[0])
    #print(np.unique(sim.Tda_prime,return_counts=True)[1])
    print(f'the share of DA mortgages at origination is {np.sum(DA[D_org])/np.sum(Dp[D_org]):.4f}')
    print(f'mean LTV is {np.mean(ltvs):.4f} and mean DTI is {np.mean(dtis):.4f} at mortgage origination')


# validate calibration targets
def val_inc_calib_targets(model):
    """ validate income distribution and calibration targets """

    # a. unpack
    sim = model.sim
    par = model.par

    # b. compute
    # examine income dynamics and taxation
    tax_to_inc = np.sum(sim.inc_tax)/np.sum(sim.y)
    print(f'taxes to labour income is {tax_to_inc:.4f}')
    print(f'median pre tax income is {np.median(sim.y):.4f}')
    print(f'mean property tax is {np.mean(sim.prop_tax):.4f}')
    print(f'mean pre tax income is {np.mean(sim.y):.4f}')

    #fig = plt.figure(figsize=(12,4))
    #ax1 = fig.add_subplot(1,2,1)
    #ax1.hist(sim.y.flatten(),bins=20)
    #ax1.set_title('distribution of pre-tax income')
    #
    #ax2 = fig.add_subplot(1,2,2)
    #ax2.scatter(np.unique(sim.p.flatten()),np.unique(sim.__annotations__p.flatten(),return_counts=True)[1])
    #ax2.set_title('distribution of income states');