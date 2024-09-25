#  This program values deSPAC warrants using Monte Carlo simulation. The warrant
#  pricing function reproduces values from Table 2 in Carrion, Imerman and Zhang (2024).
#  An example invocation reproduces a specific value and by varying input arugments
#  and keeping the preset random number generator seed the whole table can be replicated.
import numpy as np 

#  Discounted cash flow helper function
#  Inputs: a 2D cash flow array of any size (paths x timesteps) and and a discount rate. 
#          First column of cash flow array is 1 timestep in the future, does not handle time 0 cash flows.
#  Output: returns a 1D array of present values, size is one per path
def DCF(cf_in, disc_factor):
    #  if cf_in is a 1-D array, make it a 2-D array with a single column
    if cf_in.ndim == 1:
        cf_in = cf_in.reshape(cf_in.shape[0],1)
    #  initialize output array
    dcf_out = np.zeros(( cf_in.shape[0], cf_in.shape[1]) )
                          
    #  this block handles cases where input cash flow array has at least 2 time periods
    if cf_in.shape[1]>1:
        #  initialize last period in output
        dcf_out[:,-1] = disc_factor * cf_in[:,-1] + cf_in[:,-2]
        
        #  if there are more than 2 periods in input cash flow array, discount next period DCF 
        #     and add to cash flow recieved in current period
        if cf_in.shape[1]>2:
            for t in range(cf_in.shape[1]-2, 0,-1):
                dcf_out[:,t] = disc_factor * dcf_out[:,t+1] + cf_in[:,t-1]
        #  discount back one more period to present values
        dcf_out[:,0] = disc_factor * dcf_out[:,1]
        
    # handle single time period cases, can just calculate output in one step
    else:
         dcf_out = disc_factor * cf_in
         
    return dcf_out[:,0].copy()

#  Warrant pricing function
#  Inputs: starting underlying price, time to maturity in years, and volatility
#          (other hard-coded pricing inputs and algorithm parameters can easily be converted to input arguments)
#  Output: returns warrant price and MC standard error
#  Notes: hard codes typical de-SPAC warrant characteristics, assumes lockout has expired, uses risk-free rate 5%.
#         Monte Carlo simulation uses daily timesteps, 100,000 paths (50,000 random + 50,000 antithetic)
def price_warrant(S0, T, vol):
    #  hardcoded warrant characteristics and market inputs
    #  warrant strike price
    X = 11.5
    #  remaining lockout before warrant is redeemable or exercisable in days
    lockout = 0
    #  risk-free rate
    r = .05           
    #  simulation parameters/pre-calculated constants
    #  number of paths
    N = 100000
    #  step size = 1 day
    dt = 1/252
    steps = int(T/dt)
    disc = np.exp(-r*dt)

    #  generate simulated paths for underlying
    S = np.zeros(shape=(N,steps+1))
    S[:,0] = S0
    drift = (r- .5*vol**2)*dt
    vol_step = vol*np.sqrt(dt)
    np.random.seed(1234)
    #  draw random numbers using antithetical sampling
    draws = np.random.normal(size=(int(N/2),steps))
    draws = np.concatenate((draws, -draws))
    for t in range(1,steps+1):
        S[:,t] = S[:,t-1]*np.exp( drift + vol_step*draws[:,t-1]) #GBM from Hull 20.16

    #  count barrier crosses in 30 day window
    #  do not start of counting process until lockout has expired    
    B18 = np.zeros(shape=(N,steps+1))
    window = 30
    for t in range(1,steps+1):
        if t > lockout:
            if t - lockout < 29:
                B18[:,t] = np.sum(S[:,lockout:t+1] >= 18, axis=1)
            else:
                B18[:,t] = np.sum(S[:,t-window+1:t+1] >= 18, axis=1)
    #  set prices to 0 starting in 22 trading days after barrier cross count reaches 20
    #  this is a convenient way to mechanically flag redemption effectiveness after 22 day delay
    for t in range(0,steps-1):
        S[B18[:,t]>=20.0,t+22:] = 0                

    #  initialize Warrant cash flows and stop rule, exercising at maturity on all ITM paths
    #  warrant cash flows
    W = np.zeros(shape=(N,steps+1))
    W[:,steps] = np.maximum(S[:,steps]-X, np.zeros(N))     
    #  stop rule
    SR = np.zeros(shape=(N,steps+1))
    SR[S[:,steps] > X, steps] = 1

    #  loop through time steps recursively, updating W and SR in each iteration
    for t in range(steps-1, 0, -1):
        #  skip loop iteration if lockout is still in effect        
        if t > lockout:           
            #  select paths (rows) to exercise
            #  exercise condition = warrant is ITM at this timestep AND redemption is effective next timestep
            #  redemption effectiveness in next step has been flagged by next underlying price set to 0
            exercises = np.where((S[:,t]>X) &( S[:,t+1]==0))

            #  create stopping rule array for this timestep (1D, maps to column in global stop rule array)
            stop_rule = np.zeros(N)
            stop_rule[exercises] = 1
            
            #  create cash flow array for this timestep (1D, maps to column in global cash flow array)       
            cash_flow = np.zeros(N)
            exercise_payoff = np.maximum(S[:,t] - X, 0)
            cash_flow = stop_rule * exercise_payoff

            #  get cash flows for current timestep conditional on no previous exercise and insert into W
            W[:,t] = cash_flow        
            #  update stopping rule
            SR[:,t] = stop_rule              
    
            #  when exercising at current time step, zero out cash flow and stop rule arrays for later times
            W[SR[:,t]==1,t+1:] = 0
            SR[SR[:,t]==1,t+1:] = 0   

        else:
            #  lockout is still in effect
            #  W and SR are already initialized to zeros, no action needed
            pass
    
        #  time 0 processing and final valuation
        path_values = DCF(W[:,1:], disc)
        
        #  warrant value and MC standard error
        w_prc = path_values.mean()
        se = path_values.std()/np.sqrt(len(path_values))
        
    return w_prc, se

#  Example invocation of warrant pricing function. This replicates the value from
#  Table 2 in Carrion, Imerman, and Zhang (2024) for underlying price=$14.00,remaining maturity=3 years, 
#  and volatility=30%.
print('Pricing warrant for S=$14.00, T=2.0yrs, vol=30% (see Carrion, Imerman, and Zhang (2024) or code comments for other inputs) ...')
print()
prc, se = price_warrant(14, 2, .3)
print('Warrant price=', prc)
print('standard error=', se)
