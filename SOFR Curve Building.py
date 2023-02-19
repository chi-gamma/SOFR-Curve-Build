#==================================== Import the releveant packages =============================#
import QuantLib as ql
import numpy as np
from copy import deepcopy
import bisect
#================================================================================================#

#================================= Define the needed functions ==================================#
def SOFR_Swap(swapRate, startDate, endDate, discDFs):
    ''' Function to find difference between the model swap rate
    and the market swap rate for a single swap '''
    schedule = ql.Schedule(startDate, endDate, ql.Period('1Y'), calendar, ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Backward, False)
    fixedLeg = 0
    floatLeg = 0
    for i in range(1, len(schedule)):
        currDate = schedule[i]
        prevDate = schedule[i-1]
        accrualFraction = ql.Actual360().yearFraction(prevDate, currDate)
        currTenor = ql.Actual365Fixed().yearFraction(valDate, currDate)
        prevTenor = ql.Actual365Fixed().yearFraction(valDate, prevDate)
        disc_fact = discDFs(currTenor)
        p1 = discDFs(prevTenor)
        p2 = discDFs(currTenor)
        fwd_rate = (1 / accrualFraction) * (p1 / p2 - 1)
        fixedLeg += disc_fact * accrualFraction
        floatLeg += fwd_rate * disc_fact * accrualFraction
    val = (floatLeg / fixedLeg) - swapRate
    return val
       
        
def valueSwap(maturities, zeros):
    ''' Function to value all swaps and find the difference between the model swap rates
        and the market swap rates ''' 
    totalTimes = np.array(knownSofrTimes + list(maturities))
    totalZeros = np.array(knownSofrZeros + list(zeros))
    discDFs = lambda t: getDF(t, totalTimes, totalZeros)
    values = np.zeros(len(sofrSwapRates))
    # To increase performance, the last tenor alone is considered since all other
    # swaps without stub periods will have expiry dates along the schedule
    schedule = ql.Schedule(settleDate, sofrSwapDates[-1], ql.Period('1Y'), calendar, ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False)
    fixedLeg = 0
    floatLeg = 0
    scheduleDates = []
    fixeds = []
    floats = []
    for i in range(1, len(schedule)):
        currDate = schedule[i]
        prevDate = schedule[i-1]
        accrualFraction = ql.Actual360().yearFraction(prevDate, currDate)
        currTenor = ql.Actual365Fixed().yearFraction(valDate, currDate)
        prevTenor = ql.Actual365Fixed().yearFraction(valDate, prevDate)
        disc_fact = discDFs(currTenor)
        p1 = discDFs(prevTenor)
        p2 = discDFs(currTenor)
        fwd_rate = (1 / accrualFraction) * (p1 / p2 - 1)
        fixedLeg += disc_fact * accrualFraction
        floatLeg += fwd_rate * disc_fact * accrualFraction
        fixeds.append(fixedLeg)
        floats.append(floatLeg)
        scheduleDates.append(currDate)
     
    for j in range(len(sofrSwapDates)):
        if sofrSwapStub[j]:
            # all swaps with a stub period have to valued individually with a backward 
            # generated schedule
            values[j] = SOFR_Swap(sofrSwapRates[j], settleDate, sofrSwapDates[j], discDFs)
        else:
            # given the cumulative sum of fixed legs and floatlegs along the schedule of
            # the last tenor, all non-stub swaps are calculated as the difference 
            # between the model swap rate and the market swap rate
            swap_ind = scheduleDates.index(sofrSwapDates[j])
            values[j] = (floats[swap_ind] / fixeds[swap_ind]) - sofrSwapRates[j]
    return values
        
def getJacobian(maturities, zeros):
    ''' Returns the jacobian matrix of first-order sensitivities to changes from zero rates '''
    eps = 1e-6
    n = len(maturities)
    jacobian = np.zeros((n, n))
    values = valueSwap(maturities, zeros)
    zeros_up = deepcopy(zeros)
    for i in range(len(zeros)):
        zeros_up[i] = zeros[i] + eps
        values_up = valueSwap(maturities, zeros_up)
        zeros_up[i] = zeros[i]
        dv = (values_up - values) / eps
        jacobian[:,i] = dv
    return jacobian
        

def zeroCurve(maturities, zeros):
    ''' Function to determine the unknown zero rates by a multi-variate Newton-Raphson 
    Root Finding optimization procedure ''' 
    error = 10e10
    ind = 0
    while error > tol and ind <= max_iter:
        ind += 1
        values = valueSwap(maturities, zeros)
        jacobian = getJacobian(maturities, zeros)
        inv_jacobian = np.linalg.inv(jacobian)
        error_array = -1 * np.dot(inv_jacobian, values)
        zeros = zeros + error_array
        error = np.linalg.norm(error_array)
    return zeros
    
def getDF(t, times, zeros):
    ''' Perforrms linear interpolation on the
    log of discount factors using zero rates '''
    ind = bisect.bisect_left(times, t)
    if t <= times[0]:
        down, up = 0, 1
    elif t > times[-1]:
        down, up = len(times) - 2, len(times) - 1
    else:
        if times[ind]==t:
            down, up = ind, 0
        else:
            down, up = ind-1, ind
    t1, z1 = times[down], zeros[down] 
    logP1 = -t1 * z1
    t2, z2 = times[up], zeros[up] 
    logP2 = -t2 * z2
    m = (logP2 - logP1) / (t2 - t1)
    return np.exp( logP1 + (m * (t - t1)) )

#================================================================================================#

#================================== Supply Market Data Input ===================================#

valDate = ql.Date(3, 1, 2023)

sofrTenors = ['1DY', '1WK', '2WK', '3WK', '1MO', '2MO', '3MO', '4MO', '5MO', '6MO',
       '7MO', '8MO', '9MO', '10MO', '11MO', '12MO', '18MO', '2YR', '3YR',
       '4YR', '5YR', '6YR', '7YR', '8YR', '9YR', '10YR', '12YR', '15YR',
       '20YR', '25YR', '30YR', '40YR', '50YR']

sofrRates = [4.31, 4.31333, 4.31608, 4.31831, 4.3638, 4.51197, 4.602, 4.67965, 4.7537,
             4.80582, 4.84667, 4.874, 4.89325, 4.90588, 4.9083, 4.9021, 4.6865, 4.428,
             4.02275, 3.7991, 3.6686, 3.5843, 3.5245, 3.4814, 3.4529, 3.437, 3.42215,
             3.41025, 3.3405, 3.21557, 3.09625, 2.8663, 2.655]
             
#================================================================================================#             

calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
settleDate = calendar.advance(valDate, ql.Period(2, ql.Days), ql.Following, False)              
             
#================================= Formatting and Calculation ===================================#

discountFactors = [1] # discount factor at time 0
discountTimes = [0]

# to store discount factors for contracts that start on the settle date
settleDFs = []
settleTimes = []

# to store the information needed by the optimization function
sofrSwapTimes = []
sofrSwapRates = []
sofrSwapDates = []
sofrSwapStub = []
for i in range(len(sofrRates)):
    tenor = sofrTenors[i]
    rate = sofrRates[i] / 100
    stub = False
    if tenor=="ON" or tenor == "1DY":
        endDate = calendar.advance(valDate, ql.Period(1, ql.Days), ql.Following, False)
        accFrac = ql.Actual360().yearFraction(valDate, endDate)
        t = ql.Actual365Fixed().yearFraction(valDate, endDate)
        df = 1 / (1 + rate * accFrac)
        discountFactors.append(df)
        discountTimes.append(t)
    if "W" in tenor:
        n = int(tenor[:-2])
        endDate = calendar.advance(settleDate, ql.Period(n, ql.Weeks), ql.ModifiedFollowing, False)
        accFrac = ql.Actual360().yearFraction(settleDate, endDate)
        t = ql.Actual365Fixed().yearFraction(valDate, endDate)
        df = 1 / (1 + rate * accFrac)
        settleDFs.append(df)
        settleTimes.append(t)
    if "M" in tenor:
        n = int(tenor[:-2])
        endDate = calendar.advance(settleDate, ql.Period(n, ql.Months), ql.ModifiedFollowing, False)
        t = ql.Actual365Fixed().yearFraction(valDate, endDate)
        if n <= 12: 
            accFrac = ql.Actual360().yearFraction(settleDate, endDate)
            df = 1 / (1 + rate * accFrac)
            settleDFs.append(df)
            settleTimes.append(t)
        else:
            sofrSwapTimes.append(t)
            sofrSwapRates.append(rate)
            sofrSwapDates.append(endDate)
            if n % 12 != 0: stub = True
            sofrSwapStub.append(stub)
            
    if "Y" in tenor:
        n = int(tenor[:-2])
        endDate = calendar.advance(settleDate, ql.Period(n, ql.Years), ql.ModifiedFollowing, False)
        t = ql.Actual365Fixed().yearFraction(valDate, endDate)
        if n == 1: 
            accFrac = ql.Actual360().yearFraction(settleDate, endDate)
            df = 1 / (1 + rate * accFrac)
            settleDFs.append(df)
            settleTimes.append(t)
        else:
            sofrSwapTimes.append(t)
            sofrSwapRates.append(rate)
            sofrSwapDates.append(endDate)
            sofrSwapStub.append(stub)
    
# At this point all discount factors for sofr swaps less than a year have been found from 
# the settle date and need to be further discounted to the valuation date

# the base discount factor is found by interpolation, consistent with our choice (linear on
# the log of discount factors) using the ON discount factor and the first pillar swap discount 
# factor that starts on the settle date (1W in this case)
# this turns out to be easily found by simple algebra

t_s = ql.Actual365Fixed().yearFraction(valDate, settleDate)
m = ( t_s - discountTimes[-1] ) / ( settleTimes[0] - discountTimes[-1] )
y1 = discountFactors[-1]
y2 = settleDFs[0]
log_base = ( np.log(y1) * (1 - m) + (np.log(y2) * m) ) / (1 - m)
base_df = np.exp( log_base )

knownSofrTimes = [t_s]
knownSofrZeros = [np.log(1 / base_df) / t_s]

for i in range(len(settleDFs)):
    df = settleDFs[i] * base_df
    zero = np.log(1 / df) / settleTimes[i]  
    knownSofrZeros.append(zero)
    knownSofrTimes.append(settleTimes[i])
    discountFactors.append(df)
    discountTimes.append(settleTimes[i]) 


# ===================== Run the multi-variate Newton Raphson Optimization =======================#

tol = 1e-15
max_iter = 12
initialZeros = np.array([0.02] * len(sofrSwapRates)) # guess zero rates to start
optimizedZeros = zeroCurve(sofrSwapTimes, initialZeros)

#================================================================================================#

#================= Calculate the discount factors from the optimized zero rates =================#
for i in range(len(optimizedZeros)):
    z = optimizedZeros[i]
    t = sofrSwapTimes[i]
    df = np.exp(-t * z)
    discountFactors.append(df) 
    discountTimes.append(t) # Now we have the entire term structure of discount factors
            
            
