import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter



class OIS:
    def __init__(self, schedule, accrualYearFracs):
        self.schedule = schedule
        self.accrualYearFracs = accrualYearFracs
        
    def modelRate(self, discountDict):
        scheduleDF = np.array([discountDict[d] for d in self.schedule])
        rate = (scheduleDF[0] - scheduleDF[-1]) / np.sum(self.accrualYearFracs * scheduleDF[1:])
        return rate


class OISHandler:
    def __init__(self, 
                 calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond), 
                 dayCounter=ql.Actual360(), 
                 swapFreq=ql.Period('1Y')):
        self.calendar = calendar
        self.dayCounter = dayCounter
        self.swapFreq = swapFreq
        self.units = {}
        
        
    def add_unit(self, tenor, rate, settlementDays=2):
        if settlementDays < 0:
            raise ValueError("Settlement days cannot be negative")
        if not isinstance(tenor, str):
            raise ValueError("Tenor must be specified as a string")
        self.units[tenor] = {'Rate' : rate, 'SettleDays' : settlementDays}
        
    def GenerateInstruments(self, curveDate):
        instruments = []
        all_dates = {curveDate}
        bdc = ql.ModifiedFollowing 
        dateRule = ql.DateGeneration.Backward
        startDateMap = {0 : curveDate}
        maturityDates = []
        tenors = []
        for tenor, data in self.units.items():
            settleDays = data['SettleDays']
            if settleDays not in startDateMap:
                startDateMap[settleDays] = self.calendar.advance(curveDate, ql.Period(settleDays, ql.Days))
            startDate = startDateMap[settleDays]
            period = ql.Period(tenor)
            endDate = self.calendar.advance(startDate, period, ql.Unadjusted)
            schedule = ql.Schedule(startDate, endDate, self.swapFreq, self.calendar, bdc, bdc, dateRule, False)
            data['Schedule'] = schedule
            tenors.append(tenor)
            maturityDates.append(schedule[-1])
        _, unique_inds = np.unique(maturityDates, return_index=True) # sort and drop duplicates
        curveTenors = []
        curveParRates = []
        maturityDates = []
        for idx in unique_inds:
            tenor = tenors[idx]
            mkt_rate = self.units[tenor]['Rate']
            schedule = self.units[tenor]['Schedule']
            accrualYearFracs = np.array([self.dayCounter.yearFraction(schedule[i-1], schedule[i]) for i in range(1, len(schedule))])
            all_dates.update(schedule)
            instruments.append( OIS(schedule, accrualYearFracs) )
            curveTenors.append(tenor)
            curveParRates.append(mkt_rate)
            maturityDates.append(schedule[-1])
        all_dates = sorted(all_dates)
        return instruments, curveTenors, curveParRates, maturityDates, all_dates

def PricerError(instruments, targetRates, discountDict):
    n = len(instruments)
    errors = np.zeros(n)
    for i in range(n):
        errors[i] = instruments[i].modelRate(discountDict) - targetRates[i]
    return errors



def BuildPwcFwdCurve(instruments, anchorTimes, rates, all_dates, all_times, all_idx,
          termStructureDayCounter,
          init_guess=None,
          tol=1e-12,
          max_iter=9,
          shocks={}):
    """ Global Newton Scheme for Piecewise Constant Continuously Compounded Forward Rate Interpolation """
        
    all_dfs = np.ones(len(all_dates))
    n = len(instruments)
    anchorTimesDelta = np.diff(anchorTimes)
    anchorDFs = np.ones(len(anchorTimes))

    def Pricer(fwdRates):
        fwdRates = np.asarray(fwdRates, dtype=float)
        anchorDFs[1:] = np.exp(np.cumsum(-fwdRates * anchorTimesDelta))
        all_dfs[1:] = anchorDFs[all_idx - 1] * np.exp(-fwdRates[all_idx - 1] * (all_times[1:] - anchorTimes[all_idx - 1]))
        discountDict = {k:v for k,v in zip(all_dates, all_dfs)}
        errors = PricerError(instruments, rates, discountDict)
        return errors
    
    def getJacobian(fwdRates):
        ''' Returns the jacobian matrix of first-order sensitivities to changes from forward rates '''
        eps = 1e-6
        jacobian = np.zeros((n, n))
        values = Pricer(fwdRates)
        fwdRates_up = np.copy(fwdRates)
        for i in range(n):
            fwdRates_up[i] = fwdRates[i] + eps
            values_up = Pricer(fwdRates_up)
            fwdRates_up[i] = fwdRates[i]
            dv = (values_up - values) / eps
            jacobian[:,i] = dv
        return jacobian
    
    def globalObjFunc(fwdRates):
        error = 10e15
        j = 0
        while error > tol and j <= max_iter:
            j += 1
            values = Pricer(fwdRates)
            jacobian = getJacobian(fwdRates)
            inv_jacobian = np.linalg.inv(jacobian)
            err_arr = -np.dot(inv_jacobian, values)
            fwdRates += err_arr
            error = np.linalg.norm(err_arr)
        return fwdRates
    
    if init_guess is not None:
        init_guess = np.atleast_1d(init_guess)
        if len(init_guess) == 1 and n > 1:
            # Broadcast scalar to match instruments
            init_guess = np.full(n, init_guess[0])
        elif len(init_guess) != n:
            raise ValueError("Size of initial guess must match size of instruments")
    else:
        init_guess = np.full(n, 0.035)
    
    pwcFwds = globalObjFunc(init_guess)
    return pwcFwds
             

def interp_indices(arr, val):
    """ Get lower and upper interpolation bounds """
    us = np.searchsorted(arr, val)
    us = np.clip(us, 1, len(arr) - 1)
    ds = us - 1
    return ds, us
    
    
def ModifyPWCFwds(pwcFwds, curveTimes):
    """ 
    Create a smooth forward curve by linearly interpolating 
    flat-segment midpoints of piecewise constant forwards and 
    adjusting boundary values to align with those interpolations.
    """
    term1 = pwcFwds[:-1] * (curveTimes[2:] - curveTimes[1:-1]) / (curveTimes[2:] - curveTimes[:-2])
    term2 = pwcFwds[1:] * (curveTimes[1:-1] - curveTimes[:-2]) / (curveTimes[2:] - curveTimes[:-2])
    fwds = term1 + term2
    f0 = pwcFwds[0] - 0.5 * (fwds[0] - pwcFwds[0])
    fm = pwcFwds[-1] + 0.5 * (pwcFwds[-1] - fwds[-1])
    fwds = np.concatenate(([f0], fwds, [fm]))
    return fwds

    
def instantaneousForward(t, curveTimes, pwcFwds, modifiedFwds):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    fwds = np.zeros_like(t)
    flag1 = t < curveTimes[0]
    fwds[flag1] = modifiedFwds[0]
    flag2 = t > curveTimes[-1]
    fwds[flag2] = modifiedFwds[-1]
    flag = ~(flag1 | flag2)
    ds, us = interp_indices(curveTimes, t[flag])
    y = (t[flag] - curveTimes[ds]) / (curveTimes[us] - curveTimes[ds])
    fj1, fj = modifiedFwds[ds], modifiedFwds[us]
    fwds[flag] = fj1 * (1-y) + fj * y - 3 * (fj1 + fj - 2 * pwcFwds[ds]) * y * (1-y)
    return fwds[0] if len(fwds)==1 else fwds


def GetDiscFactor(t, curveTimes, pwcFwds, modifiedFwds):
    """ Discount Factor by integrating the instantaneous forward rate curve """
    t = np.asarray(t, dtype=float).reshape(-1)
    areas = np.append(0.0, np.cumsum(pwcFwds * np.diff(curveTimes)))
    ds, us = interp_indices(curveTimes, t)
    tj1, tj = curveTimes[ds], curveTimes[us]
    fj1, fj = modifiedFwds[ds], modifiedFwds[us]
    Aj = pwcFwds[ds]
    y = (t - tj1) / (tj - tj1)
    term1 = fj1 * y
    term2 = (-2 * fj1 - fj + 3 * Aj) * y**2
    term3 = (fj1 + fj - 2 * Aj) * y**3
    integral = areas[ds] + (term1 + term2 + term3) * (tj - tj1)
    df = np.exp(-integral)
    return df


def BuildCurve(curveDate, handler, 
          termStructureDayCounter = ql.Actual365Fixed(),
          pwc_tol=1e-12,
          pwc_max_iter=8,
          tol=1e-12,
          max_iter=9,
          shocks={}):
    
    """ Build discount curve using the Smart Quadratic interpolation """
    
    # include pwc details
    instruments, curveTenors, curveParRates, maturityDates, all_dates = handler.GenerateInstruments(curveDate)
    all_times = np.array([termStructureDayCounter.yearFraction(curveDate, date) for date in all_dates], dtype=float)
    curveDates = [curveDate] + maturityDates
    curveTimes = np.array([termStructureDayCounter.yearFraction(curveDate, date) for date in curveDates], dtype=float)
    all_idx = np.searchsorted(curveDates, all_dates[1:])
    
    targetRates = []
    shockSizes = []
    for i in range(len(instruments)):
        shock_val = shocks.get(curveTenors[i], 0.0)
        shockSizes.append(shock_val)
        targetRates.append(curveParRates[i] + shock_val)
    
    init_guess = None
    rates_iter = np.copy(targetRates)
    error = 10e15
    k = 0
    while error > tol and k <= max_iter:
        k += 1
        pwcFwds = BuildPwcFwdCurve(instruments, curveTimes, rates_iter, all_dates, all_times, all_idx, 
                                   termStructureDayCounter, init_guess=init_guess, tol=pwc_tol, max_iter=pwc_max_iter)
        init_guess = pwcFwds
        modifiedFwds = ModifyPWCFwds(pwcFwds, curveTimes)
        all_dfs = GetDiscFactor(all_times, curveTimes, pwcFwds, modifiedFwds)
        discountDict = {k:v for k,v in zip(all_dates, all_dfs)}
        errors = PricerError(instruments, targetRates, discountDict)
        rates_iter -= errors
        error = np.linalg.norm(errors)
        
    curveDFs = [discountDict[d] for d in curveDates]
    curveData = {
        'Tenor': curveTenors,
        'MaturityDate': [d.to_date() for d in curveDates[1:]],
        'MarketRate': curveParRates,
        'ShockSize': shockSizes,
        'DiscountFactor': curveDFs[1:], 
        'ZeroRate': np.log(curveDFs[1:]) / -curveTimes[1:],
        }
    return curveData, curveTimes, pwcFwds, modifiedFwds


curveDate = ql.Date(7, 7, 2025)
sofrTenors = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '12M', '18M', '2Y',
                '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '12Y', '15Y', '20Y', '25Y', '30Y', '40Y', '50Y']

sofrRates = [0.0432097, 0.0432268, 0.04329, 0.0433754, 0.0435234, 0.043307, 0.042977, 0.0426493, 0.042162, 0.041746, 
             0.041391, 0.0409876, 0.0405835, 0.0402319, 0.03986, 0.0374984, 0.0363314, 0.035433, 0.03538, 0.0356735, 
             0.0361632, 0.0366985, 0.0372405, 0.037765, 0.0382685, 0.039193, 0.040227, 0.0409975, 0.0409047, 0.0404535,
             0.039125, 0.0377548]
        
# Add the swap units to the handler
handler = OISHandler()
for tenor,rate in zip(sofrTenors, sofrRates):
    handler.add_unit(tenor, rate) 


curveData, curveTimes, pwcFwds, modifiedFwds = BuildCurve(curveDate, handler)
curveData = pd.DataFrame(curveData)
discountCurve = lambda t : GetDiscFactor(t, curveTimes, pwcFwds, modifiedFwds)

# Plot Daily Forward Rates
t = np.arange(curveTimes[0], curveTimes[-1], 1/365)
df = discountCurve(t)
fwds = 360 * (df[:-1] / df[1:] - 1.0) # Actual360 Basis
plt.figure(figsize=(10, 5))
plt.plot(t[1:], fwds)
plt.xlabel('Time (Years)')
plt.ylabel('Rate')
plt.title('Daily SOFR Forward Rates (' + curveDate.to_date().strftime('%m-%d-%Y') + ')')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.grid(True)
plt.tight_layout()
plt.show()



# Plot Instantaneous Forward Rates
inst_fwds = (360 / 365) * instantaneousForward(t, curveTimes, pwcFwds, modifiedFwds) # Actual360 Basis
plt.figure(figsize=(10, 5))
plt.plot(t, inst_fwds)
plt.xlabel('Time (Years)')
plt.ylabel('Rate')
plt.title('Instantaneous SOFR Forward Rates (' + curveDate.to_date().strftime('%m-%d-%Y') + ')')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.grid(True)
plt.tight_layout()
plt.show()



