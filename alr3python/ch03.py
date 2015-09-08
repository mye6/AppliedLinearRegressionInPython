import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # <-- THIS MAKES IT FAST!
from matplotlib.pyplot import *
from scipy import stats
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from pandas.stats.api import ols
import statsmodels.api as sm

def main():
    '''
    # p47
    path = '../alr3data/UN2.txt'
    df = pd.read_csv(path, sep=' ')
    
    figure(1)
    subplots(2,2,figsize=(10,10))
    x = np.array(df.logPPgdp)
    y = np.array(df.logFertility)
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]
    eLogFert = y - x*results.params[0] + results.params[1]
    subplot(2,2,1)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('log(PPgdp)')
    ylabel('log(Fertility)')
    print 'logFertility vs. logPPgdp: ', results.params
    
    x = np.array(df.Purban)
    y = np.array(df.logFertility)
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]    
    subplot(2,2,2)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('Percent urban')
    ylabel('log(Fertility)')
    print 'logFertility vs. Purban: ', results.params
    
    x = np.array(df.logPPgdp)
    y = np.array(df.Purban)
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]
    ePurban = y-x*results.params[0] + results.params[1]
    subplot(2,2,3)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('log(PPgdp)')
    ylabel('Percent urban')
    print 'logPPgdp vs. Purban: ', results.params
    
    x = ePurban
    y = eLogFert
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]
    subplot(2,2,4)
    grid(True)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('ePurban')
    ylabel('eLogFert')        
    #tight_layout()
    savefig('ch03_p48.png')    
    print 'eLogFert vs. ePurban: ', results.params
    
    x = np.array([df.logPPgdp, df.Purban])
    y = np.array(df.logFertility)
    X = sm.add_constant(np.transpose(x), prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    print 'logFertility vs. logPPgdp, Purban: ', results.params
    print results.summary()
    '''
    
    '''
    mu, sigma = 2.0, 1.5 # mean and standard deviation
    logWidth = np.random.normal(mu, sigma, 5000)
    mu, sigma = 3.0, 1.0 # mean and standard deviation
    logLength = np.random.normal(mu, sigma, 5000)
    logArea = logWidth + logLength + np.random.normal(0.2, 0.5, 5000)
    
    figure(1)
    subplots(2,2,figsize=(10,10))
    x = logWidth
    y = logArea
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]
    eLogArea = y - x*results.params[0] + results.params[1]
    subplot(2,2,1)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('logWidth')
    ylabel('logArea')
    print 'logArea vs. logWidth: ', results.params, results.rsquared
    
    x = logLength
    y = logArea
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]    
    subplot(2,2,2)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('logLength')
    ylabel('logArea')
    print 'logArea vs. logLength: ', results.params, results.rsquared
    
    x = logWidth
    y = logLength
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]
    eLogLength = y-x*results.params[0] + results.params[1]
    subplot(2,2,3)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('logWidth')
    ylabel('logLength')
    print 'logLength vs. logWidth: ', results.params, results.rsquared
    
    x = eLogLength
    y = eLogArea
    X = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*results.params[0] + results.params[1]
    subplot(2,2,4)
    grid(True)
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('eLogLength')
    ylabel('eLogArea')        
    #tight_layout()
    savefig('ch03_p49.png')    
    print 'eLogArea vs. eLogLength: ', results.params, results.rsquared
    
    x = np.array([logWidth, logLength])
    y = logArea
    X = sm.add_constant(np.transpose(x), prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    print 'logArea vs. logWidth, logLength: ', results.params, results.rsquared
    '''
    
    '''
    # p51    
    mu, sigma = 2.0, 1.5 # mean and standard deviation
    x = np.random.normal(mu, sigma, 1000)
    y = 2.0 + x + 1.2*x**2 + 1.5*x**3 + np.random.normal(3., 0.5, 1000)
    x0 = np.array([x**3, x**2, x])
    X = sm.add_constant(np.transpose(x0), prepend=False)
    model = sm.OLS(y, X)
    results = model.fit()    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = results.params[0]*xfit**3+results.params[1]*xfit**2+results.params[2]*xfit+results.params[3]
    
    figure
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('x')
    ylabel('y')
    savefig('ch03_p52.png')
    print results.summary()
    '''
    
    # Page 53
    path = '../alr3data/fuel2001.txt'
    df = pd.read_csv(path, sep=' ')
    Tax = np.array(df.Tax)
    Dlic = 1000.0 * np.array(df.Drivers)/np.array(df.Pop)
    Income = np.array(df.Income)/1000.0
    logMiles = np.log2(np.array(df.Miles))
    Fuel = 1000.0 * np.array(df.FuelC) / np.array(df.Pop)
    #data = np.array([Tax, Dlic, Income, logMiles, Fuel])
    
    df['Dlic'] = Dlic
    df['Income'] = Income
    df['logMiles'] = logMiles
    df['Fuel'] = Fuel
    print df[['Tax', 'Dlic', 'Income', 'logMiles', 'Fuel']].describe().transpose()
    
    print '----------------------------------------------'
    print df[['Tax', 'Dlic', 'Income', 'logMiles', 'Fuel']].corr()
    
    print '----------------------------------------------'
    for colname in df.columns:
        print colname
        
    dfsel = df[['Tax', 'Dlic', 'Income', 'logMiles', 'Fuel']]
    y = np.array(dfsel)[:,4]
    X = np.array(dfsel)[:,0:4]
    for i in np.arange(X.shape[1]):
        X[:,i] -= np.average(X[:,i])
    CC = np.dot(X.transpose(), X)
    
    dat = np.array(dfsel)
    for i in np.arange(dat.shape[1]):
        dat[:,i] -= np.average(dat[:,i])
    C = np.dot(dat.transpose(), dat)/(dat.shape[0]-1.)
    
    '''
    # p24
    mu, sigma = 2.0, 1.5 # mean and standard deviation
    s = np.random.normal(mu, sigma, 10000)
    figure(1)
    count, bins, ignored = hist(s, 30, normed=True)
    plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2)), linewidth=2, color='r')
    savefig('ch02_p24.png')
    '''
    
    '''
    # p24
    mu, sigma = 2.0, 1.5 # mean and standard deviation
    x = np.random.normal(mu, sigma, 20)
    e = np.random.normal(0.0, 0.1, 20)
    y = 0.7 + 0.8 * x + e       
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    xfit = np.arange(x.min()*0.8, x.max()*1.2, 1.0)
    yfit = xfit*slope + intercept
    
    figure(2)
    plot(x, y, 'ko', mfc='none')
    plot(xfit, yfit, 'b-')
    xlabel('Predictor = X')
    ylabel('Response = Y')
    axis('scaled')
    savefig('ch02_p23.png')    
    
    print '***********************'
    print "slope:", slope
    print "intercept:", intercept
    print "r-squared:", r_value**2
    print "p_value:", p_value
    print "std_err:", std_err
    
    xbar = np.average(x)
    ybar = np.average(y)
    SXX = np.dot((x - xbar), x)
    SXY = np.dot((x - xbar), (y - ybar))
    SYY = np.dot((y - ybar), y)
    beta1 = SXY/SXX
    beta0 = ybar - xbar * beta1
    Rsquare = SXY**2/(SXX*SYY)
    yHat = x*beta1 + beta0
    n = len(x)
    sigmaHatSq = np.sum((y-yHat)**2)/(n-2.0)
    beta1Var = sigmaHatSq/SXX
    beta0Var = sigmaHatSq/SXX
    beta1Se = np.sqrt(beta1Var)
    beta0Se = np.sqrt(beta0Var)
    tt = beta1/beta1Se
    pVal = stats.t.sf(np.abs(tt), n-1)*2
    
    print '***********************'
    print 'beta1: ', beta1
    print 'beta0: ', beta0
    print 'Rsquare: ', Rsquare
    print 'pVal: ', pVal
    print 'beta1Se: ', beta1Se
    '''
    
    '''
    # p24
    path = '../alr3data/forbes.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Temp)
    y = np.array(df.Lpres)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*slope + intercept
    
    figure(3)
    plot(x, y, 'ko', mfc='none')
    plot(xfit, yfit, 'b-')
    xlabel('Temp')
    ylabel('Lpres')
    axis('scaled')
    savefig('ch02_p23_f2.png')    
    
    print '***********************'
    print "slope:", slope
    print "intercept:", intercept
    print "r-squared:", r_value**2
    print "p_value:", p_value
    print "std_err:", std_err
    
    n = len(x)
    xbar = np.average(x)
    ybar = np.average(y)
    
    SXX = np.dot((x - xbar), x)
    SXY = np.dot((x - xbar), (y - ybar))
    SYY = np.dot((y - ybar), y)
    
    beta1 = SXY/SXX
    beta0 = ybar - xbar * beta1
    yHat = x*beta1 + beta0
    
    RSS = np.sum((y-yHat)**2)
    SSreg = np.sum((yHat-ybar)**2)
    MSreg = SSreg/1
    MSR = RSS/(n-2)
    
    Rsquare = SXY**2/(SXX*SYY)
    F = MSreg/MSR
    
    Rsquare2 = 1.0 - RSS/SYY
    
    
    sigmaHatSq = RSS/(n-2.0)
    beta1Var = sigmaHatSq/SXX
    beta0Var = sigmaHatSq*(1.0/n + xbar**2/SXX)
    beta1Se = np.sqrt(beta1Var)
    beta0Se = np.sqrt(beta0Var)
    tt = beta1/beta1Se
    pVal = stats.t.sf(np.abs(tt), n-1)*2
    
    print '***********************'
    print 'beta1: ', beta1
    print 'beta0: ', beta0
    print 'Rsquare: ', Rsquare
    print 'Rsquare2: ', Rsquare2
    print 'pVal: ', pVal
    print 'beta1Se: ', beta1Se
    print 'beta0Se: ', beta0Se
    
    print '***********************'
    print 'xbar: ', xbar
    print 'ybar: ', ybar
    print 'SXX: ', SXX
    print 'SXY: ', SXY
    print 'SYY: ', SYY
    print 'RSS: ', RSS
    print 'SSreg: ', SSreg
    print 'SSreg+RSS: ', SSreg+RSS
    print 'MSreg: ', MSreg
    print 'MSR: ', MSR
    print 'F: ', F
    
    print 'Source\t\tdf\tSS\tMS\tF\tp-value'
    print 'Regression\t%.2f\t%.2f\t%.2f\t%.2f' % (1, SSreg, MSreg, F)
    print 'Residual\t%.2f\t%.2f\t%.2f' % (n-2, RSS, MSR)
    
    alpha = 0.1
    print 't(alpha/2, n-2): %.3f' % (stats.t.ppf(1.-alpha/2., n-2))
    f = stats.t.ppf(1.-alpha/2., n-2)
    print '%.3f <= beta0 <= %.3f' % (beta0 - f*beta0Se, beta0 + f*beta0Se)    
    '''
    
    '''
    path = '../alr3data/heights.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Mheight)
    y = np.array(df.Dheight)
    
    n = len(x)
    xbar = np.average(x)
    ybar = np.average(y)    
    SXX = np.dot((x - xbar), x)
    SXY = np.dot((x - xbar), (y - ybar))
    SYY = np.dot((y - ybar), y)
    
    beta1 = SXY/SXX
    beta0 = ybar - xbar * beta1
    yHat = x*beta1 + beta0    
    RSS = np.sum((y-yHat)**2)
    
    sigmaHatSq = RSS/(n-2.0)
    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*beta1 + beta0
    
    sigmaHatSq = RSS/(n-2.0)
    
    sepredY = np.sqrt(sigmaHatSq * (1. + 1./n + (xfit - xbar)**2/SXX))
    sefitY = np.sqrt(sigmaHatSq * (1./n + (xfit - xbar)**2/SXX))
    alpha = 0.05
    f = stats.t.ppf(1.-alpha/2., 15)
    print 'f: %.3f, n: %.f' % (f, n)
    predYLower = yfit - sepredY * f
    predYUpper = yfit + sepredY * f
    fitYLower = yfit - sefitY * f
    fitYUpper = yfit + sefitY * f
    
    
    figure(4)    
    plot(x, y, 'k.')
    plot(xfit, yfit, 'k--')
    plot(xfit, predYLower, 'k-')
    plot(xfit, predYUpper, 'k-')
    plot(xfit, fitYLower, 'b-.')
    plot(xfit, fitYUpper, 'b-.')
    xlabel('Mheight')
    ylabel('Dheight')
    axis('scaled')
    axis([54, 72, 54, 72])
    savefig('ch02_p36.png')
    '''
    
    '''
    path = '../alr3data/htwt.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Ht)
    y = np.array(df.Wt)
    
    n = len(x)
    xbar = np.average(x)
    ybar = np.average(y)    
    SXX = np.dot((x - xbar), x)
    SXY = np.dot((x - xbar), (y - ybar))
    SYY = np.dot((y - ybar), y)
    
    beta1 = SXY/SXX
    beta0 = ybar - xbar * beta1
    yHat = x*beta1 + beta0    
    RSS = np.sum((y-yHat)**2)
    
    sigmaHatSq = RSS/(n-2.0)
    
    xfit = np.arange(x.min()*0.9, x.max()*1.1, 1.0)
    yfit = xfit*beta1 + beta0
    
    sigmaHatSq = RSS/(n-2.0)
    
    sepredY = np.sqrt(sigmaHatSq * (1. + 1./n + (xfit - xbar)**2/SXX))
    sefitY = np.sqrt(sigmaHatSq * (1./n + (xfit - xbar)**2/SXX))
    alpha = 0.05
    f = stats.t.ppf(1.-alpha/2., 15)
    print 'f: %.3f, n: %.f' % (f, n)
    predYLower = yfit - sepredY * f
    predYUpper = yfit + sepredY * f
    fitYLower = yfit - sefitY * f
    fitYUpper = yfit + sefitY * f
    
    
    figure(4)    
    plot(x, y, 'k.')
    plot(xfit, yfit, 'k--')
    plot(xfit, predYLower, 'k-')
    plot(xfit, predYUpper, 'k-')
    plot(xfit, fitYLower, 'b-.')
    plot(xfit, fitYUpper, 'b-.')
    xlabel('Ht')
    ylabel('Wt')
    axis('scaled')
    #axis([54, 72, 54, 72])
    savefig('p2.1.png')
    
    
    n = len(x)
    xbar = np.average(x)
    ybar = np.average(y)
    
    SXX = np.dot((x - xbar), x)
    SXY = np.dot((x - xbar), (y - ybar))
    SYY = np.dot((y - ybar), y)
    
    beta1 = SXY/SXX
    beta0 = ybar - xbar * beta1
    yHat = x*beta1 + beta0
    
    RSS = np.sum((y-yHat)**2)
    SSreg = np.sum((yHat-ybar)**2)
    MSreg = SSreg/1
    MSR = RSS/(n-2)
    
    Rsquare = SXY**2/(SXX*SYY)
    F = MSreg/MSR
    
    Rsquare2 = 1.0 - RSS/SYY
    
    
    sigmaHatSq = RSS/(n-2.0)
    beta1Var = sigmaHatSq/SXX
    beta0Var = sigmaHatSq*(1.0/n + xbar**2/SXX)
    beta1Se = np.sqrt(beta1Var)
    beta0Se = np.sqrt(beta0Var)
    
    print '***********************'
    print 'beta1: ', beta1
    print 'beta0: ', beta0
    print 'Rsquare: ', Rsquare
    print 'Rsquare2: ', Rsquare2    
    print 'beta1Se: ', beta1Se
    print 'beta0Se: ', beta0Se
    '''
    
    
    
    
    
    '''
    path = '../alr3data/heights.txt'
    df = pd.read_csv(path, sep=' ')
    figure(1)
    plot(np.array(df.Mheight), np.array(df.Dheight), 'k.')
    xlabel('Mheight')
    ylabel('Dheight')
    axis('scaled')
    axis([55, 75, 55, 75])
    savefig('ch01_p3.png')

    df2 = df[((df.Mheight >= 57.5) & (df.Mheight < 58.5)) | ((df.Mheight >= 63.5) & (df.Mheight < 64.5)) | ((df.Mheight >= 67.5) & (df.Mheight < 68.5))]

    figure(2)
    plot(np.array(df2.Mheight), np.array(df2.Dheight), 'k.')
    xlabel('Mheight')
    ylabel('Dheight')
    axis('scaled')
    axis([55, 75, 55, 75])
    savefig('ch01_p4.png')
    '''


    '''
    path = '../alr3data/forbes.txt'
    df = pd.read_csv(path, sep=' ')

    x = np.array(df.Temp)
    y = np.array(df.Pressure)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    yfit = x*slope + intercept
    res = y - yfit

    figure(3)
    subplot(2, 1, 1)
    plot(x, y, 'ko', mfc='none', label = 'raw')
    plot(x, yfit, 'b-', label = 'fit')
    legend(loc='best')
    xlabel('Temperature')
    ylabel('Pressure')
    #axis('scaled')
    grid(True)
    #axis([190, 210, 20, 32])
    xticks(np.arange(195, 215, 5))
    #yticks(np.arange(18, 32, 2))

    subplot(2, 1, 2)
    plot(x, res, 'ko', mfc='none', label = 'raw')
    xlabel('Temperature')
    ylabel('Residuals')
    #axis('scaled')
    grid(True)
    #axis([190, 210, 20, 30])
    xticks(np.arange(195, 215, 5))
    #yticks(np.arange(18, 32, 2))
    tight_layout()
    savefig('ch01_p5.png')

    print '*******../alr3data/forbes.txt*******'
    print '******Pressure vs. Temperature******'
    print "slope:", slope
    print "intercept:", intercept
    print "r-squared:", r_value**2
    print "p_value:", p_value
    print "std_err:", std_err


    x = np.array(df.Temp)
    y = np.array(df.Lpres)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    yfit = x*slope + intercept
    res = y - yfit

    figure(4)
    subplot(2, 1, 1)
    plot(x, y, 'ko', mfc='none', label = 'raw')
    plot(x, yfit, 'b-', label = 'fit')
    legend(loc='best')
    xlabel('Temperature')
    ylabel('log(pressure)')
    #axis('scaled')
    grid(True)
    #axis([190, 210, 20, 32])
    xticks(np.arange(195, 215, 5))
    #yticks(np.arange(18, 32, 2))

    subplot(2, 1, 2)
    plot(x, res, 'ko', mfc='none', label = 'raw')
    xlabel('Temperature')
    ylabel('Residuals')
    #axis('scaled')
    grid(True)
    #axis([190, 210, 20, 30])
    xticks(np.arange(195, 215, 5))
    #yticks(np.arange(18, 32, 2))
    tight_layout()
    savefig('ch01_p6.png')

    print '*******../alr3data/forbes.txt*******'
    print '***log(pressure) vs. Temperature****'
    print "slope:", slope
    print "intercept:", intercept
    print "r-squared:", r_value**2
    print "p_value:", p_value
    print "std_err:", std_err
    '''

    '''
    path = '../alr3data/wblake.txt'
    df = pd.read_csv(path, sep=' ')

    x = np.array(df.Age)
    y = np.array(df.Length)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    xfit = np.arange(x.min()*0.8, x.max()*1.2, 1.0)
    yfit = xfit*slope + intercept
    res = y - (x*slope+intercept)

    xu = np.unique(x)
    #y[np.where(x == 1)]
    #np.average(y[np.where(x == 1)])
    yav = np.array([np.average(y[np.where(x == i)]) for i in xu])

    figure(5)
    subplot(2, 1, 1)
    plot(x, y, 'ko', mfc='none', label = 'raw')
    plot(xfit, yfit, 'b-', label = 'fit')
    plot(xu, yav, 'r*-', label = 'average')
    legend(loc='best')
    xlabel('Age')
    ylabel('Length')
    xticks(np.arange(0, 12, 2))

    subplot(2, 1, 2)
    plot(x, res, 'ko', mfc='none', label = 'raw')
    xlabel('Age')
    ylabel('Residuals')
    grid(True)
    xticks(np.arange(0, 12, 2))
    tight_layout()
    savefig('ch01_p7.png')

    print '*******../alr3data/wblake.txt*******'
    print '**********Age vs. Length************'
    print "slope:", slope
    print "intercept:", intercept
    print "r-squared:", r_value**2
    print "p_value:", p_value
    print "std_err:", std_err
    '''

    '''
    path = '../alr3data/ftcollinssnow.txt'
    df = pd.read_csv(path, sep=' ')

    x = np.array(df.Early)
    y = np.array(df.Late)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    xfit = np.arange(x.min()*0.8, x.max()*1.2, 1.0)
    yfit = xfit*slope + intercept

    xav = np.arange(0, 70, 1)
    yav = np.ones(len(xav))*np.average(y)

    figure(6)
    plot(x, y, 'ko', mfc='none', label = 'raw')
    plot(xav, yav, 'k-', label = 'average')
    plot(xfit, yfit, 'b--', label = 'fit')
    legend(loc='best')
    xlabel('Early')
    ylabel('Late')
    xticks(np.arange(0, 70, 10))
    savefig('ch01_p8.png')

    print '*******../alr3data/wblake.txt*******'
    print '**********Age vs. Length************'
    print "slope:", slope
    print "intercept:", intercept
    print "r-squared:", r_value**2
    print "p_value:", p_value
    print "std_err:", std_err
    '''

    '''
    path = '../alr3data/turkey.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.A)
    y = np.array(df.Gain)
    s = np.array(df.S)
    m = ['ko', 'k^', 'k+']
    l = ['1', '2', '3']

    figure(7)
    for i in range(0, 3):    
        plot(x[np.where(s==i+1)], y[np.where(s==i+1)], m[i], mfc='none', label = l[i])
    legend(loc='best')
    xlabel('Amount (percent of diet)')
    ylabel('Weight gain (g)')
    xlim(-0.02, 0.3)
    ylim(610, 800)
    xticks(np.arange(0, 0.3, 0.05))
    yticks(np.arange(650, 850, 50))
    savefig('ch01_p9.png')
    '''

    '''
    path = '../alr3data/heights.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Mheight)
    y = np.array(df.Dheight)

    x1 = np.arange(x.min()*0.8, x.max()*1.2, 1.0)
    y1 = x1

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    xfit = np.arange(x.min()*0.8, x.max()*1.2, 1.0)
    yfit = xfit*slope + intercept

    figure(8)
    plot(x, y, 'k.')
    plot(x1, y1, 'k--')
    plot(xfit, yfit, 'b-')
    xlabel('Mheight')
    ylabel('Dheight')
    axis('scaled')
    xlim(54, 73)
    ylim(54, 73)
    xticks(np.arange(55, 75, 5))
    yticks(np.arange(55, 75, 5))
    savefig('ch01_p10.png')
    '''
    
    '''
    path = '../alr3data/fuel2001.txt'
    df = pd.read_csv(path, sep=' ')
    Tax = np.array(df.Tax)
    Dlic = 1000.0 * np.array(df.Drivers)/np.array(df.Pop)
    Income = np.array(df.Income)/1000.0
    logMiles = np.log2(np.array(df.Miles))
    Fuel = 1000.0 * np.array(df.FuelC) / np.array(df.Pop)        
    data = np.array([Tax, Dlic, Income, logMiles, Fuel])
    fig = scatterplot_matrix(data, ['Tax', 'Dlic', 'Income', 'logMiles', 'Fuel'],
            linestyle='none', marker='o', color='black', mfc='none')
    #fig.suptitle('Simple Scatterplot Matrix')
    savefig('ch01_p16.png')
    '''
    
def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_major_locator(MaxNLocator(2))
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_major_locator(MaxNLocator(2))
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_major_locator(MaxNLocator(2))
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_major_locator(MaxNLocator(2))

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(j,i), (i,j)]:
            axes[x,y].plot(data[y], data[x], **kwargs)

    # Label the diagonal subplots.
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):           
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)
    
    return fig
    
main()    
    