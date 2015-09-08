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
    path = '../alr3data/forbes.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Temp)
    #u = 1./(255.37+9.*x/5.)
    y = np.array(df.Lpres)
    data = np.array([x, y])
    simple_linear_regression(data, names = ['Temp', 'Lpres'], figname = 'p2.2.png')
    one = ols(y=df['Lpres'], x=df['Temp'])
    print one
    '''
    
    '''
    path = '../alr3data/forbes.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Temp)
    #u = 1./(255.37+9.*x/5.)
    y = np.array(df.Lpres)
    data = np.array([x, y])
    simple_linear_regression(data, names = ['Temp', 'Lpres'], figname = 'p2.2.png')
    one = ols(y=df['Lpres'], x=df['Temp'])
    print one
    
    x = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, x)
    results = model.fit()
    print results.summary()
    '''
    
    path = '../alr3data/snake.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.X)    
    y = np.array(df.Y)
    data = np.array([x, y])
    simple_linear_regression(data, names = ['X', 'Y'], figname = 'p2.7.png')
    one = ols(y=df['X'], x=df['Y'])
    print one
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    #x = sm.add_constant(x, prepend=False)
    model = sm.OLS(y, x)
    results = model.fit()
    print results.summary()
    xmax = x.max()
    xmin = x.min()
    xfit = np.arange(xmin*0.99-20.0, xmax*1.01, (xmax-xmin)/5.)
    yfit = xfit*slope + intercept
    yfit2 = xfit*results.params[0]

    figure(2)
    plot(x, y, 'ko', mfc='none')
    plot(xfit, yfit, 'b--', label = 'Intercept')    
    plot(xfit, yfit2, 'b-', label = 'No intercept')    
    #xticks(np.arange(xmin*0.99, xmax*1.01, (xmax-xmin)/2.))
    #xlim(0.0, 60.0)
    #ylim(0.0, 60.0)
    axis([0., 60., 0., 60.])
    axis('scaled')
    legend(loc='best')
    grid(True)
    xlabel('xx')
    ylabel('yy')
    savefig('p2.7_2.png')
    
    yHat = x * results.params[0]
    Rsquare = np.sum(yHat**2)/np.sum(y**2)
    print 'Rsquare: ', Rsquare
    
    '''
def simple_linear_regression(data, names = ['x', 'y'], figname = 'newfig.png'):
    """ Computes the coefficients and goodness of fit for simple linear regression"""
    numvars, numdata = data.shape
    [x, y] = data
    xbar = np.average(x)
    ybar = np.average(y)    
    SXX = np.dot((x - xbar), x)
    SXY = np.dot((x - xbar), (y - ybar))
    SYY = np.dot((y - ybar), y)    
    beta1 = SXY/SXX
    beta0 = ybar - xbar * beta1
    
    yHat = x*beta1 + beta0
    Residuals = y - yHat
    
    mq = stats.mstats.mquantiles(Residuals, [0., 0.25, 0.5, 0.75, 1.])
    print 'Residuals:'
    print '\tMin\t1Q\tMedian\t3Q\tMax'
    print '\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (mq[0], mq[1], mq[2], mq[3], mq[4])
        
    RSS = np.sum((y-yHat)**2)
    sigmaHatSq = RSS/(numdata-2.0)    
    beta1Se = np.sqrt(sigmaHatSq/SXX)
    beta0Se = np.sqrt(sigmaHatSq*(1.0/numdata + xbar**2/SXX))
    beta0tt = beta0/beta0Se
    beta0pval = stats.t.sf(np.abs(beta0tt), numdata-1)*2
    beta1tt = beta1/beta1Se
    beta1pval = stats.t.sf(np.abs(beta1tt), numdata-1)*2
    
    print 'Coefficients:'
    print '\tEst.\tStdErr\ttt\tPr(>|tt|)'
    print 'beta0\t%.3f\t%.3f\t%.3f\t%.3f\n' % (beta0, beta0Se, beta0tt, beta0pval)
    print 'beta1\t%.3f\t%.3f\t%.3f\t%.3f\n' % (beta1, beta1Se, beta1tt, beta1pval)
    
    
    resSe = stats.sem(Residuals)
    Rsquare = 1.0 - RSS/SYY
    print 'Residual standard error: %.3f on %d degrees of freedom' % (resSe, numdata-2)
    print 'R-squared: %.3f' % Rsquare
    
    SSreg = np.sum((yHat-ybar)**2)    
    MSreg = SSreg/1
    MSR = RSS/(numdata-2)
    F = MSreg/MSR
    Fpval = stats.f.sf(F, 1, numdata-2)
    print 'F-statistic: %.3f on %d and %d DF, p-val: %.3f' % (F, 1, numdata-2, Fpval)
    
    xmax = x.max()
    xmin = x.min()
    xfit = np.arange(xmin*0.99-15., xmax*1.01, (xmax-xmin)/5.)
    yfit = xfit*beta1 + beta0

    figure
    plot(x, y, 'ko', mfc='none')
    plot(xfit, yfit, 'b-')    
    #xticks(np.arange(xmin*0.99, xmax*1.01, (xmax-xmin)/2.))
    xlim(0.0, 60.0)
    ylim(0.0, 60.0)
    axis('scaled')
    grid(True)
    xlabel(names[0])
    ylabel(names[1])   
    savefig(figname)
    
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
    