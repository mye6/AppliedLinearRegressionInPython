import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # <-- THIS MAKES IT FAST!
#from pylab import *
from matplotlib.pyplot import *
from scipy import stats

import itertools
from matplotlib.backends.backend_pdf import PdfPages

def main():
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
    