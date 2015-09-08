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
    # p1.1
    path = '../alr3data/wblake.txt'
    df = pd.read_csv(path, sep=' ')

    x = np.array(df.Age)
    y = np.array(df.Length)
    
    xu = np.unique(x)
    yavg = np.array([np.average(y[np.where(x == i)]) for i in xu])
    ystd = np.array([np.std(y[np.where(x == i)]) for i in xu])

    figure(1)    
    plot(x, y, 'k.', mfc='none', label = 'raw')
    linestyle = {"linestyle":"-", "linewidth":4, "markeredgewidth":5, "elinewidth":4, "capsize":4}
    errorbar(xu, yavg, ystd, marker = 'o', ecolor='r', color = 'r', **linestyle)
    #errorbar(xu, yavg, ystd, fmt='o', ecolor='g', capthick=4)
    legend(loc='best')
    xlabel('Age')
    ylabel('Length')
    xticks(np.arange(0, 12, 2))
    savefig('p1.1.png')
    '''
    
    '''
    # p1.2
    path = '../alr3data/Mitchell.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Month)
    y = np.array(df.Temp)

    figure(2)
    figure(figsize = (20, 1.5))
    plot(x, y, 'ko', mfc='none')
    #linestyle = {"linestyle":"-", "linewidth":4, "markeredgewidth":5, "elinewidth":4, "capsize":4}
    #errorbar(xu, yavg, ystd, marker = 'o', ecolor='r', color = 'r', **linestyle)
    #errorbar(xu, yavg, ystd, fmt='o', ecolor='g', capthick=4)
    #legend(loc='best')    
    xlabel('Month')
    ylabel('Temp')
    xticks(np.arange(0, 250, 50))
    yticks(np.arange(-5, 35, 20))
    xlim(-10, 210)
    tight_layout()
    grid(True)
    savefig('p1.2.png')
    '''
    
    '''
    # p1.3
    path = '../alr3data/UN1.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.PPgdp)
    y = np.array(df.Fertility)

    figure(3)
    figure(figsize = (5, 3))
    plot(x, y, 'ko', mfc='none')
    xlabel('ppgdp')
    ylabel('fertility')
    xticks(np.arange(0, 6e4, 1e4))
    yticks(np.arange(1, 9, 2))
    xlim(-1e3, 5e4)
    ylim(0.8, 8.2)
    ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    tight_layout()
    grid(True)
    savefig('p1.3.png')
    
    figure(4)
    figure(figsize = (5, 3))
    plot(np.log10(x), np.log10(y), 'ko', mfc='none')
    xlabel('log(ppgdp)')
    ylabel('log(fertility)')
    
    xticks(np.arange(1.5, 5.0, 1))
    yticks(np.arange(0, 1, 0.2))
    xlim(1.5, 5)
    ylim(-0.1, 1)
    ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    tight_layout()
    grid(True)
    savefig('p1.3_2.png')
    '''
    
    '''
     # p1.4
    path = '../alr3data/oldfaith.txt'
    df = pd.read_csv(path, sep=' ')
    x = np.array(df.Duration)
    y = np.array(df.Interval)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    xfit = np.arange(x.min()*0.8, x.max()*1.2, 1.0)
    yfit = xfit*slope + intercept
    
    figure(5)
    figure(figsize = (6, 4))
    plot(x, y, 'k.')
    plot(xfit, yfit, 'b-')
    xlabel('Duration')
    ylabel('Interval')
    #xticks(np.arange(0, 6e4, 1e4))
    #yticks(np.arange(1, 9, 2))
    #xlim(-1e3, 5e4)
    #ylim(0.8, 8.2)
    #ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    tight_layout()
    grid(True)
    savefig('p1.4.png')
    '''
    
    '''
    # p1.5
    path = '../alr3data/water.txt'
    df = pd.read_csv(path, sep=' ')
    data = np.array([df.Year, df.APMAM, df.APSAB, df.APSLAKE, df.OPBPC, df.OPRC, df.OPSLAKE, df.BSAAM])
    fig = scatterplot_matrix(data, ['Year', 'APMAM', 'APSAB', 'APSLAKE', 'OPBPC', 'OPRC', 'OPSLAKE', 'BSAAM'], linestyle='none', marker='.', color='black')
    #fig.suptitle('Simple Scatterplot Matrix')
    savefig('p1.5.png')
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
    