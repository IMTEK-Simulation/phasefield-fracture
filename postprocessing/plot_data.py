import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
#sys.path.append("/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/combined/src")
import get_stats

def plot_stat(datalist, xkey, ykey, xlabel, ylabel,plotname):   
    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.15, 0.78, 0.78])
    for data in datalist:
        ax.plot(data[xkey],data[ykey],
             linestyle=data["linestyle"],marker=data["marker"],label=data["label"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(plotname)
    plt.clf()


mpl.rc('lines', markerfacecolor='None', markersize=4)

keynames = ["strain", "stress", "max_overforce", "max_yy_stress","max_delta_phi","delta_phi"]


dir = '../../muspectre_misc'
maxind = 1100
data = get_stats.get_data(dir, keynames, maxind, output_dup=True)
data["label"] = "Alt. Min."
data["iteration"] = np.array(range(0,data["strain"].size))
data["marker"] = None
data["linestyle"] = '-'
datalist = [data]

ykeys = ["stress", "max_overforce", "max_yy_stress","max_delta_phi","delta_phi"]
xkeys = ["strain", "iteration", "iteration","iteration","iteration"]
xlabels = [r"$\bar \varepsilon_{yy}$", "Iteration", "Iteration","Iteration","Iteration"]
ylabels = ["Average yy stress", "Maximum overforce", "Maximum yy stress",
        r"Maximum $\Delta \phi$",r"Integrated $\Delta \phi$"]
plotnames = ["stress.svg", "max_overforce.svg",
    "max_yy_stress.svg","max_delta_phi.svg","delta_phi.svg"]
        
for i in range(0,len(xkeys)):
    plot_stat(datalist, xkeys[i], ykeys[i], xlabels[i], ylabels[i],plotnames[i])

