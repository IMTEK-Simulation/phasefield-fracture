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

keynames = ["strain", "stress", "max_overforce", "max_yy_stress","max_delta_phi","delta_phi",
    "total_elastic_energy", "total_energy", "total_fracture_energy"]


maxind = 1100
fdir = '../../muspectre_misc'
data = get_stats.get_data(fdir, keynames, maxind, output_dup=True)
data["label"] = "Alternating Minimization"
data["iteration"] = np.array(range(0,data["strain"].size))
data["marker"] = None
data["linestyle"] = '-'
datalist = [data]


plot_stat(datalist, "strain", "stress", r"Average yy Strain", "Average yy Stress", "stress.svg")
plot_stat(datalist,  "iteration", "max_overforce", "Iteration", "Maximum Overforce", "max_overforce.svg")
plot_stat(datalist,  "iteration", "total_fracture_energy", "Iteration", "Total Fracture Energy", "fracture_energy.svg")
plot_stat(datalist,  "iteration", "total_elastic_energy", "Iteration", "Total Elastic Energy", "total_energy.svg")
plot_stat(datalist,  "iteration", "total_energy", "Iteration", "Total Energy", "total_elastic_energy.svg")


