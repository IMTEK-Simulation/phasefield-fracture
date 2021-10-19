#
# Copyright 2021 W. Beck Andrews
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def read_pandas(fname):
    f=open(fname)
    return pd.io.json.read_json(f,orient='columns',lines=True)

def stress_strain(base,xmax,ymax):
    timedep = read_pandas(base+'timedep/stats.json')
    nocontrol = read_pandas(base+'nocontrol/stats.json')
    altmin = read_pandas(base+'altmin/stats.json')
    fig, ax = plt.subplots(1,1, figsize=(3,2.25),dpi=300)
    ax.plot(np.append(np.zeros(1),timedep['strain']),
        np.append(np.zeros(1),timedep['stress']),'-',label='Near Equilibrium', dashes=(1, 1), color='tab:blue')
    ax.plot(np.append(np.zeros(1),altmin['strain']),
        np.append(np.zeros(1),altmin['stress']),'-',label='Alternating Min.', color='tab:orange')
    ax.plot(np.append(np.zeros(1),nocontrol['strain']),
        np.append(np.zeros(1),nocontrol['stress']),'-',label='Time-Dependent', dashes=(2, 1), color='tab:green')
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])
    plt.xlabel("Strain")
    plt.ylabel("Stress ($G_c/\ell$)")
    plt.legend(loc='upper right')
    plt.savefig('stress_strain.svg',dpi=600)

def fracture_iteration(base,xmax):
    timedep = read_pandas(base+'timedep/stats.json')
    nocontrol = read_pandas(base+'nocontrol/stats.json')
    altmin = read_pandas(base+'altmin/stats.json')
    fig, ax = plt.subplots(1,1, figsize=(3,2.25),dpi=300)
    ax.plot(timedep.index.values,timedep['total_fracture_energy'],
        '-',label='Near Equilibrium', dashes=(1, 1), color='tab:blue')
    ax.plot(altmin.index.values,altmin['total_fracture_energy'],
        '-',label='Alternating Min.', color='tab:orange')
    ax.plot(nocontrol.index.values,nocontrol['total_fracture_energy'],
        '-',label='Time-Dependent', dashes=(2, 1), color='tab:green')
    ax.set_xlim([0, 3100])
    plt.xlabel("Iteration")
    plt.ylabel("Fracture Energy ($G_c\ell^2$)")
    plt.legend()
    plt.savefig('fracture_iteration.svg',dpi=600)

if(__name__ == '__main__'):
    utils.set_mpl_params()
    stress_strain('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/combined/tension/point/',0.005,55)
    fracture_iteration('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/combined/tension/point/',3100)
    
