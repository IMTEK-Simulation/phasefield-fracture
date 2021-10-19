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


import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import os
import utils
runname = os.path.basename(os.getcwd())

def plot_field(ds,outname, Lx, field='phi', contour=False, secondary=False, n=None):
    startpoint, secondary = utils.find_init(ds, 0.9)
    if(secondary):
        startpoint = secondary
    if(n == None):
        n = ds['phi'].shape[0]-1
    nx = ds['phi'].shape[1]
    if (nx < 1000):
        interp = 'None'
    else:
        interp = 'antialiased'
    halfnx = int((nx+1)/2)
    shift = (halfnx - startpoint[0], halfnx-startpoint[1])
    fig = plt.figure(figsize=(3,3))
    fig.patch.set_visible(False)
    ax = fig.add_axes((0,0,1,1))
    ax.set_axis_off()
    if(field == 'phi'):
        axim = ax.imshow(np.roll(ds['phi'][n,...],shift,axis=(0,1)).T,vmin=0.0,vmax=1.0, origin='lower')
        ticks = [0.0,0.5,1.0]
        ticklabels = ['{:1.1}'.format(tick) for tick in ticks] 
    elif(field == 'Cx'):
        axim = ax.imshow(np.roll(ds['Cx'][n,...],shift,axis=(0,1)).T, origin='lower')
        ticks = [np.min(ds['Cx'][n,...]),1e4,np.max(ds['Cx'][n,...])]
        ticklabels = ['{:4.2}'.format(tick) for tick in ticks] 
    else:
        print('other fields are not supported')
    if(contour):
        ax.contour(np.roll(ds['phi'][n,...],shift,axis=(0,1)).T,[0.9],colors='red')
    fig.savefig(outname+'.pdf',dpi=300)
    cb = fig.colorbar(axim)
    cb.set_ticks(ticks)
    if(field == 'phi'):
        cb.set_xticklabels(['{:1.1}'.format(tick) for tick in ticks])
    else:
        cb.ticklabel_format(style='sci',scilimits=(1e4,1e4))
    ax.set_visible(False)
    fig.savefig(outname+'_cbar.svg')
    
if(__name__ == '__main__'):
    utils.set_mpl_params()
    basedir =  '../random/grf_ln4_lambda6_L100_r1/nocontrol/'
    dat = nc.Dataset(basedir+'test.nc')
    plot_field(dat,'phi_plot_test', 100, field='phi')
    plot_field(dat,'Cx_plot_test', 100, field='Cx')
    plot_field(dat,'contour_plot_test', 100, field='Cx', contour=True)
