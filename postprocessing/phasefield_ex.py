import numpy as np
import matplotlib.pyplot as plt
import utils
import netCDF4 as nc

def get_phi(fname):
    dat = nc.Dataset(fname)
    n = dat['phi'].shape[0] - 1
    return dat['phi'][n,...]

def plot_G_x(base):
    Lx = 100
    timedep = get_phi(base+'timedep/test.nc')
    nocontrol = get_phi(base+'nocontrol/test.nc')
    altmin = get_phi(base+'altmin/test.nc')
    x = np.linspace(-Lx/2,Lx/2,timedep.shape[0])
    # this function returns the fracture energy G as a function of
    # the x coordinate (i.e., integrated in y/z)
    energy_line_td = utils.fracture_energy_line(timedep,[255,255],Lx)
    energy_line_am = utils.fracture_energy_line(altmin,[255,255],Lx)
    energy_line_nc = utils.fracture_energy_line(nocontrol,[255,255],Lx)
    fig, ax = plt.subplots(1,1, figsize=(3,2.25),dpi=200)
    ax.plot(x, energy_line_am,'-',label='Alternating Min.', color='tab:orange')
    ax.plot(x, energy_line_nc,'-',label='Time-Dependent', dashes=(2, 1), color='tab:green')
    ax.plot(x, energy_line_td,'-',label='Near Equilibrium', dashes=(1, 1), color='tab:blue')
    ax.set_xlim([0, Lx/2])
    plt.xlabel("$x$-coordinate")
    plt.ylabel("$G/G_c$")
    plt.legend(loc = 'upper left')
    plt.savefig('G_x.svg',dpi=600)

def plot_phi_y(base,xind, ycenter):
    Lx = 100
    timedep = get_phi(base+'timedep/test.nc')
    nocontrol = get_phi(base+'nocontrol/test.nc')
    altmin = get_phi(base+'altmin/test.nc')
    nx = timedep.shape[0]
    halfnx = int((nx+1)/2)
    phi_td = np.roll(timedep[xind,:], halfnx-ycenter)
    phi_nc = np.roll(nocontrol[xind,:], halfnx-ycenter)
    phi_am = np.roll(altmin[xind,:], halfnx-ycenter)
    y = np.linspace(-Lx/2,Lx/2,timedep.shape[0])
    y_analytic = np.linspace(-Lx/2,Lx/2,timedep.shape[0]*2+1)
    phi_analytic = utils.analytical_1D(y_analytic)
    fig, ax = plt.subplots(1,1, figsize=(3,2.25),dpi=200)
    ax.plot(y, phi_am,'o',label='Alternating Min.',color='None')
    ax.plot(y, phi_nc,'s',label='Time-Dependent',color='None')
    ax.plot(y, phi_td,'^',label='Near Equilibrium',color='None')
    ax.plot(y_analytic, phi_analytic,'-k',label='$(1-x/2)^2$',linewidth=0.5)
    ax.set_xlim([-4, 4])
    ax.set_ylim([0, 1.0])
    plt.xlabel("$y$-coordinate")
    plt.ylabel("$\phi$")
    plt.legend(bbox_to_anchor=(1.05, 0.8), loc='upper left')
    plt.savefig('phi_y.svg',dpi=600)

if(__name__ == '__main__'):
    utils.set_mpl_params()
    plot_G_x('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/combined/tension/point/')
    plot_phi_y('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/combined/tension/point/',510,257)

