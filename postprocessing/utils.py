import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_mpl_params():
    mpl.style.use('classic')
    fm = mpl.font_manager.json_load("/home/fr/fr_fr/fr_wa1005/.cache/matplotlib/fontlist-v330.json")
    fm.findfont("Arial", rebuild_if_missing=False)
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["figure.figsize"] = 3, 2.25
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["figure.facecolor"] = 'w'
    mpl.rcParams["legend.loc"] = 'upper left'
    mpl.rcParams["legend.fontsize"] = 'medium'
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.numpoints"] = 3
    mpl.rcParams["image.cmap"] = 'viridis'
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["lines.markersize"] = 3
    mpl.rcParams["lines.markeredgewidth"] = 0.5

def grad2_serial(phi,Lx):
    nx = phi.shape[0]
    freqs = np.fft.fftfreq(nx)*(nx/Lx)
    diff1op = (-1)**(0.5)*2*np.pi*np.tile(np.expand_dims(freqs,1),nx)
    diff2op = diff1op.T
    print(diff1op.shape)
    diff1 = np.fft.ifftn(diff1op*np.fft.fftn(phi), phi.shape)
    diff2 = np.fft.ifftn(diff2op*np.fft.fftn(phi), phi.shape)
    print(np.sum(diff1**2), np.sum(diff2**2))
    grad2 = np.zeros_like(phi)
    grad2 += np.real(abs(diff1)**2)
    grad2 += np.real(abs(diff2)**2)
    print(np.max(np.max(grad2)))
    return grad2

def fracture_energy_serial(phi,Lx):
    energy = np.zeros_like(phi)
    energy += grad2_serial(phi,Lx)
    print('grad term:', np.sum(energy)*(Lx/phi.shape[0])**2)
    energy += phi
    print('phi term:', np.sum(phi)*(Lx/phi.shape[0])**2)
    energy *= (3/8)
    return energy

def fracture_energy_line(phi,init_site,Lx):
    nx = phi.shape[0]
    halfnx = int((nx+1)/2)
    shift = (halfnx-init_site[0], halfnx-init_site[1])
    centphi = np.roll(phi,shift,axis=(0,1))
    line = np.sum(fracture_energy_serial(centphi,Lx)*Lx/nx, axis=1)
    print(np.sum(line)*(Lx/nx))
    return line

def analytical_1D(x):
    analytical = np.zeros_like(x)
    analytical[abs(x) < 2] = (1-abs(x[abs(x) < 2])/2)**2
    return analytical

def find_init(ds, threshhold):
    nx = ds['phi'].shape[1]
    for j in range(0,ds['phi'].shape[0]):
        if(np.max(ds['phi'][j,...]) > threshhold):
            startpoint = np.unravel_index(np.argmax(ds['phi'][j,...]),
                (nx,nx))
            break
    print('primary initiation at ', startpoint)
    # experimental bit to find secondary site
    halfnx = (nx+1)/2
    shift = (halfnx - startpoint[0], halfnx-startpoint[1])
    phi = np.roll(ds['phi'][j,...], shift,axis=(0,1))  + 0.0
    radius = 8
    phi[halfnx-radius:halfnx+radius,halfnx-radius:halfnx+radius] = 0.0
    secondary = np.argmax(phi)
    if(phi[secondary] > 0.5):
        print('secondary initiation at ', secondary, ' with phi=',
            phi[secondary])
    else:
        print('no secondary initiation, maximum of phi=', phi[secondary],
            ' at ', secondary)
    return startpoint, secondary


