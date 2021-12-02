import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl


def pstresses(strains, phi_interp, Cx, Poisson=0.2, dim=2, model="spectral"):
    lamb_factor = Poisson/(1+Poisson)/(1-2*Poisson)
    mu_factor = 1/2/(1+Poisson)
    pstress1 = np.zeros_like(Cx)
    pstress2 = np.zeros_like(Cx)
    angle2 = np.zeros_like(Cx)
    for pixel, Cxval in np.ndenumerate(Cx):
        strain = np.reshape(strains[:,0,pixel[0],pixel[1]],(dim,dim))
        if (model == "spectral"):
            stress = stress_spectral(strain, lamb_factor, mu_factor,
                 phi_interp[pixel[0],pixel[1]], Cx[pixel[0],pixel[1]])
        if (model == "isotropic"):
            stress = stress_iso(strain, lamb_factor, mu_factor,
                 phi_interp[pixel[0],pixel[1]], Cx[pixel[0],pixel[1]])
        if (model == "voldev"):
            stress = stress_voldev(strain, lamb_factor, mu_factor,
                 phi_interp[pixel[0],pixel[1]], Cx[pixel[0],pixel[1]])
        pstress, vecs = np.linalg.eigh(stress)
        pstress1[pixel[0],pixel[1]] = pstress[1]
        pstress2[pixel[0],pixel[1]] = pstress[0]
        angle2[pixel[0],pixel[1]] = np.arctan(vecs[1,0]/vecs[0,0])
    return pstress1, pstress2, angle2

def stress_spectral(strain, lamb_factor, mu_factor, phi_interp, Cx):
    pstrains, vecs = np.linalg.eigh(strain)
    stress = np.zeros_like(strain)
    trace = pstrains.sum()
    if (trace > 0):
        tracefact = lamb_factor*phi_interp*trace
    else:
        tracefact = lamb_factor*trace
    for i in range(0,pstrains.size):
        if (pstrains[i] > 0):
            psfact = mu_factor*phi_interp*pstrains[i]
        else:
            psfact = mu_factor*pstrains[i]
        stress += np.outer(vecs[:,i],vecs[:,i])*(tracefact + 2*psfact)
    return stress*Cx

def stress_iso(strain, lamb_factor, mu_factor, phi_interp, Cx):
    stress = lamb_factor*np.trace(strain)*np.eye(strain.shape[0]) + 2*mu_factor*strain
    return stress*Cx*phi_interp

def stress_voldev(strain, lamb_factor, mu_factor, phi_interp, Cx):
    vol = np.eye(strain.shape[0])*np.trace(strain)
    if (vol[0,0] > 0):
        tracemat = (lamb_factor+2*mu_factor/3)*phi_interp*vol
    else:
        tracemat = (lamb_factor+2*mu_factor/3)*vol
    stress = tracemat + 2*mu_factor*((strain-vol)*phi_interp)
    return stress*Cx

def plot_stress(stress,outname, angle=None, phi=None, Cx=None, extent=None, arrowlen=4):
    nx = phi.shape[0]
    ny = phi.shape[1]
    if (nx < 2000):
        interp = 'None'
    else:
        interp = 'antialiased'
    fig = plt.figure(figsize=(4,4))
    fig.patch.set_visible(False)
    ax = fig.add_axes((0.1,0.1,0.85,0.85))
    axim = ax.imshow(stress, origin='lower', interpolation=interp)
    if (extent is not None):
        n_xlabels = 5
        x_labels = np.linspace(extent[0],extent[1],n_xlabels)
        x_positions = np.linspace(0,nx,n_xlabels)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        n_ylabels = 5
        y_labels = np.linspace(extent[2],extent[3],n_ylabels)
        y_positions = np.linspace(0,ny,n_ylabels)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
    if (Cx is not None):
        ax.contour(Cx,[1.0,1.0],colors='white')
    if (phi is not None):
        ax.contourf(Cx,[0.5,1.1],colors='red')
    if (angle is not None):
        maxinds = np.unravel_index(np.argmax(stress),(nx,ny))
        print(angle[tuple(maxinds)]*180/np.pi)
        vector = arrowlen*np.array([np.cos(angle[tuple(maxinds)]),
              np.sin(angle[tuple(maxinds)])])
        plt.quiver(maxinds[1],maxinds[0],vector[0],vector[1])
    
    fig.savefig(outname+'.pdf',dpi=300)
    cb = fig.colorbar(axim)
    ax.set_visible(False)
    fig.savefig(outname+'_cbar.svg')

if (__name__ == '__main__'):
    basedir =  '../test/'
    dat = nc.Dataset(basedir+'test.nc')
    n = 1
    phi = dat["phi"][n,...]
    phi_interp = (phi-1)**2
    Cx = dat["Cx"][n,...]
    strain = dat["strain"][n,...]
    sp1, sp2, sang = pstresses(strain, phi_interp,Cx,
        model="spectral")
    mask = np.where(abs(sp1) > abs(sp2))
    maxmagnitude = sp2
    maxmagnitude[mask] = sp1[mask]
    plot_stress(sp1.T,'spectral_stress', angle=sang.T, phi=phi.T,
        Cx=Cx.T, extent=[0,20,0,20])
    
    

