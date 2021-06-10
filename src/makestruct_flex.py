import numpy as np
import json
from scipy.special import erfinv

def wavelength_filter2D(field, lamb, sigma, hipass=False):
    nx = field.shape[0]
    measure = nx**2
    Lx = 1
    mu_init = np.sum(field)**2/measure
    sigma_init = np.sqrt(np.sum((field - mu_init)**2)/measure)
    print('sigma_init=',sigma_init)
    qx = np.arange(0,nx, dtype=np.float64)
    qx = np.where(qx <= nx//2, qx/Lx, (nx-qx)/Lx)
    qx *= 2 * np.pi
    qy = np.arange(0,nx//2 +1, dtype=np.float64)
    qy*= 2*np.pi/Lx
    q2 = (qx**2).reshape(-1,1) + (qy**2).reshape(1,-1)
    filt = np.ones_like(q2)
    q_s = 2*np.pi/lamb
    if (hipass is True):
        filt *= (q2 >= q_s ** 2)
    else:
        filt *= (q2 <= q_s ** 2)
    h_qs = np.fft.irfftn( np.fft.rfftn(field) * filt, field.shape)
    mu_filt = np.sum(h_qs)/measure
    sigma_filt = np.sqrt(np.sum((h_qs - mu_filt)**2)/measure)
    print('sigma_filt=',sigma_filt)
    print('mu_filt=',mu_filt)
    h_qs *= sigma/sigma_filt
    mu_scaled = np.sum(h_qs)/measure
    sigma_scaled = np.sqrt(np.sum((h_qs - mu_scaled)**2)/measure)
    print('sigma_scaled=',sigma_scaled)
    return h_qs
    
def smoothcutoff2D(field, minimum_val, k=10):
    measure = np.array(field.shape).prod()
    mu0 = np.sum(field)/measure
    print('mu0', mu0)
    print('cutval=', minimum_val-mu0)
    cutfield = half_sigmoid(field-mu0, minimum_val-mu0, k=k)
    mu_cutoff = np.sum(cutfield)/measure
    sigma_cutoff = np.sqrt(np.sum((cutfield - mu_cutoff)**2)/measure)
    print('sigma_cutoff=',sigma_cutoff)
    print('minval_cutoff=',np.amin(cutfield)+mu0)
    return cutfield + mu0

def half_sigmoid(f, cutoff, k=10):
    x = np.asarray(f)
    y = np.asarray(x+0.0)
    y[np.asarray(x < 0)] = x[np.asarray(x < 0)]*abs(cutoff)/(
        abs(cutoff)**k+np.abs(x[np.asarray(x < 0)])**k)**(1/k)
    return y

def threshsymm(field, Vf):
    measure = np.array(field.shape).prod()
    mu = np.sum(field)/measure
    sigma = np.sqrt(np.sum((field-mu)**2/measure))
    thresh = 2**0.5*erfinv(2*Vf - 1)
    thresh_scaled = thresh*sigma + mu
    thresh_field = np.ones_like(field)
    thresh_field[field < thresh_scaled] = -1
    print(np.sum(thresh_field)/measure)
    return thresh_field

def threshmatrix(field, Vf):
    measure = np.array(field.shape).prod()
    vfl = 0.5-Vf/2
    vfu = 0.5+Vf/2
    print(vfl, vfu)
    threshL = 2**0.5*erfinv(2*vfl - 1)
    threshU = 2**0.5*erfinv(2*vfu - 1)
    print(threshL, threshU)
    mu = np.sum(field)/measure
    sigma = np.sqrt(np.sum((field-mu)**2/measure))
    thresh_field = np.ones_like(field)
    threshscL = threshL*sigma + mu
    thresh_field[field < threshscL] = -1
    threshscU = threshU*sigma + mu
    thresh_field[field > threshscU] = -1
    print(np.sum(thresh_field)/measure)
    return thresh_field

def ACsmooth2D(field, nits, ACwidth=2**0.5):
    a = field+0.0
    # f=(W/4)*(1-a)^2*(1+a)^2
    # da/dx = 2*sqrt(f)/e
    # max da/dx = 2sqrt(W/4*1^4)/e = sqrt(W)/e
    # L = delta a/(max da/dx) = 2/(sqrt(W)/e) = 2e/sqrt(W)
    # use W = 1, e^2 = (ACw/2)^2 -> L = 2*ACw/2/sqrt(1) = ACw
    for n in range(0,nits):
        a -= 0.05*(0.5*(2*a**3-2*a) + (ACwidth/2)**2*(4*a - np.roll(a,1,axis=0) - np.roll(a,-1,axis=0)
                  - np.roll(a,1,axis=1) - np.roll(a,-1,axis=1)))
    return a

def save_params(propdict, fname="struct2D.json"):
    jsonfile = open(fname, mode='w')
    json.dump(propdict,jsonfile,default=lambda o: "(array)")
    jsonfile.close()
