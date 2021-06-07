import numpy as np
import json

def wavelength_filter2D(field, lamb, sigma, minimum_val):
    nx = field.shape[0]
    Lx = 1
    mu_init = np.sum(field)**2/nx**2
    sigma_init = np.sqrt(np.sum((field - mu_init)**2)/nx**2)
    print('sigma_init=',sigma_init)
    qx = np.arange(0,nx, dtype=np.float64)
    qx = np.where(qx <= nx//2, qx/Lx, (nx-qx)/Lx)
    qx *= 2 * np.pi
    qy = np.arange(0,nx//2 +1, dtype=np.float64)
    qy*= 2*np.pi/Lx
    q2 = (qx**2).reshape(-1,1) + (qy**2).reshape(1,-1)
    filt = np.ones_like(q2)
    q_s = 2*np.pi/lamb
    filt *= (q2 <= q_s ** 2)
    h_qs = np.fft.irfftn( np.fft.rfftn(field) * filt, field.shape)
    mu_filt = np.sum(h_qs)/nx**2
    sigma_filt = np.sqrt(np.sum((h_qs - mu_filt)**2)/nx**2)
    print('sigma_filt=',sigma_filt)
    print('mu_filt=',mu_filt)
    h_qs *= sigma/sigma_filt
    mu_scaled = np.sum(h_qs)/nx**2
    sigma_scaled = np.sqrt(np.sum((h_qs - mu_scaled)**2)/nx**2)
    print('sigma_scaled=',sigma_scaled)
    return h_qs
    
def smoothcutoff2D(field, minimum_val, k=10):
    nx = field.shape[0]
    mu0 = np.sum(field)/nx**2
    print('mu0', mu0)
    print('cutval=', minimum_val-mu0)
    cutfield = half_sigmoid(field-mu0, minimum_val-mu0, k=k)
    mu_cutoff = np.sum(cutfield)/nx**2
    sigma_cutoff = np.sqrt(np.sum((cutfield - mu_cutoff)**2)/nx**2)
    print('sigma_cutoff=',sigma_cutoff)
    print('minval_cutoff=',np.amin(cutfield)+mu0)
    return cutfield + mu0

def half_sigmoid(f, cutoff, k=10):
    x = np.asarray(f)
    y = np.asarray(x+0.0)
    y[np.asarray(x < 0)] = x[np.asarray(x < 0)]*abs(cutoff)/(
        abs(cutoff)**k+np.abs(x[np.asarray(x < 0)])**k)**(1/k)
    return y

class randomfield():

    def __init__(self, Lx=10, nx=127, lamb=1, sigma=0.3, mu=1, minimum_val=0, k=10, fname='teststruct.npy'):
        self.Lx=Lx
        self.nx=nx
        self.lamb=lamb
        self.sigma=sigma
        self.mu=mu
        self.minimum_val=minimum_val
        self.k = k
        self.fname=fname

    def struct3D(self):
        np.random.seed()
        eta = np.random.normal(0.0,1.0,size=(self.nx,self.nx,self.nx))    
        sigma_init = np.sum(eta**2)/self.nx**3 - np.sum(eta)**2/self.nx**6
        print('sigma_init=',sigma_init)
        qx = np.arange(0,self.nx, dtype=np.float64)
        qx = np.where(qx <= self.nx//2, qx/self.Lx, (self.nx-qx)/self.Lx)
        qx *= 2 * np.pi
        qy = qx
        qz = np.arange(0,self.nx//2 +1, dtype=np.float64)
        qz *= 2*np.pi/self.Lx
        q2 = (qx**2).reshape(1,-1,1)+(qx**2).reshape(-1,1,1)+(qz**2).reshape(1,1,-1)
        filt = np.ones_like(q2)
        q_s = 2*np.pi/self.lamb
        filt *= (q2 <= q_s ** 2)
        h_qs = np.fft.irfftn( np.fft.rfftn(eta) * filt)
        sigma_filt = np.sum(h_qs**2)*(1/self.nx)**3 - np.sum(h_qs)**2*(1/self.nx)**6
        print('sigma_filt=',sigma_filt)
        print('mu_filt=',np.sum(h_qs)*(1/self.nx)**6)
        h_qs *= self.sigma/sigma_filt
        sigma_scaled = np.sum(h_qs**2)*(1/self.nx)**3 - np.sum(h_qs)**2*(1/self.nx)**6
        print('sigma_scaled=',sigma_scaled)
        h_qs += self.mu
        h_qs = np.maximum(h_qs,self.minimum_val)
        sigma_final = np.sum(h_qs**2)*(1/self.nx)**3 - np.sum(h_qs)**2*(1/self.nx)**6
        print('sigma_final=',sigma_final)
        return h_qs
    
    def struct2D(self, eta_in=None):
        np.random.seed()
        if (eta_in is not None):
            eta = eta_in + 0.0
        else:
            eta = np.random.normal(0.0,1.0,size=(self.nx,self.nx))
        eta = wavelength_filter2D(eta, self.lamb/self.Lx, self.sigma, self.minimum_val)
        eta += self.mu
        smoothcutoff2D(eta, self.minimum_val,k=self.k)
        return eta

    def makestruct2D(self):
        np.save(self.fname,self.struct2D())
        jsonfile = open("struct2D.json", mode='w')
        json.dump(self.__dict__,jsonfile,default=lambda o: "(array)")
        jsonfile.close()
