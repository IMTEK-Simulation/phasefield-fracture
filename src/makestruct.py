import numpy as np
import json
class randomfield():

    def __init__(self, Lx=10, nx=127, lamb=1, sigma=0.3, mu=1, minimum_val=0, fname='teststruct.npy'):
        self.Lx=Lx
        self.nx=nx
        self.lamb=lamb
        self.sigma=sigma
        self.mu=mu
        self.minimum_val=minimum_val
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
            eta = eta_in
        else:
            eta = np.random.normal(0.0,1.0,size=(self.nx,self.nx))
        sigma_init = np.sqrt(np.sum(eta**2)/self.nx**2 - np.sum(eta)**2/self.nx**4)
        print('sigma_init=',sigma_init)
        qx = np.arange(0,self.nx, dtype=np.float64)
        qx = np.where(qx <= self.nx//2, qx/self.Lx, (self.nx-qx)/self.Lx)
        qx *= 2 * np.pi
        qy = np.arange(0,self.nx//2 +1, dtype=np.float64)
        qy*= 2*np.pi/self.Lx
        q2 = (qx**2).reshape(-1,1) + (qy**2).reshape(1,-1)
        filt = np.ones_like(q2)
        q_s = 2*np.pi/self.lamb
        filt *= (q2 <= q_s ** 2)
        h_qs = np.fft.irfftn( np.fft.rfftn(eta) * filt, eta.shape)
        sigma_filt = np.sqrt(np.sum(h_qs**2)*(1/self.nx)**2 - np.sum(h_qs)**2*(1/self.nx)**4)
        print('sigma_filt=',sigma_filt)
        print('mu_filt=',np.sum(h_qs)*(1/self.nx)**4)
        h_qs *= self.sigma/sigma_filt
        sigma_scaled = np.sqrt(np.sum(h_qs**2)*(1/self.nx)**2 - np.sum(h_qs)**2*(1/self.nx)**4)
        print('sigma_scaled=',sigma_scaled)
        h_qs += self.mu
        h_qs = np.maximum(h_qs,self.minimum_val)
        sigma_final = np.sqrt(np.sum(h_qs**2)*(1/self.nx)**2 - np.sum(h_qs)**2*(1/self.nx)**4)
        print('sigma_final=',sigma_final)
        return h_qs

    def makestruct2D(self):
        np.save(self.fname,self.struct2D())
        jsonfile = open("struct2D.json", mode='w')
        json.dump(self.__dict__,jsonfile,default=lambda o: "(array)")
        jsonfile.close()
