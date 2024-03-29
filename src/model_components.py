import numpy as np

class AT1():
    def __init__(self):
        self.name = "AT1"
        self.cw = 8.0/3.0
    
    def energy(self, x):
        return x
        
    def jac(self,x):
        return np.ones_like(x)
        
    def hessp(self,p):
        return np.zeros_like(p)
        
class AT2():
    def __init__(self):
        self.name = "AT2"
        self.cw = 2.0
        
    def energy(self, x):
        return x**2
        
    def jac(self,x):
        return 2*x
        
    def hessp(self,p):
        return 2*p
        

class interp1():
    def __init__(self, ksmall):
        self.name = "interp1"
        self.ksmall = ksmall
        
    def energy(self, x):
        return ((1.0-x)**2*(1-self.ksmall)+self.ksmall)
        
    def jac(self,x):
        return 2*(x-1.0)*(1-self.ksmall)
        
    def hessp(self,p):
        return 2*(1-self.ksmall)*p

