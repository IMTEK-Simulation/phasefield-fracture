import sys
sys.path.append('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/quasistatic-parallel-2D/src')
#sys.path.append('/Users/andrews/code/muspectre_misc/parallel2D/src')
import parallel_fracture
import makestruct
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time

class statdump():

    def __init__(self, fname):
        self.subiterations = 0
        self.avg_strain = 0
        self.total_energy = 0
        self.elastic_energy = 0
        self.delta_phi = 0
        self.strain_time = 0
        self.phi_time = 0
        self.fname = fname
    
    def clear(self):
        if os.path.exists(self.fname):
            os.remove(self.fname)

    def dump(self):
        jsonfile = open(self.fname, mode='a+')
        json.dump(self.__dict__,jsonfile,default=lambda o: "(array)")
        jsonfile.write('\n')
        jsonfile.close()

def init_crack(obj):
    vals = np.zeros_like(obj.phi.array())
    for ind, val in np.ndenumerate(vals):
        coords = (np.array(ind) + np.array(obj.fftengine.subdomain_locations))*obj.dx
        val = 0.0
        distcoord = np.abs(coords - np.array(obj.lens)/2)
        distcoord[0] = max(distcoord[0]-1.0,0)
        dist = np.sqrt(np.sum(distcoord**2))  - obj.dx[0]/4.0
        if (dist < 2.0**0.5):
            val = (1.0-dist/2.0**0.5)**2
        if (dist < 0):
            val = 1.0
        vals[ind] = val
    return vals

def iteration(obj,statobj):
    delta_energy = 1.0
    delta_energy_old = 1e6
    energy_old = obj.total_energy + 0.0
    statobj.subiterations = 0
    statobj.strain_time = 0.0
    statobj.phi_time = 0.0
    
    while(delta_energy > obj.delta_energy_tol):
        start = time.time() 
        obj.strain_result = obj.strain_solver()
        straint = time.time() 
        obj.phi_solver()
        obj.total_energy = obj.objective(obj.phi.array())
        phit = time.time()
        delta_energy = abs(obj.total_energy-energy_old)
        energy_old = obj.total_energy + 0.0

        statobj.subiterations += 1
        statobj.strain_time += straint-start
        statobj.phi_time += phit-straint
        if(obj.comm.rank == 0):
            print('delta energy = ',delta_energy)
        if((delta_energy > delta_energy_old + obj.delta_energy_tol)
               and (obj.strain_step > 0.001) and (statobj.subiterations > 1)):
            obj.F_tot[1,1] -= obj.strain_step
            obj.strain_step /= 2
            obj.F_tot[1,1] += obj.strain_step
            statobj.subiterations = 1
            delta_energy_old = 1e6
            if (obj.comm.rank == 0):
                print('non-monotonicity of energy convergence detected, strain step reduced to ', obj.strain_step)
        else:
            delta_energy_old = delta_energy

def run_test(obj):
    nmax = 50
    obj.strain_step = 0.008
    strain_time = []
    phi_time = []
    subiterations = []
    obj.F_tot[1,1] = 0.0
    obj.phi_old = obj.phi.array() + 0.0
    fieldoutputname = 'test.nc'
    obj.muOutput(fieldoutputname,new=True)
    stats = statdump('stats.json',)
    if(obj.comm.rank == 0): 
        stats.clear()
    n = 0
    while (n < nmax):
        iteration(obj,stats)
        stats.avg_strain = obj.F_tot[1,1]
        stats.total_energy = obj.total_energy
        stats.delta_phi = obj.integrate(obj.phi.array()-obj.phi_old)
        stats.strain_energy = obj.integrate((1.0-obj.phi.array())**2*obj.straineng.array())
        strain_time.append(stats.strain_time)
        phi_time.append(stats.phi_time)
        subiterations.append(stats.subiterations)
        obj.phi_old = np.maximum(obj.phi.array(),obj.phi_old)
        if(obj.comm.rank == 0):
            print('strain: ', stats.avg_strain, 'energy: ',stats.total_energy,'delta phi',stats.delta_phi)
            stats.dump()
        obj.muOutput(fieldoutputname)
        #obj.crappyIO('fields'+str(n).rjust(2,'0'))
        if((stats.strain_energy < 1.0) and (n > 1)):
            break
        obj.F_tot[1,1] += obj.strain_step
        n += 1
    
    if(obj.comm.rank == 0):
        print('strain time:', np.sum(strain_time))
        print('phi time:', np.sum(phi_time))
        print('total subiterations:', np.sum(subiterations))

nx=127
Lx=20

f = parallel_fracture.parallel_fracture(Lx=Lx,nx=nx)
f.delta_energy_tol = 1e-8*f.lens[0]**2
f.solver_tol = 1e-10
f.title = 'test'
f.phi.array()[...] = init_crack(f)

structobj = makestruct.randomfield(Lx=Lx,nx=nx,lamb=2,sigma=0.3,mu=1,minimum_val=0)
if(f.comm.rank == 0):
    structobj.makestruct2D()
f.comm.barrier()
f.Cx.array()[...] = f.initialize_serial(structobj.fname)*f.Young
#f.Cx.array()[...] = (1.0-init_crack(f))**2*f.Young
if(f.comm.rank == 0):
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

startt = time.time()
run_test(f)
endt = time.time()

if(f.comm.rank == 0):
    print('total walltime: ',endt-startt)
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

