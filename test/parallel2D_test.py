import sys
sys.path.append('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/quasistatic-parallel-2D/src')
#sys.path.append('/Users/andrews/code/muspectre_misc/parallel2D/src')
import parallel_fracture
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time

class statdump():

    def __init__(self, fname, avg_strain):
        self.subiterations = 0
        self.avg_strain = avg_strain
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

def iteration(obj,statobj):
    delta_energy = 1.0
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

def run_test(obj):
    nmax = 24
    avg_strain_all = np.linspace(0.05,0.073,num=nmax) #np.linspace(0.07,0.093,num=nmax)
    obj.strain_step = avg_strain_all[1] - avg_strain_all[0]
    if(obj.comm.rank == 0):
        print(avg_strain_all)
    
    delta_phi = np.zeros(nmax)
    total_energy = np.zeros(nmax)
    strain_time = np.zeros(nmax)
    phi_time = np.zeros(nmax)
    obj.phi_old = obj.phi.array() + 0.0
    fieldoutputname = 'test.nc'
    obj.muOutput(fieldoutputname,new=True)
    stats = statdump('stats.json',avg_strain_all[0])
    if(obj.comm.rank == 0): 
        stats.clear()
    for n in range(0,nmax):
        obj.F_tot[1,1] = avg_strain_all[n]
        iteration(obj,stats)
        
        stats.avg_strain = avg_strain_all[n]
        stats.total_energy = obj.total_energy
        stats.delta_phi = obj.integrate(obj.phi.array()-obj.phi_old)
        stats.strain_energy = obj.integrate((1.0-obj.phi.array())**2*obj.straineng.array())
        delta_phi[n] = stats.delta_phi
        total_energy[n] = stats.total_energy 
        strain_time[n] = stats.strain_time
        phi_time[n] = stats.phi_time

        obj.phi_old = np.maximum(obj.phi.array(),obj.phi_old)
        if(obj.comm.rank == 0):
            print('strain: ', avg_strain_all[n], 'energy: ',total_energy[n],'delta phi',delta_phi[n])
            stats.dump()
        obj.muOutput(fieldoutputname)
        #obj.crappyIO('fields'+str(n).rjust(2,'0'))
        if(stats.strain_energy < 1.0):
            break
    
    if(obj.comm.rank == 0):
        print('strain time:', np.sum(strain_time))
        print('phi time:', np.sum(phi_time))
        np.save('total_energy',total_energy)
        np.save('delta_phi',delta_phi)
        np.save('avg_strain',avg_strain_all)
        np.save('strain_time',strain_time)
        np.save('phi_time',phi_time)
    
f = parallel_fracture.parallel_fracture(Lx=10,nx=127)
f.delta_energy_tol = 1e-6
f.solver_tol = 1e-10
f.title = 'test'
#f.initialize_serial('delta_phi.npy')
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

