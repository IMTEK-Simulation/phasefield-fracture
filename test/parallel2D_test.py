import sys
#sys.path.append('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/fracture2D_quasistatic/src')
sys.path.append('/Users/andrews/code/muspectre_misc/parallel2D/src')
import parallel_fracture
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time

class statdump():

    def __init__(self, avg_strain):
        self.subiterations = 1
        self.avg_strain = avg_strain
        self.total_energy = 0
        self.delta_phi = 0
        self.strain_time = 0
        self.phi_time = 0
        
    def dump(self,fname):
        jsonfile = open(fname, mode='a+')
        json.dump(self.__dict__,jsonfile,default=lambda o: "(array)")
        jsonfile.write('\n')
        jsonfile.close()

def iteration(obj,statobj):
    delta_energy = 1.0
    energy_old = obj.total_energy + 0.0
    n = 0
    
    while(delta_energy > obj.delta_energy_tol):
        n += 1
        start = time.time() 
        obj.strain_result = obj.strain_solver()
        straint = time.time() 
        obj.phi_solver()
        obj.total_energy = obj.objective(obj.phi.array())
        phit = time.time()
        delta_energy = abs(obj.total_energy-energy_old)
        energy_old = obj.total_energy + 0.0
        statobj.strain_time += straint-start
        statobj.phi_time += phit-straint
        if(obj.comm.rank == 0):
            print('delta energy = ',delta_energy)

def run_test(obj):
    nmax = 41
    avg_strain_all = np.linspace(0.07,0.11,num=nmax)
    obj.strain_step = avg_strain_all[1] - avg_strain_all[0]
    if(obj.comm.rank == 0):
        print(avg_strain_all)
    
    delta_phi = np.zeros(nmax)
    total_energy = np.zeros(nmax)
    straintime = np.zeros(nmax)
    phitime = np.zeros(nmax)
    obj.phi_old = obj.phi.array() + 0.0

    for n in range(0,nmax):
        obj.F_tot[1,1] = avg_strain_all[n]+0.0
        stats = statdump(avg_strain_all[n])
        iteration(obj,stats)
        
        stats.total_energy = obj.total_energy
        stats.delta_phi = obj.integrate(obj.phi.array()-obj.phi_old)
        delta_phi[n] = stats.delta_phi
        total_energy[n] = stats.total_energy 
        
        obj.phi_old = np.maximum(obj.phi.array(),obj.phi_old)
        if(obj.comm.rank == 0):
            print('strain: ', avg_strain_all[n], 'energy: ',total_energy[n],'delta phi',delta_phi[n])
            stats.dump('stats.json')
        #obj.muIO()
        obj.crappyIO('fields'+str(n).rjust(2,'0'))
    
    if(obj.comm.rank == 0):
        np.save('total_energy',total_energy)
        np.save('delta_phi',delta_phi)
        np.save('avg_strain',avg_strain_all)
        np.save('straintime',straintime)
        np.save('phitime',phitime)
    
f = parallel_fracture.parallel_fracture()
f.delta_energy_tol = 1e-6
f.solver_tol = 1e-12
f.title = 'test'
if(f.comm.rank == 0):
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

startt = time.time()
run_test(f)
endt = time.time()

p.total_walltime = endt - startt
if(f.comm.rank == 0):
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

