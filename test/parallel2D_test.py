import sys
#sys.path.append('/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/fracture2D_quasistatic/src')
sys.path.append('/Users/andrews/code/muspectre_misc/parallel2D/src')
import parallel_fracture
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time

class perfstats():

    def __init__(self):
        self.total_walltime = 0
        self.total_iterations = 0
        self.total_subiterations = 0
        self.max_subiterations = 1
        self.strain_at_max_subits = 0
        self.total_strain_time = 0
        self.total_phi_time = 0
        self.max_strain_time = 0
        self.strain_at_max_strain_time= 0
        self.max_phi_time = 0
        self.strain_at_max_phi_time= 0
        
    def update_subiter(self,obj,num_subiterations, strain_time, phi_time):
        self.total_subiterations += num_subiterations
        if (num_subiterations > self.max_subiterations):
            self.max_subiterations = num_subiterations
            self.strain_at_max_subits = obj.F_tot[1,1]
        self.total_strain_time += strain_time
        if (strain_time > self.max_strain_time):
            self.max_strain_time = strain_time
            self.strain_at_max_strain_time = obj.F_tot[1,1]
        self.total_phi_time += phi_time
        if (phi_time > self.max_phi_time):
            self.max_phi_time = phi_time
            self.phi_at_max_subits = obj.F_tot[1,1]

def iteration(obj,perfobj):
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
        if(obj.comm.rank == 0):
            print('delta energy = ',delta_energy)
            print('strain solve time: ', straint-start)
            print('phi solve time: ', phit-straint)
            perfobj.update_subiter(obj,n,straint-start,phit-straint)
            jsonfile = open("perfobj.json", mode='w')
            json.dump(perfobj.__dict__,jsonfile,default=lambda o: "(array)")
            jsonfile.close()

def run_test(obj,perfobj):
    nmax = 21
    avg_strain_all = np.linspace(0.07,0.08,num=nmax)
    obj.strain_step = avg_strain_all[1] - avg_strain_all[0]
    if(obj.comm.rank == 0):
        print(avg_strain_all)
    delta_phi = np.zeros(nmax)
    total_energy = np.zeros(nmax)
    obj.phi_old = obj.phi.array() + 0.0
    for n in range(0,nmax):
        print('average yy strain:', avg_strain_all[n])
        obj.F_tot[1,1] = avg_strain_all[n]+0.0
        iteration(obj,perfobj)
        delta_phi[n] = obj.integrate(obj.phi.array()-obj.phi_old)
        total_energy[n] = obj.total_energy
        obj.phi_old = np.maximum(obj.phi.array(),obj.phi_old)
        if(obj.comm.rank == 0):
            print('energy: ',total_energy[n],'delta phi',delta_phi[n])
        obj.crappyIO('fields'+str(n).rjust(2,'0'))
        perfobj.total_iterations = n+1
 #   plot_comp_en(avg_strain_all,delta_phi,total_energy,obj.title)
    if(obj.comm.rank == 0):
        np.save('total_energy',total_energy)
        np.save('delta_phi',delta_phi)
        np.save('avg_strain',avg_strain_all)
    
    
def plot_comp_time(x,y1,y2,title):
    plt.style.use('beck_outside')
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.55, 0.78])
    ax.plot(x,y1,'o',label=r'Strain solver')
    ax.plot(x,y2,'s',label=r'Phase field solver')
    plt.xlabel(r'$\bar \varepsilon$')
    plt.ylabel(r'Walltime')
    plt.legend(loc='upper left',bbox_to_anchor=(0.65, 1))
    plt.savefig('timing.svg',format='svg')
    plt.close(fig)
    
def plot_comp_en(x,y1,y2,title):
    plt.style.use('beck_outside')
    fig = plt.figure()
    ax1 = fig.add_axes([0.15, 0.15, 0.55, 0.78])
    ax1.plot(x,y1,label=r'$\left<\Delta \phi\right>$')
    ax1.set_ylabel(r'$\left<\Delta \phi\right>$')
    ax2 = ax1.twinx()
    ax2.plot(x,y2,label=r'Total energy')
    plt.xlabel(r'$\bar \varepsilon$')
    ax2.set_ylabel(r'Total energy')
    plt.legend(loc='upper left',bbox_to_anchor=(0.65, 1))
    plt.savefig('evolution.svg',format='svg')
    plt.close(fig)
    
p = perfstats()
f = parallel_fracture.parallel_fracture()
f.penalty_coeff = 0
f.delta_energy_tol = 1e-4
f.phi_newton_tol = 1e-6
f.cg_tol = 1e-7
f.title = 'test'
if(f.comm.rank == 0):
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

startt = time.time()
run_test(f,p)
endt = time.time()

p.total_walltime = endt - startt
if(f.comm.rank == 0):
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()
    jsonfile = open("perfobj.json", mode='w')
    json.dump(p.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()
    
import matplotlib.pyplot as plt
if (f.comm.rank == 0):
    plt.pcolor(f.phi)
    #plt.pcolor(obj.straineng)
    plt.colorbar()
    plt.axis('square')
    plt.show()

