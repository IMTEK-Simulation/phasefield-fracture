import sys
sys.path.append("/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/quasistatic-parallel-2D/src")
import makestruct
import mechanics
import model_components
import parallel_fracture
import simulation
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time


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


nx=63
Lx=20

f = parallel_fracture.parallel_fracture(Lx=Lx,nx=nx)
f.strain_step_tensor = np.array([[0,0],[0,1.0]])
f.solver_tol = 1e-6
f.title = 'test'
f.phi.array()[...] = init_crack(f)

structobj = makestruct.randomfield(Lx=Lx,nx=nx,lamb=2,sigma=0.3,mu=1,minimum_val=0)
#if(f.comm.rank == 0):
#    structobj.makestruct2D()
#f.comm.barrier()
#f.Cx.array()[...] = f.initialize_serial('teststruct.npy')*f.Young
#f.Cx.array()[...] = (1.0-init_crack(f))**2*f.Young
if(f.comm.rank == 0):
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

f.initialize_material()

startt = time.time()
sim = simulation.simulation(f)
sim.delta_energy_tol = 1e-4
sim.run_simulation()
endt = time.time()

if(f.comm.rank == 0):
    print('total walltime: ',endt-startt)
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

