import sys
# append src directory to path
sys.path.append("/work/ws/nemo/fr_wa1005-mu_test-0/quasistatic/combined/src")
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

nx=63
Lx=20

f = parallel_fracture.parallel_fracture(Lx=Lx,nx=nx,
        mechanics_formulation=mechanics.anisotropic_tc(),
        pfmodel=model_components.AT1(),Poisson=0.2,
        crackset=False)

f.title = 'test'

def init_crack(obj):
    vals = np.zeros_like(obj.phi.array())
    radius = 4
    for ind, val in np.ndenumerate(vals):
        coords = (np.array(ind) + np.array(obj.fftengine.subdomain_locations))*obj.dx
        val = 0.0
        distcoord = np.zeros_like(coords)
        distcoord[0] = max(np.abs(coords[0] - np.array(obj.lens[0])/2)-obj.lens[0]/4,0)
        distcoord[1] = max(np.abs(coords[1] - obj.lens[1]/2 - obj.dx[1]/2),0)
        dist = np.sqrt(np.sum(distcoord**2))
        if (dist < 2.0**0.5):
            val = (1.0-dist/2.0**0.5)**2
        vals[ind] = val
    return vals

f.phi.array()[...] = init_crack(f)

#f.Cx.array()[...] = f.initialize_serial('../noise1023.npy')*f.Young
f.initialize_material()
if(f.comm.rank == 0):
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

startt = time.time()
sim = simulation.simulation(f, time_dependent=True)
sim.overforce_lim=2
sim.strain_step_scalar = 0.0001
sim.min_its = 10
sim.strain_step_tensor = np.array([[0,0],[0,1]])
sim.run_simulation()
endt = time.time()

if(f.comm.rank == 0):
    print('total walltime: ',endt-startt)
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

