#
# Copyright 2021 W. Beck Andrews
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import sys
# append src directory to path
sys.path.append("../src")
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
        mechanics_formulation=mechanics.nonvar_both_tc(),
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

f.Cx.array()[...] = f.initialize_serial('teststruct.npy')*f.Young
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
sim.strain_step_tensor = np.array([[0,0.5],[0.5,0.0]])
f.F_tot = sim.strain_step_tensor*sim.strain_step_scalar
sim.run_simulation()
endt = time.time()

if(f.comm.rank == 0):
    print('total walltime: ',endt-startt)
    jsonfile = open("runobj.json", mode='w')
    json.dump(f.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.close()

