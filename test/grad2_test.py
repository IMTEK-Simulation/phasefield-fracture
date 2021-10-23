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
import muFFT

nx=5
Lx=2*np.pi

f = parallel_fracture.parallel_fracture(Lx=Lx,nx=nx,
        mechanics_formulation=mechanics.anisotropic_tc(),
        pfmodel=model_components.AT1(),Poisson=0.2,
        crackset=False)

f.title = 'grad2_test'

def init_analytical(obj):
    vals = np.zeros_like(obj.phi.array())
    for ind, val in np.ndenumerate(vals):
        coords = (np.array(ind) + np.array(obj.fftengine.subdomain_locations))*obj.dx
        vals[ind] = (np.sin(coords[0])*np.cos(coords[1]))**2 + (np.sin(coords[1])*np.cos(coords[0]))**2
    return vals

def init_test(obj):
    vals = np.zeros_like(obj.phi.array())
    for ind, val in np.ndenumerate(vals):
        coords = (np.array(ind) + np.array(obj.fftengine.subdomain_locations))*obj.dx
        vals[ind] = np.sin(coords[0])*np.sin(coords[1]) 
    return vals


grad2 = f.grad2(init_test(f))
analytical = init_analytical(f)

np.testing.assert_allclose(grad2, analytical, rtol=1e-07, atol=1e-07)
