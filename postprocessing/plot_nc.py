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


import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

Lx = 400
nx = 1023
x = np.linspace(0,Lx,nx)
y = np.linspace(0,Lx,nx)
xmap, ymap = np.meshgrid(x,y)

fname = './strong_scaling2/c08_2/test.nc'
ds = nc.Dataset(fname)
print(ds['phi'].shape)
print(np.sum(ds['straineng'][8,...])*(400/1023)**2)
exit()
#plt.pcolor(x,y,ds['phi'][10,...].T,shading='nearest')
plt.pcolor(x,y,ds['Cx'][10,...].T,shading='nearest')
plt.colorbar()
plt.contour(x,y,ds['phi'][10,...].T,[0.9])
plt.axis('square')
plt.savefig('testL400.png',dpi=600)
#plt.show()
