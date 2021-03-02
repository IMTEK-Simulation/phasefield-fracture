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
