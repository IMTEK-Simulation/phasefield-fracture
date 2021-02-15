import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import sys


print(sys.argv)
if (sys.argv == None):
    rundir = ''
else:
    rundir = sys.argv[1]


Lx = 800
nx = 2047
Ly = Lx
ny = nx

fname = rundir+'test.nc'
ds = nc.Dataset(fname)
print(ds['phi'].shape[0])
for j in range(0,ds['phi'].shape[0]):
    print(np.max(ds['phi'][j,...]))
    print(np.unravel_index(np.argmax(ds['phi'][j,...]), (nx,ny)))

startpoint = np.unravel_index(np.argmax(ds['phi'][ds['phi'].shape[0]-2,...]), (nx,ny))

phi = ds['phi'][ds['phi'].shape[0]-1,...]

x = np.linspace(0,Lx,nx)
y = np.linspace(0,Lx,nx)
xmap, ymap = np.meshgrid(x,y)

fig1 = plt.figure(dpi=600)
ax1 = fig1.add_axes([0.16, 0.15, 0.65, 0.78])
ax1.contour(x,y,phi.T,[0.5])
ax1.plot(startpoint[0]*(Lx/nx),startpoint[1]*(Ly/ny),'mo',label=r"initiation point")
plt.axis('square')
plt.savefig(rundir+'contour.png')

# lower crack
x_list = []
y_list = []
# upper crack
y_list2 = []
# 2nd upper crack
y_list3 = []
for i in range(0,nx):
    toty = 0.0
    phisum = 0.0
    flag_single = 0
    for j in range(0,ny):
        if (phi[i,j] > 0.8):
            if (flag_single > 2):
                print("more than 3 crack branches at x-index "+str(i))
                exit()
            toty += j*phi[i,j]
            phisum += phi[i,j]
        if ((phisum > 0) and (phi[i,j] < 0.8) and (flag_single == 0)):
            flag_single = 1
            x_list.append(i*(Lx/nx))
            y_list.append(toty/phisum*(Ly/ny))
            y_list2.append(toty/phisum*(Ly/ny))
            y_list3.append(toty/phisum*(Ly/ny))
            toty = 0.0
            phisum = 0.0
        if ((phisum > 0) and (phi[i,j] < 0.8) and (flag_single == 1)):
            flag_single = 2
            y_list2[-1] = toty/phisum*(Ly/ny)
            y_list3[-1] = toty/phisum*(Ly/ny)
            toty = 0.0
            phisum = 0.0
        if ((phisum > 0) and (phi[i,j] < 0.8) and (flag_single == 2)):
            print("more than 2 crack branches at x-index "+str(i))
            flag_single = 3
            y_list3[-1] = toty/phisum*(Ly/ny)
            toty = 0.0
            phisum = 0.0

x_list.append(Lx)
y_list.append(y_list[0])
y_list2.append(y_list2[0])
y_list3.append(y_list3[0])

fig2 = plt.figure(dpi=600)
ax2 = fig2.add_axes([0.16, 0.15, 0.65, 0.78])
ax2.plot(x_list,y_list3,'-',label=r"upper crack")
ax2.plot(x_list,y_list,'-',label=r"lower crack")
ax2.plot(startpoint[0]*(Lx/nx),startpoint[1]*(Ly/ny),'mo',label=r"initiation point")

plt.legend(loc="upper right")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.savefig(rundir+'crackpath.png')

data_xy =  np.column_stack((np.array(x_list), np.array(y_list)))
np.save(rundir+"crackpath.npy",data_xy)
np.savetxt(rundir+"crackpath.xyz", data_xy)

data_xy3 =  np.column_stack((np.array(x_list), np.array(y_list3)))

if (np.not_equal(data_xy,data_xy3).any):
    np.save(rundir+"crackpath_upper.npy", data_xy3)
    np.savetxt(rundir+"crackpath_upper.xyz", data_xy3)

