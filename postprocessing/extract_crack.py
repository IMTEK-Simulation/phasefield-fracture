import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import sys


print(sys.argv)
if (len(sys.argv) == 1):
    rundir = ''
else:
    rundir = sys.argv[1]


Lx = 20
nx = 63
Ly = Lx
ny = nx

fname = rundir+'test.nc'
ds = nc.Dataset(fname)
print(ds['phi'].shape[0])
for j in range(0,ds['phi'].shape[0]):
    print(j, np.max(ds['phi'][j,...]))
    print(np.unravel_index(np.argmax(ds['phi'][j,...]), (nx,ny)))
    if(np.max(ds['phi'][j,...]) > 0.5):
        startpoint = np.unravel_index(np.argmax(
            ds['phi'][j,...]), (nx,ny))
        break

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
# main crack
x_main = []
y_main = []
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
            y_main.append(toty/phisum*(Ly/ny))
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
        # the x coordinate is the same for all of them
        if (len(x_list) > 1):
            distmain = np.abs(y_main[-1]-y_main[-2])
            dist2 = np.abs(y_list2[-1]-y_main[-2])
            dist3 = np.abs(y_list3[-1]-y_main[-2])
            if (dist2 < distmain):
                y_main[-1] = y_list2[-1]
            if ((dist3 < distmain) and (dist3 < dist2)):
                y_main[-1] = y_list3[-1]

x_list.append(x_list[0]+Lx)
y_list.append(y_list[0])
y_list2.append(y_list2[0])
y_list3.append(y_list3[0])
y_main.append(y_main[0])

fig2 = plt.figure(dpi=600)
ax2 = fig2.add_axes([0.16, 0.15, 0.65, 0.78])
ax2.plot(x_list,y_list3,'-',label=r"upper crack")
ax2.plot(x_list,y_list,'-',label=r"lower crack")
ax2.plot(x_list,y_main,':',label=r"main crack")
ax2.plot(startpoint[0]*(Lx/nx),startpoint[1]*(Ly/ny),'mo',label=r"initiation point")

plt.legend(loc="upper right")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.savefig(rundir+'crackpath.png')

data_xy =  np.column_stack((np.array(x_list), np.array(y_main)))

np.save(rundir+"crackpath.npy",data_xy)
np.savetxt(rundir+"crackpath.xyz", data_xy)

data_xy_upper =  np.column_stack((np.array(x_list), np.array(y_list3)))
if (np.not_equal(data_xy,data_xy_upper).any):
    np.save(rundir+"crackpath_upper.npy", data_xy_upper)
    np.savetxt(rundir+"crackpath_upper.xyz", data_xy_upper)

data_xy_lower =  np.column_stack((np.array(x_list), np.array(y_list3)))
if (np.not_equal(data_xy,data_xy_upper).any):
    np.save(rundir+"crackpath_upper.npy", data_xy_upper)
    np.savetxt(rundir+"crackpath_upper.xyz", data_xy_upper)

def dataslice(data, startx, frac):
    Lx = data[-1,0] - data[0,0]
    datcopy = data+0.0
    datcopy[:,0] += 20
    datbig = np.vstack((data[0:-1,:],datcopy))
    dist = Lx*frac/2
    if (startx < dist):
        startx += Lx
    datblank = np.zeros_like(datbig)
    inds = np.intersect1d(np.where(startx-dist < datbig[:,0]),
                np.where(startx+dist > datbig[:,0]))
    datblank[inds,:] = datbig[inds,:]
    return np.column_stack((np.trim_zeros(datblank[:,0]),
        np.trim_zeros(datblank[:,1])))

data50 = dataslice(data_xy, startpoint[0], 0.50)
np.save(rundir+"crackpath50.npy",data50)
np.savetxt(rundir+"crackpath50.xyz", data50)

data75 = dataslice(data_xy, startpoint[0], 0.75)
np.save(rundir+"crackpath75.npy",data75)
np.savetxt(rundir+"crackpath75.xyz", data75)
