from mpi4py import MPI
import sys
import time
import numpy as np

mu_build_path = "/home/fr/fr_fr/fr_wa1005/muspectre_stuff/builds/muspectre-20201130/build"
sys.path.append(mu_build_path + '/language_bindings/python')
sys.path.append(mu_build_path + '/language_bindings/libmufft/python')
sys.path.append(mu_build_path + '/language_bindings/libmugrid/python')

import muSpectre as msp
import muSpectre.vtk_export as vt_ex
import muSpectre.gradient_integration as gi
import muFFT
import muGrid


##################### Newton-CG algorithm ####################################
def parallelNewtonCG(x0,jacobian,hessp, comm, maxiter_newton = 100, newton_tol = 1e-8):
    # equation to solve iteratively: J(x) = 0
    # linear system: He(x_n)delta = - J(x_n)
    # update: x_{n+1} = x_n + delta
    x = x0
    for iternum_newton in range(maxiter_newton):
        Jx = -jacobian(x)
        residual = np.sqrt(comm.allreduce((Jx*Jx).sum(),MPI.SUM))
        if(comm.rank == 0):
            print('Newton residual at step {} is {}'.format(iternum_newton+1,residual))
        if (residual < newton_tol):
            break
        delta = parallelCG(Jx*1e-2,x,hessp, Jx, comm)
        x = x + delta
    return x

def parallelCG(delta,x,A,b, comm, maxiter_cg = 40000, cg_tol = 1e-8):
    # CG algorithm, copied shamelessly from the CG wikipedia page and parallelized
    # (delta is now the x from the wiki code, since x from the Newton iteration is
    # needed for the hessian)
    r = b - A(x,delta)
    p = r
    rsold = comm.allreduce((r*r).sum(),MPI.SUM)
    for iternum_cg in range(maxiter_cg):
        Ap = A(x,p)
        pAp = comm.allreduce((p*Ap).sum(),MPI.SUM)
        alpha = rsold/pAp
        delta = delta + alpha*p
        r = r - alpha*Ap
        rsnew = comm.allreduce((r*r).sum(),MPI.SUM)
        if (np.sqrt(rsold) < cg_tol):
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    if (comm.rank == 0):
        if (iternum_cg+1 < maxiter_cg):
            print('CG converged after {} iterations with residual {}'.format(iternum_cg+1, rsnew))
        else:
            print('CG reached max iterations without converging; returing anyway')
    return delta

############################## test problem class #################################
# solves a simple non-linear version of my phase field system with a constant strain energy
class testclass():
    def __init__(self, Lx = 10, nx = 83):
        self.dim = 2
        self.lens = [Lx, Lx] #simulation cell lengths
        self.nb_grid_pts  = [nx, nx] #number of grid points in each spatial direction
        self.dx = np.array([Lx/nx, Lx/nx])
        self.comm = MPI.COMM_WORLD

        self.fftengine = muFFT.FFT(self.nb_grid_pts,fft='fftwmpi',communicator=self.comm)
        self.fourier_buffer = self.fftengine.register_fourier_space_field("fourier-space", 1)
        self.straineng      = np.zeros(self.fftengine.nb_subdomain_grid_pts)
        self.x0             = np.zeros(self.fftengine.nb_subdomain_grid_pts)
        
        self.fc_glob = muGrid.GlobalFieldCollection(self.dim)
        print(self.fftengine.nb_subdomain_grid_pts)
        self.fc_glob.initialise(
            self.nb_grid_pts, self.fftengine.nb_subdomain_grid_pts)
        self.solution = self.fc_glob.register_real_field("solution", 1)
        self.initialize_parallel()
        
### initialization of 'strain energy' term
    def initialize_parallel(self):
        for ind, val in np.ndenumerate(self.straineng):
            coords = (np.array(ind) + np.array(self.fftengine.subdomain_locations))*self.dx
            self.straineng[ind] = self.init_function(coords)
        
    def init_function(self, coords):
        return 10*(np.cos(np.pi*coords[0]/self.lens[0])*np.cos(np.pi*coords[1]/self.lens[1]))**4

### key parts of test system
    def laplacian(self,x):
        return_arr = np.zeros(self.fftengine.nb_subdomain_grid_pts)
        self.fftengine.fft(x, self.fourier_buffer)
        self.fftengine.ifft(-np.einsum('i...,i', self.fftengine.fftfreq**2, (2*np.pi/self.dx)**2)
            *self.fourier_buffer.array(), return_arr)
        return return_arr*self.fftengine.normalisation

    def jacobian(self,x):
        return 4.0*(x-1.0)**3*self.straineng + x - self.laplacian(x)

    def hessp(self,x,p):
        return 12.0*(x-1.0)**2*self.straineng*p + p - self.laplacian(p)

### wrappers
    def solve(self):
        self.solution.array()[...] = parallelNewtonCG(self.x0,self.jacobian,self.hessp, self.comm)
  
  # currently broken, in the name of progress          
 #   def crappyIO(self):  ## will replace this with a real parallel IO at some point
 #       np.save('solution{:02d}'.format(self.comm.rank),self.solution)
        
    def muIO(self):
        file_io_object = muGrid.FileIONetCDF(
            "testfile.nc", muGrid.FileIONetCDF.OpenMode.Write, muGrid.Communicator(self.comm))
        file_io_object.register_field_collection(self.fc_glob)
        file_frame = file_io_object.append_frame()
        file_frame.write()
        file_io_object.close()

        
################### test for testclass ###########################
obj = testclass()
solution = obj.solve()
obj.muIO()

