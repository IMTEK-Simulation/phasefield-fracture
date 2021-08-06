import sys
import time
import numpy as np
import os
import math
from mpi4py import MPI

import muSpectre as msp
import muFFT
import muGrid
import mechanics
import model_components
import constrainedCG as cCG

class parallel_fracture():
    def __init__(self, Lx = 10, nx = 63, cfield=None, mechanics_formulation=None, pfmodel=None, Poisson=0.0):
        self.dim = 2
        self.lens = [Lx, Lx] #simulation cell lengths
        self.domain_measure = np.array(self.lens).prod()
        self.nb_grid_pts  = [nx, nx] #number of grid points in each spatial direction
        self.dx = np.array([Lx/nx, Lx/nx])
        self.comm = MPI.COMM_WORLD
        self.Gc = 1.0
        self.ksmall = 0.0
        self.Young = 10000.0
        self.Poisson = Poisson
        self.lamb_factor = self.Poisson/(1+self.Poisson)/(1-2*self.Poisson)
        self.mu_factor = 1/2/(1+self.Poisson)
        
        self.F_tot = np.array([[ 0.0,  0.0],
                  [ 0.0 , 0.00]])

        self.fftengine = muFFT.FFT(self.nb_grid_pts,fft='fftwmpi',communicator=self.comm)
        self.fourier_buffer   = self.fftengine.register_fourier_space_field("fourier-space", 1)
        self.fourier_gradient = [msp.FourierDerivative(self.dim , i) for i in range(self.dim)]
        
        self.fc_glob = muGrid.GlobalFieldCollection(self.dim)
        self.fc_glob.initialise(self.nb_grid_pts,
                self.fftengine.nb_subdomain_grid_pts,
                self.fftengine.subdomain_locations)
        self.phi    = self.fc_glob.register_real_field("phi", 1)
        self.Cx     = self.fc_glob.register_real_field("Cx", 1)
        self.Cx.array()[...] = self.Young*np.ones(self.fftengine.nb_subdomain_grid_pts)
        self.strain = self.fc_glob.register_real_field("strain", self.dim*self.dim)
        self.straineng = self.fc_glob.register_real_field("straineng", 1)
        self.phi_old = self.phi.array() + 0.0
        self.cell = msp.Cell(self.nb_grid_pts, self.lens ,msp.Formulation.small_strain,
            self.fourier_gradient,fft='fftwmpi',communicator=self.comm)

        ## choosing components
        if (mechanics_formulation==None):
            self.mechform = mechanics.anisotropic_tc()
        else:
            self.mechform = mechanics_formulation

        self.interp = model_components.interp1(self.ksmall)
        if (pfmodel==None):
            self.bulk = model_components.AT1()
        else:
            self.bulk = pfmodel

        self.solver_CG_tol = 1e-4
        self.solver_abs_tol = 1e-6
        self.solver_rel_tol = 1e-6
        self.maxiter_cg = 40000
        self.dt = 1

    def initialize_material(self):
        self.material = self.mechform.initialize_material(self)
        self.cell.initialise()  #initialization of fft to make faster fft
        
    def initialize_serial(self,fname):
        newfield = np.load(fname)
        return newfield[self.fftengine.subdomain_locations[0]:
            self.fftengine.subdomain_locations[0]+self.fftengine.nb_subdomain_grid_pts[0],
            self.fftengine.subdomain_locations[1]:self.fftengine.subdomain_locations[1]
            +self.fftengine.nb_subdomain_grid_pts[1]]

    def strain_solver(self):            
        self.mechform.update_material(self)
        verbose = msp.Verbosity.Silent
        solver = msp.solvers.KrylovSolverCG(self.cell, self.solver_CG_tol, self.maxiter_cg, verbose)
        return msp.solvers.newton_cg(self.cell, self.F_tot, solver, self.solver_rel_tol, self.solver_abs_tol, verbose)
    
    def phi_solver(self):
        Jx = -self.jacobian(self.phi.array())
        solve = cCG.constrainedCG(Jx, self.hessp, Jx,
                                 self.phi_old - self.phi.array(), self.comm)
        return solve
    
    def phi_implicit_solver(self):
         Jx = -self.jacobian(self.phi.array()) - (self.phi.array() - self.phi_old)/self.dt
         solve = cCG.constrainedCG(Jx, self.implicit_hessp, Jx,
                                  self.phi_old - self.phi.array(), self.comm)
         return solve
    
    def laplacian(self,x):
        return_arr = np.zeros(self.fftengine.nb_subdomain_grid_pts)
        self.fftengine.fft(x, self.fourier_buffer)
        self.fftengine.ifft(-np.einsum('i...,i', self.fftengine.fftfreq**2, (2*np.pi/self.dx)**2)
            *self.fourier_buffer.array(), return_arr)
        return return_arr*self.fftengine.normalisation
        
    def grad2(self,x):
        self.fftengine.fft(x, self.fourier_buffer)
        return_arr = np.zeros(self.fftengine.nb_subdomain_grid_pts)
        tempfield  = np.zeros(self.fftengine.nb_subdomain_grid_pts)
        for d in range(0,self.dim):
            diffop = muFFT.FourierDerivative(self.dim, d)
            self.fftengine.ifft( diffop.fourier(self.fftengine.fftfreq)
                *self.dx[d]*self.fourier_buffer, tempfield)
            return_arr += tempfield**2
        return return_arr*self.fftengine.normalisation**2

    def integrate(self,f):
        return self.comm.allreduce(np.sum(f)*np.prod(self.dx),MPI.SUM)

    def max(self,f):
         return self.comm.allreduce(np.max(f),MPI.MAX)
    
    def value_at_max(self,f,g):
        maxval = self.max(f)
        maxg = 0.0
        maxind = np.argmax(f)
        maxloc = np.unravel_index(maxind, self.fftengine.nb_subdomain_grid_pts, order='C')
        if (math.isclose(np.max(f[maxloc]), maxval)):
            maxg = g[maxloc]
        maxg_all = self.max(maxg)
        return maxg_all
    
    def argmax(self,f):
        x = -np.ones(self.dim)
        maxind = np.argmax(f)
        maxloc = np.unravel_index(maxind, self.fftengine.nb_subdomain_grid_pts, order='C')
        maxval = self.max(f)
        if (math.isclose(np.max(f[maxloc]), maxval)):
            for i in range(0,self.dim):
                x[i] = self.fftengine.subdomain_locations[i] + maxloc[i]
        xall = np.zeros(self.dim)
        for i in range(0,self.dim):
            xall[i] = self.max(x[i])
        return xall
    
    def fracture_energy(self,x):
        return self.Gc/self.bulk.cw*(self.bulk.energy(x) + self.grad2(x))

    def energy_density(self,x):
        return (self.mechform.get_elastic_energy(self) + self.fracture_energy(x))

    def objective(self,x):
        return self.integrate(self.energy_density(x))

    def jacobian(self,x):
        return (self.interp.jac(x)*self.straineng.array() +
               self.Gc/self.bulk.cw*(self.bulk.jac(x) - 2.0*self.laplacian(x)))

    def hessp(self,p):
        return (self.interp.hessp(p)*self.straineng.array() +
               self.Gc/self.bulk.cw*(self.bulk.hessp(p) - 2.0*self.laplacian(p)))

    def implicit_hessp(self,p):
        return (self.interp.hessp(p)*self.straineng.array() +
               self.Gc/self.bulk.cw*(self.bulk.hessp(p) - 2.0*self.laplacian(p)) + p/self.dt)

    def muOutput(self,fname,new=False):
        comm = muGrid.Communicator(self.comm)
        if (new):
            if(self.comm.rank == 0):
                if os.path.exists(fname):
                    os.remove(fname)
            file_io_object = muGrid.FileIONetCDF(
                fname, muGrid.FileIONetCDF.OpenMode.Write, comm)
        else:
            file_io_object = muGrid.FileIONetCDF(
                fname, muGrid.FileIONetCDF.OpenMode.Append, comm)
        file_io_object.register_field_collection(self.fc_glob)
        file_frame = file_io_object.append_frame()
        file_frame.write()
        file_io_object.close()

        

