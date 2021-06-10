import mechanics
import model_components
import parallel_fracture
import numpy as np
import json
import time
import statlog


class simulation():
    
    def __init__(self, obj):
        self.obj = obj
        self.nmax = 400
        self.subit_outputname = 'altmin_subit.nc'
        self.fullit_outputname = 'test.nc'
        self.subit_statsname = 'stats_subit.json'
        self.fullit_statsname = 'stats_fullit.json'
        self.stats = statlog.full_iteration_stats(self.fullit_statsname)
        dummy_subit_stats = statlog.altmin_iteration_stats(self.subit_statsname, 0.00)
        if(obj.comm.rank == 0): 
            self.stats.clear()
            dummy_subit_stats.clear()
        self.domain_measure = np.array(obj.lens).prod()
        self.delta_energy_tol = 1e-5
        self.total_energy = self.obj.objective(self.obj.phi.array())
        self.delta_energy = 0.0
        self.delta_phi = 0.0
        self.energy_old = 0.0
        self.strain_step_tensor = np.array([[0,0],[0,1.0]])
        self.strain_step_scalar = 0.0002
        self.min_strain_step = 0.0002
        self.min_its = 40
        self.dt0 = 2**16
        self.dtmin = 2**(-8)
        self.dphidt = 0.2
        self.couplinglim = 1.5

    def avg_strain(self):
        return np.max(np.linalg.eigvals(self.obj.F_tot))
        
    def subiteration(self, IsImplicit=False):
        subit_stats = statlog.altmin_iteration_stats(self.subit_statsname, self.avg_strain())
        self.energy_old = self.total_energy
        start = time.time() 
        strain_solve = self.obj.strain_solver()
        straint = time.time()
        self.obj.mechform.get_elastic_coupling(self.obj, strain_solve)
        if(IsImplicit==False):
            phi_solve = self.obj.phi_solver()
        else:
            phi_solve = self.obj.phi_implicit_solver()
        self.obj.phi.array()[...] += phi_solve.result
        self.total_energy = self.obj.objective(self.obj.phi.array())
        phit = time.time()
        subit_stats.delta_energy = self.total_energy - self.energy_old
        subit_stats.delta_phi = self.obj.integrate(phi_solve.result)
        self.delta_energy = abs(subit_stats.delta_energy)
        self.delta_phi = abs(subit_stats.delta_phi)
        subit_stats.elastic_CG_its = strain_solve.nb_fev
        subit_stats.elastic_newton_its = strain_solve.nb_it
        subit_stats.elastic_time = straint-start
        subit_stats.phi_subits = phi_solve.n_iterations
        subit_stats.phi_time = phit-straint
        self.stats.subiteration_update(subit_stats)
        subit_stats.subiteration = self.stats.subiterations
        if(self.obj.comm.rank == 0):
            subit_stats.dump()
        
    def implicit_iterator(self):
        delta_energy_timestep = 1e8 
        n = 0
        self.obj.dt = self.dt0
        while True:
            n = n+1
            energy_old = self.total_energy
            self.delta_phi = 0.0
            while True:
                self.subiteration(IsImplicit=True)
                if((self.delta_phi > self.dphidt) and (self.obj.dt > self.dtmin)):
                    self.obj.dt /= 2
                    if(self.obj.comm.rank == 0):
                        print('decreasing timestep, delta phi = ', self.delta_phi)
                    self.obj.phi.array()[...] = self.obj.phi_old + 0.0
                else:
                    if ((self.delta_phi < self.dphidt/2) and (self.obj.dt < self.dt0)):
                        self.obj.dt *= 2
                    break
            couplingmax = self.obj.max(self.obj.interp.energy(self.obj.phi_old)*self.obj.straineng.array())
            if (couplingmax > self.couplinglim):
                self.obj.F_tot *= (self.couplinglim/couplingmax)**0.5
            if (self.obj.comm.rank == 0):
                print('couplingmax, ', couplingmax, ', dt = ', self.obj.dt)
                print('energy', self.total_energy, 'delta energy = ', self.delta_energy,
                   'delta phi = ', self.delta_phi)
            self.obj.phi_old = self.obj.phi.array() + 0.0
            self.stats.avg_strain = self.avg_strain()
            self.stats.total_energy = self.total_energy
            self.stats.delta_phi = self.obj.integrate(self.obj.phi.array()-self.obj.phi_old)
            self.stats.coupling_energy = self.obj.integrate(self.obj.straineng*
                self.obj.interp.energy(self.obj.phi.array()))
            if ((n % 10 == 0) and (n > 2)):
                if(self.obj.comm.rank == 0):
                    self.stats.dump()
                    print('saving implicit timestep # ', n,' with energy = ', self.total_energy,
                        'avg strain = ', self.stats.avg_strain)
                self.obj.muOutput(self.fullit_outputname)
            if ((self.delta_energy < self.delta_energy_tol) and 
                (self.obj.dt == self.dt0 )):
                if(self.obj.comm.rank == 0):
                    self.stats.dump()
                    print('saving implicit timestep # ', n,' with energy = ', self.total_energy,
                        'avg strain = ', self.stats.avg_strain)
                break


    def run_simulation(self):
        self.obj.F_tot = self.strain_step_scalar*self.strain_step_tensor
        self.obj.phi_old = self.obj.phi.array() + 0.0
        self.obj.muOutput(self.fullit_outputname,new=True)
        self.stats = statlog.full_iteration_stats(self.fullit_statsname)
        n = 0
        while (n < self.nmax):
            self.implicit_iterator()
            self.obj.muOutput(self.fullit_outputname)
            if(self.obj.comm.rank == 0):
                    self.stats.dump()
            if ((self.stats.coupling_energy < 2e-2*self.domain_measure) and
               (n > self.min_its)):
                break
            self.stats.iteration_reset()
            self.obj.F_tot += self.strain_step_tensor*self.strain_step_scalar
            n += 1
           
